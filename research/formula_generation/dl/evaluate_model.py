import argparse
import os
import gzip
import logging
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from mass_automation.formula import Formula
from mass_automation.formula.model import LSTM, MLP
from mass_automation.formula.data import OriginalFormulaDataset, converter_mapper, PadSequence, PadSequenceConstant
from mass_automation.utils import ELEMENT_DICT
from mass_automation.plot import plot_periodic_table

MEANS = np.clip(np.array(
    [3.60372658e+01, 2.88056417e-05, 9.15330276e-03, 3.08382820e-04,
     6.18755445e-02, 3.00645962e+01, 3.51274824e+00, 5.16170597e+00,
     1.31945515e+00, 1.37611278e-05, 3.95996347e-02, 3.43262684e-03,
     5.76452632e-03, 1.50951594e-01, 1.68682814e-01, 6.85634851e-01,
     4.73159939e-01, 1.12804177e-04, 1.19415522e-02, 2.99758953e-03,
     2.41936243e-04, 4.67379903e-03, 1.62352400e-03, 2.47604656e-03,
     2.08865036e-03, 6.78629940e-03, 3.97799024e-03, 4.54592053e-03,
     7.37428898e-03, 6.03908859e-03, 1.48310850e-03, 4.28765640e-03,
     4.68633603e-03, 1.26135405e-02, 1.61566749e-01, 1.87759651e-05,
     1.38155802e-03, 6.59421494e-04, 1.96829415e-03, 3.79745616e-03,
     3.73575720e-04, 1.59218127e-03, 9.99999975e-06, 4.42681834e-03,
     1.35899126e-03, 3.66707053e-03, 2.12250045e-03, 8.80074338e-04,
     1.20478508e-03, 1.06351869e-02, 2.25790101e-03, 2.57885060e-03,
     7.80835003e-02, 5.51335397e-05, 9.01387422e-04, 1.13081618e-03,
     4.58827970e-04, 4.71365056e-04, 2.39428831e-04, 4.38768620e-04,
     9.99999975e-06, 3.48501519e-04, 3.73575720e-04, 5.75422950e-04,
     2.39428831e-04, 3.23427346e-04, 1.84265606e-04, 2.31906568e-04,
     1.10296751e-04, 4.07425861e-04, 2.00563838e-04, 7.17092131e-04,
     2.89577176e-04, 2.00339803e-03, 4.09933302e-04, 6.59421494e-04,
     1.49940676e-03, 4.61612828e-03, 1.84417679e-03, 2.18643970e-03,
     8.39955639e-04, 1.55206257e-03, 1.34143932e-03, 9.99999975e-06,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
     9.99999975e-06, 2.24384319e-04, 2.25370932e-05, 7.68494210e-04,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
     9.99999975e-06, 9.99999975e-06, 9.99999975e-06]
), 1e-2, 1)


def load_data(dset_path, prefix):
    if prefix is not None:
        logging.log(logging.INFO, 'Loading test dataset')
        with open(prefix + '_test_dset.pkl', 'rb') as f:
            test_dset = pkl.load(f)

        return test_dset

    formulas = []
    representations = []
    with gzip.open(dset_path, 'rt') as f:
        for line in tqdm(f):
            if (not line.strip()) or (line.strip() == 'None') or (line.strip() == 'killed') or '+' in \
                    line.split('\t', 1)[
                        0] or ('-' in line.split('\t', 1)[0]):
                continue

            splitted = line.strip().split('\t', 1)

            formulas.append(Formula(splitted[0]))
            representations.append(splitted[1])

    train_X, test_X, train_y, test_y = train_test_split(representations, formulas, random_state=42)

    test_dset = OriginalFormulaDataset(test_y,
                                       converter_mapper['vector'],
                                       test_X, None,
                                       normalizer='one', is_classifier=False)

    return test_dset


def evaluate_model(model_path, model_type, data, task_type):
    if model_type == 'LSTM':
        model = LSTM.load_from_checkpoint(model_path)
    elif model_type == 'MLP':
        model = MLP.load_from_checkpoint(model_path)
    else:
        raise ValueError('Unknown model type')

    model.eval()

    if not os.path.exists(model_path + '_results'):
        os.makedirs(model_path + '_results')

    n_formulas = len(data)
    y_hat = []
    y = []

    input_ = [data[i] for i in range(n_formulas)]

    for i in trange(0, len(input_), 4096):
        if model_type == 'LSTM':
            seq, y_ = PadSequence()(input_[i:i + 4096])
        else:
            seq, y_ = PadSequenceConstant(5)(input_[i:i + 4096])
            
        y.append(y_.cpu().detach().numpy())
        y_hat.append(model.forward(seq).cpu().detach().numpy())

    y_hat = np.vstack(y_hat)
    y = np.vstack(y)

    np.savez(os.path.join(model_path + '_results', 'y'), y)
    np.savez(os.path.join(model_path + '_results', 'y_hat'), y_hat)

    y_hat = y_hat * MEANS
    y = y * MEANS

    plt.figure(figsize=(10, 12))
    for _, (i, lim, alpha) in enumerate(
            zip(
                [0, 5, 6, 7, 14, 15, 16, 34, 45],
                [.75, .75, .25, .2, .05, 0.1, 0.1, 0.1, 0.1],
                [0.005, 0.005, 0.007, 0.006, 0.01, 0.01, 0.01, 0.01, 0.01]
            )):
        ax = plt.subplot(3, 3, _ + 1)

        low = np.percentile(y[:, i], 0.1) * 0.9
        high = np.percentile(y[:, i], 99.9) * 1.1

        n_labels = int(max(y[:, i]))
        plt.boxplot([y_hat[:, i][y[:, i] == j] for j in trange(int(max(y[:, i])))],
                    showfliers=False,
                    positions=list(range(n_labels)),
                    whiskerprops=dict(color='gray'),
                    capprops=dict(color='gray'),
                    boxprops=dict(color='gray'),
                    medianprops=dict(color='dodgerblue')
                    )
        plt.xlim(low - .5, high + .5)
        plt.ylim(low - .5, high + .5)

        for ind, label in enumerate(ax.get_xticklabels()):
            if ind % (high // 5) < 1 or high < 5:  # every 10th label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)

        for ind, tick in enumerate(ax.get_xticklines()):
            if ind % (high // 5) < 1 or high < 5:  # every 10th label is kept
                tick.set_visible(True)
            else:
                tick.set_visible(False)

        plt.ylabel('Neural network prediction')
        plt.xlabel('Groud truth')

        plt.title(f'{ELEMENT_DICT[i + 1]} \t $R^2=${r2_score(y[:, i], y_hat[:, i]):.3f}')

    plt.subplots_adjust(wspace=.3, hspace=.5)

    plt.savefig(os.path.join(model_path + '_results', 'regressor_box.pdf'), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 12))
    for _, (i, lim, alpha) in enumerate(
            zip(
                [0, 5, 6, 7, 14, 15, 16, 34, 45],
                [.75, .75, .25, .2, .05, 0.1, 0.1, 0.1, 0.1],
                [0.005, 0.005, 0.007, 0.006, 0.01, 0.01, 0.01, 0.01, 0.01]
            )):
        plt.subplot(3, 3, _ + 1)

        low = np.percentile(y[:, i], 0.1) - .5
        high = np.percentile(y[:, i], 99.9) + .5

        slice_ = ((y[:, i] >= low) & (y[:, i] <= high)) & ((y_hat[:, i] >= low) & (y_hat[:, i] <= high))
        plt.hexbin(y[slice_, i], y_hat[slice_, i], gridsize=50, mincnt=1, cmap='Blues', bins='log')

        plt.xlim(low, high)
        plt.ylim(low, high)

        plt.colorbar()

        plt.ylabel('Neural network prediction')
        plt.xlabel('Groud truth')

        r2_score_value = r2_score(y[:, i], y_hat[:, i])
        rmse_value = (mean_squared_error(y_hat[:, i], y[:, i]))

        plt.title(f'{ELEMENT_DICT[i + 1]}: $R^2=${r2_score_value:.3f} $RMSE=${rmse_value:.2f}')

    plt.subplots_adjust(wspace=.5, hspace=.4)
    plt.savefig(os.path.join(model_path + '_results', 'regressor_scatter.pdf'), dpi=300, bbox_inches='tight')

    r2_score_values = [r2_score(y[:, i], y_hat[:, i]) if np.sum(y[:, i]) > 100 else None for i in range(119)]

    errors = dict(zip(
        np.arange(1, 121),
        r2_score_values
    ))

    plot_periodic_table(errors, os.path.join(model_path + '_results', 'results.html'), show_mono=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dset_path', type=str, help='Path to the dataset')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--model_type', type=str, help='Type of the model')
    parser.add_argument('--task_type', type=str, help='Type of the task')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix value')

    args = parser.parse_args()

    data = load_data(args.dset_path, args.prefix)

    evaluate_model(args.model_path, args.model_type, data, args.task_type)
