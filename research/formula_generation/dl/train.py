import argparse
import gzip
import gc
import pickle as pkl

import yaml
import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from mass_automation.formula.model import LSTM, MLP
from mass_automation.formula.data import OriginalFormulaDataset, converter_mapper, PadSequence, PadSequenceConstant
from mass_automation.formula.augmentations import AugmentationWrapper, Shift, Add, RandomNoise, Scale, DroppingAug
from utils import flatten

from mass_automation.formula import Formula

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str)
parser.add_argument('--sample', type=float, default=1.0)
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--save_prefix', type=str, default='')

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

checkpoint_callback_val = ModelCheckpoint(
    monitor='val_loss',
    filename='best_val_{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}',
    save_top_k=5,
    mode='min',
)

checkpoint_callback_train = ModelCheckpoint(
    monitor='train_loss',
    filename='best_train_{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}',
    save_top_k=5,
    mode='min',
)

trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[checkpoint_callback_train, checkpoint_callback_val])

with open(hparams.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if not hparams.prefix:
    formulas = []
    representations = []
    with gzip.open(config['data']['representations'], 'rt') as f:
        for line in tqdm(f):
            if (not line.strip()) or (line.strip() == 'None') or (line.strip() == 'killed') or '+' in \
                    line.split('\t', 1)[
                        0] or ('-' in line.split('\t', 1)[0]):
                continue

            splitted = line.strip().split('\t', 1)

            formulas.append(Formula(splitted[0]))
            representations.append(splitted[1])

    train_X, test_X, train_y, test_y = train_test_split(representations, formulas, random_state=42)

    if hparams.sample != 1.0:
        selection = np.random.choice(np.arange(0, len(train_X)), int(len(train_X) * hparams.sample), replace=False)
        train_X = [train_X[i] for i in selection]
        train_y = [train_y[i] for i in selection]

    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, random_state=42)

    train_augmentations = AugmentationWrapper(
        Shift(config['augmentation']['shift']),
        Add(config['augmentation']['add']),
        RandomNoise(config['augmentation']['random_noise']),
        Scale(config['augmentation']['scale']),
        DroppingAug(*config['augmentation']['dropping_aug'])
    )

    train_dset = OriginalFormulaDataset(train_y,
                                        converter_mapper[config['data']['converter']],
                                        train_X, train_augmentations, config['data']['is_classifier'],
                                        **config['data']['converter_settings'])

    val_dset = OriginalFormulaDataset(val_y,
                                      converter_mapper[config['data']['converter']],
                                      val_X, train_augmentations, config['data']['is_classifier'],
                                      **config['data']['converter_settings'])

    test_dset = OriginalFormulaDataset(test_y,
                                       converter_mapper[config['data']['converter']],
                                       test_X, None, config['data']['is_classifier'],
                                       **config['data']['converter_settings'])

    del train_X, val_X, test_X, train_y, val_y, test_y, formulas, representations
    gc.collect()

    if hparams.save_prefix:
        logging.log(logging.INFO, 'Saving to {}'.format(hparams.save_prefix))

        logging.log(logging.INFO, 'Saving train dataset')
        with open(hparams.save_prefix + '_train_dset.pkl', 'wb') as f:
            pkl.dump(train_dset, f)

        logging.log(logging.INFO, 'Saving validation dataset')
        with open(hparams.save_prefix + '_val_dset.pkl', 'wb') as f:
            pkl.dump(val_dset, f)

        logging.log(logging.INFO, 'Saving test dataset')
        with open(hparams.save_prefix + '_test_dset.pkl', 'wb') as f:
            pkl.dump(test_dset, f)

else:
    logging.log(logging.INFO, 'Loading from {}'.format(hparams.prefix))

    logging.log(logging.INFO, 'Loading train dataset')
    with open(hparams.prefix + '_train_dset.pkl', 'rb') as f:
        train_dset = pkl.load(f)

    logging.log(logging.INFO, 'Loading val dataset')
    with open(hparams.prefix + '_val_dset.pkl', 'rb') as f:
        val_dset = pkl.load(f)

    logging.log(logging.INFO, 'Loading test dataset')
    with open(hparams.prefix + '_test_dset.pkl', 'rb') as f:
        test_dset = pkl.load(f)

padder = PadSequence() if config['model_type'] == 'LSTM' else PadSequenceConstant(config['get_first'])

train_loader = DataLoader(
    train_dset, batch_size=config['training']['bs'], collate_fn=padder, num_workers=0, shuffle=True, pin_memory=True
)

val_loader = DataLoader(
    val_dset, batch_size=config['training']['bs'], collate_fn=padder, num_workers=0, pin_memory=True
)

test_loader = DataLoader(
    test_dset, batch_size=config['training']['bs'], collate_fn=padder, num_workers=0
)

if config['model_type'] == 'LSTM':
    model = LSTM(**flatten(config))
elif config['model_type'] == 'MLP':
    model = MLP(**flatten(config))
else:
    raise ValueError('Unknown model type {}'.format(config['model_type']))

trainer.fit(
    model,
    train_dataloader=train_loader,
    val_dataloaders=val_loader,
)

trainer.test(model, test_loader)
