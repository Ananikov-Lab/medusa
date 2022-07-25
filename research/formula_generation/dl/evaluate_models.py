import os
import argparse
from evaluate_model import load_data, evaluate_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dset_path', type=str, help='Path to the dataset')
    parser.add_argument('--models_path', type=str, help='Path to the models')
    parser.add_argument('--model_type', type=str, help='Type of the model')
    parser.add_argument('--task_type', type=str, help='Type of the task')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix value')

    args = parser.parse_args()

    data = load_data(args.dset_path, args.prefix)

    for root, dirs, files in os.walk(args.models_path):
        for file in files:
            if file.endswith('.ckpt'):
                model_path = os.path.join(root, file)
                model_name = file.split('.')[0]
                print('Evaluating model {}'.format(model_name))
                evaluate_model(model_path, args.model_type, data)
