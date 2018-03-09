import os

import pandas as pd

import common
from rnn import RNN
from cnn import CNN
from mlp import MLP
from xgb import XGB


class Bagging(object):

    models = {'rnn': RNN, 'cnn': CNN, 'mlp': MLP, 'xgb': XGB}

    random_seeds = {
        'rnn': [
            9144, 26871, 53072, 93732, 1895,
            49322, 62881, 20266, 63693, 64789,
            31227, 29511, 85081, 76077, 67604,
            96250, 29544, 2305, 48880, 93100,
        ],
        'cnn': [
            47353, 19168, 10183, 42201, 71124,
            13008, 1233, 81194, 317, 92014,
            20952, 24879, 56456, 313, 66088,
            57098, 41786, 28477, 48242, 69630,
        ],
        'mlp': [
            17020, 56990, 87365, 50084, 17100,
            82993, 70640, 36822, 58374, 90613,
            62663, 82821, 41931, 45652, 47827,
            66098, 30633, 6704, 98359, 80705,
        ],
        'xgb': [
            64554, 3155, 87476, 63627, 42140,
            20676, 79527, 5148, 60709, 97981,
            60271, 45670, 65522, 94692, 38212,
            79848, 47116, 21454, 70993, 6530,
        ],
    }

    params = {
        'rnn': {
            'vocab_size': 30000,
            'max_len': 300,
            'vectors': 'glove.42B.300d',
            'rnn_size': 500,
            'rnn_dropout': 0.2,
            'dense_layers': 1,
            'dense_dropout': 0.3,
            'batch_size': 128,
            'lr_high': 0.5,
            'lr_low': 0.01,
        },
        'cnn': {
            'vocab_size': 50000,
            'max_len': 400,
            'vectors': 'glove.42B.300d',
            'conv_blocks': 1,
            'conv_dropout': 0.1,
            'dense_layers': 1,
            'dense_dropout': 0.5,
            'batch_size': 256,
            'lr_high': 0.01,
            'lr_low': 0.001,
        },
        'mlp': {
            'vocab_size': 100000,
            'max_len': 600,
            'vectors': 'glove.42B.300d',
            'hidden_layers': 2,
            'hidden_units': 600,
            'input_dropout': 0.1,
            'hidden_dropout': 0.5,
            'batch_size': 512,
            'lr_high': 0.3,
            'lr_low': 0.1,
        },
        'xgb': {
            'vocab_size': 300000,
            'max_len': 1000,
            'min_df': 5,
            'learning_rate': 0.1,
            'max_depth': 6,
        },
    }

    def __init__(self, name):
        self.name = name
        self.output_dir = os.path.join(common.OUTPUT_DIR, 'bagging', self.name)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.train_output = os.path.join(self.output_dir, 'train.csv')
        self.test_output = os.path.join(self.output_dir, 'test.csv')

    def main(self):
        train_outputs = []
        test_outputs = []

        model_cls = self.models[self.name]
        for random_seed in self.random_seeds[self.name]:
            model = model_cls(self.name, self.params[self.name], random_seed)
            model.main()

            train_df = pd.read_csv(model.train_output)
            val_df = pd.read_csv(model.validation_output)
            train_output = pd.concat([train_df, val_df])
            train_outputs.append(train_output)

            test_df = pd.read_csv(model.test_output)
            test_outputs.append(test_df)

        train_df = pd.concat(train_outputs).groupby('id', as_index=False).mean()
        train_df.to_csv(self.train_output, index=False)

        test_df = pd.concat(test_outputs).groupby('id', as_index=False).mean()
        test_df.to_csv(self.test_output, index=False)


if __name__ == '__main__':
    for name in ['rnn', 'cnn', 'mlp', 'xgb']:
        model = Bagging(name)
        model.main()
