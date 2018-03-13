import os
import sys
import logging
import pprint
import functools
import itertools
from datetime import datetime

import numpy as np
from numpy.random import RandomState
from sklearn.metrics import roc_auc_score
import pandas as pd
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import common
import base
from rnn import RNN
from cnn import CNN
from mlp import MLP
from xgb import XGB


logger = logging.getLogger(__name__)


RANDOM_SEED = 3468526


class StackingModule(nn.Module):

    def __init__(self, input_size, hidden_units_factor, hidden_layers, hidden_nonlinearily,
                 input_dropout=0, hidden_dropout=0):
        super().__init__()

        self.dense = base.Dense(
            input_size, 1,
            output_nonlinearity='sigmoid',
            hidden_layers=[hidden_units_factor * input_size] * hidden_layers,
            hidden_nonlinearity=hidden_nonlinearily,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout)

    def forward(self, x):
        return self.dense(x)


class Stacking(object):

    model_cls = {'rnn': RNN, 'cnn': CNN, 'mlp': MLP, 'xgb': XGB}

    model_params = {
        'rnn': {
            'vocab_size': 30000,
            'max_len': 300,
            'vectors': 'glove.42B.300d',
            'rnn_size': 500,
            'rnn_dropout': 0.2,
            'dense_layers': 1,
            'dense_dropout': 0.5,
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

    def __init__(self, params, random_seed):
        self.params = params
        self.random_seed = random_seed

        self.output_dir = os.path.join(
            common.OUTPUT_DIR,
            'stacking', str(self.random_seed),
            common.params_str(self.params))
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def main(self):
        t_start = datetime.now()
        logger.info(' stacking / {} '.format(self.random_seed).center(62, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(os.path.join(self.output_dir, 'test.csv')):
            logger.info('Output already exists - skipping')
            return

        self.random_state = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))
        torch.manual_seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))

        test_df = common.load_data('test')
        train_df = common.load_data('train')

        folds = common.stratified_kfold(train_df, random_seed=self.random_seed)
        for fold_num, train_ids, val_ids in folds:
            logger.info(f'Fold #{fold_num}')

            val_predictions = []
            test_predictions = []

            for label in common.LABELS:
                logger.info('Loading the training and validation data for the %s model', label)
                X_train = self.load_inputs(label, train_ids, 'train')
                X_val = self.load_inputs(label, val_ids, 'train')
                y_train = train_df.loc[train_df['id'].isin(train_ids)].sort_values('id')
                y_train = y_train[label].values
                y_val = train_df[train_df['id'].isin(val_ids)].sort_values('id')
                y_val = y_val[label].values

                logger.info('Training the %s model', label)
                model = self.train(fold_num, label, X_train, y_train, X_val, y_val)

                logger.info('Generating the out-of-fold predictions')
                y_model = self.predict(model, X_val)
                val_predictions.append(pd.DataFrame({'id': sorted(list(val_ids)), label: y_model}))

                logger.info('Generating the test predictions')
                X_test = self.load_inputs(label, test_df['id'].values, 'test')
                y_model = self.predict(model, X_test)
                test_predictions.append(pd.DataFrame({'id': test_df['id'], label: y_model}))

            val_predictions = functools.reduce(lambda l, r: pd.merge(l, r, on='id'), val_predictions)
            val_predictions = val_predictions[['id'] + common.LABELS]
            path = os.path.join(self.output_dir, f'fold{fold_num}_validation.csv')
            val_predictions.to_csv(path, index=False)

            test_predictions = functools.reduce(lambda l, r: pd.merge(l, r, on='id'), test_predictions)
            test_predictions = test_predictions[['id'] + common.LABELS]
            path = os.path.join(self.output_dir, f'fold{fold_num}_test.csv')
            test_predictions.to_csv(path, index=False)

        logger.info('Combining the out-of-fold predictions')
        df_parts = []
        for fold_num in range(1, 11):
            path = os.path.join(self.output_dir, f'fold{fold_num}_validation.csv')
            df_part = pd.read_csv(path, usecols=['id'] + common.LABELS)
            df_parts.append(df_part)
        train_pred = pd.concat(df_parts)
        path = os.path.join(self.output_dir, 'train.csv')
        train_pred.to_csv(path, index=False)

        logger.info('Averaging the test predictions')
        df_parts = []
        for fold_num in range(1, 11):
            path = os.path.join(self.output_dir, f'fold{fold_num}_test.csv')
            df_part = pd.read_csv(path, usecols=['id'] + common.LABELS)
            df_parts.append(df_part)
        test_pred = pd.concat(df_parts).groupby('id', as_index=False).mean()
        path = os.path.join(self.output_dir, 'test.csv')
        test_pred.to_csv(path, index=False)

        logger.info('Elapsed time - {}'.format(datetime.now() - t_start))

    def load_inputs(self, label, ids, dataset):
        X = []
        for name in self.params['models']:
            model = self.model_cls[name](name, self.model_params[name], random_seed=base.RANDOM_SEED)
            df = pd.read_csv(os.path.join(model.output_dir, f'{dataset}.csv'))
            df = df[df['id'].isin(ids)]
            df = df[['id'] + common.LABELS].sort_values('id')
            if self.params['input'] == 'single':
                values = np.expand_dims(df[label].values, -1)
            elif self.params['input'] == 'all':
                values = df[common.LABELS].values
            X.append(values)
        X = np.hstack(X)
        return X

    def build_model(self, input_size):
        model = StackingModule(
            input_size=input_size,
            hidden_units_factor=self.params['hidden_units_factor'],
            hidden_layers=self.params['hidden_layers'],
            hidden_nonlinearily='relu',
            input_dropout=self.params['input_dropout'],
            hidden_dropout=self.params['hidden_dropout'])
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def train(self, fold_num, label, X_train, y_train, X_val, y_val):
        model = self.build_model(X_train.shape[1])
        parameters = list(model.parameters())
        model_size = sum([np.prod(p.size()) for p in parameters])
        logger.info('Optimizing {:,} parameters:\n{}'.format(model_size, model))
        optimizer = optim.Adam(parameters, lr=self.params['lr'])

        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

        best_val_auc = 0
        patience_count = 0
        for epoch in itertools.count(start=1):
            for X_batch, y_batch in loader:
                X_batch = autograd.Variable(X_batch).float()
                y_batch = autograd.Variable(y_batch).float()
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                model.train()
                optimizer.zero_grad()
                output = model(X_batch).squeeze(-1)
                loss = F.binary_cross_entropy(output, y_batch)
                loss.backward()
                optimizer.step()

            y_model = self.predict(model, X_val)
            val_auc = roc_auc_score(y_val, y_model)

            logger.info('Epoch {} - loss {:.6f} - val_auc {:.6f}'
                        .format(epoch, loss.data[0], val_auc))

            if val_auc > best_val_auc:
                logger.info('Saving best model - val_auc {:.6f}'.format(val_auc))
                self.save_model(fold_num, label, model)
                best_val_auc = val_auc
                patience_count = 0
            else:
                patience_count += 1
                if patience_count == self.params['patience']:
                    logger.info('Finished training the %s model', label)
                    break

        model = self.load_model(fold_num, label, X_train.shape[1])
        return model

    def save_model(self, fold_num, label, model):
        model_file = os.path.join(self.output_dir, f'fold{fold_num}_{label}.pickle')
        torch.save(model.state_dict(), model_file)

    def load_model(self, fold_num, label, input_size):
        model = self.build_model(input_size)
        model_file = os.path.join(self.output_dir, f'fold{fold_num}_{label}.pickle')
        model.load_state_dict(torch.load(model_file))
        return model

    def predict(self, model, X):
        model.eval()
        dataset = TensorDataset(torch.FloatTensor(X), torch.zeros(X.shape[0]))
        loader = DataLoader(dataset, batch_size=self.params['batch_size'])
        scores = []
        for X_batch, _ in loader:
            x = autograd.Variable(X_batch)
            if torch.cuda.is_available():
                x = x.cuda()
            batch_scores = model(x)
            scores.extend(batch_scores.data.tolist())
        return np.array(scores).flatten()


if __name__ == '__main__':
    params = {
        'input': 'all',
        'models': ['rnn', 'cnn', 'mlp', 'xgb'],
        'hidden_layers': 3,
        'hidden_units_factor': 3,
        'input_dropout': 0.0,
        'hidden_dropout': 0.5,
        'lr': 0.005,
        'batch_size': 512,
        'patience': 10,
    }
    model = Stacking(params, random_seed=RANDOM_SEED)
    model.main()
