import os
import sys
import logging
import pprint
import functools
from datetime import datetime
import multiprocessing as mp

import numpy as np
from numpy.random import RandomState
import pandas as pd
import xgboost as xgb
import joblib

import common
import base
from lstm import LSTM
from gru import GRU
from gcnn import GCNN
from dpcnn import DPCNN
from ngram import NGram
from mlp import MLP
from xgb import XGB


logger = logging.getLogger(__name__)


RANDOM_SEED = 3468526


class LabelStacking(object):

    model_cls = {
        'lstm': LSTM,
        'gru': GRU,
        'dpcnn': DPCNN,
        'gcnn': GCNN,
        'char-ngram': NGram,
        'word-ngram': NGram,
        'mlp': MLP,
        'xgb': XGB,
    }

    model_params = {
        'gcnn': {
            'vocab_size': 100000,
            'max_len': 300,
            'vectors': 'glove.42B.300d',
            'num_blocks': 1,
            'num_layers': 2,
            'num_channels': 128,
            'kernel_size': 3,
            'dense_layers': 0,
            'dense_dropout': 0.5,
            'batch_size': 64,
            'lr_high': 1.0,
            'lr_low': 0.2,
        },
        'lstm': {
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
        'dpcnn': {
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
        'char-ngram': {
            'analyzer': 'char',
            'min_df': 5,
            'max_ngram': 5,
            'max_features': 100000,
            'C': 1.0,
        },
        'word-ngram': {
            'analyzer': 'word',
            'min_df': 5,
            'max_ngram': 2,
            'max_features': 50000,
            'C': 1.0,
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
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.4,
            'colsample_bytree': 0.5,
            'learning_rate': 0.1,
            'patience': 50,
        },
        'gru': {
            'vocab_size': 100000,
            'max_len': 300,
            'vectors': 'glove.twitter.27B.200d',
            'annotation_dropout': 0.1,
            'prediction_dropout': 0.3,
            'batch_size': 256,
            'lr_high': 0.5,
            'lr_low': 0.1,
        },
    }

    def __init__(self, label, params, random_seed):
        self.label = label
        self.params = params
        self.random_seed = random_seed

        self.output_dir = os.path.join(
            common.OUTPUT_DIR,
            'label_stacking', str(self.random_seed), self.label,
            common.params_str(self.params))
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def main(self):
        t_start = datetime.now()
        logger.info(' label_stacking / {} '.format(self.random_seed).center(62, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(os.path.join(self.output_dir, 'test.csv')):
            logger.info('Output already exists - skipping')
            return

        self.random_state = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))

        test_df = common.load_data('test')
        train_df = common.load_data('train')

        folds = common.stratified_kfold(train_df, random_seed=self.random_seed)
        for fold_num, train_ids, val_ids in folds:
            logger.info(f'Fold #{fold_num}')

            logger.info('Loading the training and validation data for the %s model', self.label)
            X_train = self.load_inputs(train_ids, 'train')
            X_val = self.load_inputs(val_ids, 'train')
            y_train = train_df.loc[train_df['id'].isin(train_ids)].sort_values('id')
            y_train = y_train[self.label].values
            y_val = train_df[train_df['id'].isin(val_ids)].sort_values('id')
            y_val = y_val[self.label].values

            logger.info('Training the %s model', self.label)
            model = self.train(fold_num, self.label, X_train, y_train, X_val, y_val)

            logger.info('Generating the out-of-fold predictions')
            y_model = self.predict(model, X_val)
            val_pred = pd.DataFrame({'id': sorted(list(val_ids)), self.label: y_model})
            path = os.path.join(self.output_dir, f'fold{fold_num}_validation.csv')
            val_pred.to_csv(path, index=False)

            logger.info('Generating the test predictions')
            X_test = self.load_inputs(test_df['id'].values, 'test')
            y_model = self.predict(model, X_test)
            test_pred = pd.DataFrame({'id': test_df['id'], self.label: y_model})
            path = os.path.join(self.output_dir, f'fold{fold_num}_test.csv')
            test_pred.to_csv(path, index=False)

        logger.info('Averaging the test predictions')
        df_parts = []
        for fold_num in range(1, 11):
            path = os.path.join(self.output_dir, f'fold{fold_num}_test.csv')
            df_part = pd.read_csv(path, usecols=['id', self.label])
            df_parts.append(df_part)
        test_pred = pd.concat(df_parts).groupby('id', as_index=False).mean()
        path = os.path.join(self.output_dir, 'test.csv')
        test_pred.to_csv(path, index=False)

        logger.info('Elapsed time - {}'.format(datetime.now() - t_start))

    def load_inputs(self, ids, dataset):
        X = []
        for name in self.params['models']:
            model = self.model_cls[name](self.model_params[name], random_seed=base.RANDOM_SEED)
            df = pd.read_csv(os.path.join(model.output_dir, f'{dataset}.csv'))
            df = df[df['id'].isin(ids)]
            df = df[['id'] + common.LABELS].sort_values('id')
            X.append(df[common.LABELS].values)
        X = np.hstack(X)
        return X

    def train(self, fold_num, label, X_train, y_train, X_val, y_val):

        model = xgb.XGBClassifier(
            n_estimators=10000,  # determined by early stopping
            objective='binary:logistic',
            max_depth=self.params['max_depth'],
            min_child_weight=self.params['min_child_weight'],
            subsample=self.params['subsample'],
            colsample_bytree=self.params['colsample_bytree'],
            learning_rate=self.params['learning_rate'],
            random_state=self.random_seed,
            n_jobs=mp.cpu_count())

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], eval_metric='auc',
                  early_stopping_rounds=self.params['patience'])

        self.save_model(fold_num, label, model)
        return model

    def save_model(self, fold_num, label, model):
        model_file = os.path.join(self.output_dir, f'fold{fold_num}_{label}.pickle')
        joblib.dump(model, model_file)

    def load_model(self, fold_num, label):
        model_file = os.path.join(self.output_dir, f'fold{fold_num}_{label}.pickle')
        model = joblib.load(model_file)
        return model

    def predict(self, model, X):
        output = model.predict_proba(X, ntree_limit=model.best_ntree_limit)[:, 1]
        return output


class Stacking(object):

    def __init__(self, tag, params, random_seed):
        self.tag = tag
        self.params = params
        self.random_seed = random_seed

        self.output_dir = os.path.join(
            common.OUTPUT_DIR, 'label_stacking',
            str(self.random_seed), self.tag)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def main(self):
        if os.path.isfile(os.path.join(self.output_dir, 'test.csv')):
            logger.info('Output already exists - skipping')
            return

        # Generate each label's predictions
        models = {}
        for label in common.LABELS:
            model = LabelStacking(label, self.params[label], self.random_seed)
            model.main()
            models[label] = model

        logger.info('Combining the out-of-fold predictions')
        train_pred = []
        for fold_num in range(1, 11):
            fold_pred = []
            for label, model in models.items():
                path = os.path.join(model.output_dir, f'fold{fold_num}_validation.csv')
                df_part = pd.read_csv(path, usecols=['id', label])
                fold_pred.append(df_part)
            fold_pred = functools.reduce(lambda l, r: pd.merge(l, r, on='id'), fold_pred)
            path = os.path.join(self.output_dir, f'fold{fold_num}_validation.csv')
            fold_pred.to_csv(path, index=False)
            train_pred.append(fold_pred)
        train_pred = pd.concat(train_pred)
        path = os.path.join(self.output_dir, 'train.csv')
        train_pred.to_csv(path, index=False)

        logger.info('Combining the test predictions')
        df_parts = []
        for label, model in models.items():
            path = os.path.join(model.output_dir, 'test.csv')
            df_part = pd.read_csv(path, usecols=['id', label])
            df_parts.append(df_part)
        test_pred = functools.reduce(lambda l, r: pd.merge(l, r, on='id'), df_parts)
        test_pred = test_pred[['id'] + common.LABELS]
        path = os.path.join(self.output_dir, 'test.csv')
        test_pred.to_csv(path, index=False)


if __name__ == '__main__':
    params = {
        'identity_hate': {
            'models': ['char-ngram', 'dpcnn', 'gcnn', 'lstm', 'mlp', 'word-ngram', 'xgb'],
            'max_depth': 5,
            'min_child_weight': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'learning_rate': 0.2,
            'patience': 25,
        },
        'insult': {
            'models': ['char-ngram', 'dpcnn', 'gcnn', 'lstm', 'mlp', 'word-ngram', 'xgb'],
            'max_depth': 5,
            'min_child_weight': 5,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'learning_rate': 0.1,
            'patience': 25,
        },
        'obscene': {
            'models': ['char-ngram', 'dpcnn', 'gcnn', 'lstm', 'mlp', 'word-ngram', 'xgb'],
            'max_depth': 5,
            'min_child_weight': 5,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'learning_rate': 0.1,
            'patience': 25,
        },
        'severe_toxic': {
            'models': ['char-ngram', 'dpcnn', 'gcnn', 'lstm', 'mlp', 'word-ngram', 'xgb'],
            'max_depth': 2,
            'min_child_weight': 5,
            'subsample': 0.5,
            'colsample_bytree': 0.7,
            'learning_rate': 0.2,
            'patience': 25,
        },
        'threat': {
            'models': ['dpcnn', 'gcnn', 'lstm', 'mlp', 'word-ngram', 'xgb'],
            'max_depth': 10,
            'min_child_weight': 5,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'learning_rate': 0.3,
            'patience': 25,
        },
        'toxic': {
            'models': ['char-ngram', 'dpcnn', 'gcnn', 'lstm', 'mlp', 'word-ngram', 'xgb'],
            'max_depth': 7,
            'min_child_weight': 10,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'learning_rate': 0.1,
            'patience': 25,
        },
    }
    model = Stacking('submission119', params, random_seed=RANDOM_SEED)
    model.main()

    for label in common.LABELS:
        params[label]['models'].append('gru')
    model = Stacking('submission122', params, random_seed=RANDOM_SEED)
    model.main()
