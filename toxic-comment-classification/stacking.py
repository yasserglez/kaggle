import os
import sys
import logging
import pprint
import functools
import itertools
from datetime import datetime

import numpy as np
from numpy.random import RandomState
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import pandas as pd
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

import common
import base
from bagging import Bagging


logger = logging.getLogger(__name__)


class ScoreModule(nn.Module):

    def __init__(self, input_size, hidden_units_factor, hidden_layers, hidden_nonlinearily,
                 input_dropout=0, hidden_dropout=0):
        super().__init__()

        self.dense = base.Dense(
            input_size, 1,
            output_nonlinearity=None,
            hidden_layers=[hidden_units_factor * input_size] * hidden_layers,
            hidden_nonlinearity=hidden_nonlinearily,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout)

    def forward(self, x):
        return self.dense(x)


class Stacking(object):

    def __init__(self, params, random_seed, debug=False):
        self.params = params
        self.random_seed = random_seed
        self.debug = debug

        self.output_dir = os.path.join(
            common.OUTPUT_DIR,
            'stacking', str(self.random_seed),
            common.params_str(self.params))
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if self.debug:
            self.writer = SummaryWriter(self.output_dir)

        self.test_output = os.path.join(self.output_dir, 'test.csv')

    def main(self):
        t_start = datetime.now()
        logger.info(' stacking / {} '.format(self.random_seed).center(62, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(self.test_output):
            logger.info('Output already exists - skipping')
            return

        self.random_state = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))
        torch.manual_seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))

        input_models = self.load_input_models()
        val_set = set(common.load_data(self.random_seed, 'validation')['id'].values)

        predictions = []
        for label, ids, X, y in self.load_data(input_models, 'train'):
            # Split into training and validation
            ids_train, ids_val = [], []
            X_train, X_val = [], []
            y_train, y_val = [], []
            for i in range(len(ids)):
                if ids[i] in val_set:
                    ids_val.append(ids[i])
                    X_val.append(X[i])
                    y_val.append(y[i])
                else:
                    ids_train.append(ids[i])
                    X_train.append(X[i])
                    y_train.append(y[i])
            X_train = np.array(X_train)
            X_val = np.array(X_val)
            y_train = np.array(y_train)
            y_val = np.array(y_val)

            logger.info('Training the %s model', label)
            self.train(label, X_train, y_train, X_val, y_val)

            logger.info('Generating validation predictions for the %s model', label)
            model = self.load_model(label, X.shape[1])
            y_model = self.predict(model, X_val)
            predictions.append(pd.DataFrame({'id': ids_val, label: y_model}))

        predictions = functools.reduce(lambda l, r: pd.merge(l, r, on='id'), predictions)
        predictions = predictions[['id'] + common.LABELS]
        predictions.to_csv(os.path.join(self.output_dir, 'validation.csv'), index=False)

        predictions = []
        for label, ids, X in self.load_data(input_models, 'test'):
            logger.info('Generating test predictions for the %s model', label)
            model = self.load_model(label, X.shape[1])
            y_model = self.predict(model, X)
            predictions.append(pd.DataFrame({'id': ids, label: y_model}))

        predictions = functools.reduce(lambda l, r: pd.merge(l, r, on='id'), predictions)
        predictions = predictions[['id'] + common.LABELS]
        predictions.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)

        logger.info('Elapsed time - {}'.format(datetime.now() - t_start))

    def build_model(self, input_size):
        model = ScoreModule(
            input_size=input_size,
            hidden_units_factor=self.params['hidden_units_factor'],
            hidden_layers=self.params['hidden_layers'],
            hidden_nonlinearily=self.params['hidden_nonlinearily'],
            input_dropout=self.params['input_dropout'],
            hidden_dropout=self.params['hidden_dropout'])
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def train(self, label, X_train, y_train, X_val, y_val):
        positives = np.where(y_train == 1)[0]
        negatives = np.where(y_train == 0)[0]
        logger.info('There are {:,} positive and {:,} negative examples'
                    .format(positives.shape[0], negatives.shape[0]))

        model = self.build_model(X_train.shape[1])
        parameters = list(model.parameters())
        model_size = sum([np.prod(p.size()) for p in parameters])
        logger.info('Optimizing {:,} parameters:\n{}'.format(model_size, model))
        optimizer = optim.Adam(parameters, lr=self.params['lr'],
                               weight_decay=self.params['weight_decay'])

        best_val_auc = 0
        patience_count = 0
        for iteration in itertools.count(start=1):
            # BPR: Bayesian Personalized Ranking from Implicit Feedback
            # https://arxiv.org/pdf/1205.2618.pdf
            model.train()
            optimizer.zero_grad()
            pos_indices = self.random_state.choice(positives, self.params['batch_size'])
            neg_indices = self.random_state.choice(negatives, self.params['batch_size'])
            pos_input = autograd.Variable(torch.FloatTensor(X_train[pos_indices]))
            neg_input = autograd.Variable(torch.FloatTensor(X_train[neg_indices]))
            if torch.cuda.is_available():
                pos_input = pos_input.cuda()
                neg_input = neg_input.cuda()
            pos_scores = model(pos_input)
            neg_scores = model(neg_input)
            loss = (-F.logsigmoid(pos_scores - neg_scores)).mean()
            loss.backward()
            optimizer.step()

            model.eval()
            y_model = self.predict(model, X_val)
            val_auc = roc_auc_score(y_val, y_model)
            logger.info('Iteration {} - loss {:.6f} - val_auc {:.6f}'
                        .format(iteration, loss.data[0], val_auc))

            if self.debug:
                y_model = self.predict(model, X_train)
                auc = roc_auc_score(y_train, y_model)
                self.writer.add_scalars(f'{label}/auc', {'train': auc, 'validation': val_auc}, iteration)
                self.writer.add_scalar(f'{label}/loss', loss.data[0], iteration)

            if val_auc > best_val_auc:
                logger.info('Saving best model - val_auc {:.6f}'.format(val_auc))
                self.save_model(model, label)
                best_val_auc = val_auc
                patience_count = 0
            else:
                patience_count += 1
                if patience_count == self.params['patience']:
                    logger.info('Finished training the %s model', label)
                    break

    def save_model(self, model, label):
        model_file = os.path.join(self.output_dir, '{}.pickle'.format(label))
        torch.save(model.state_dict(), model_file)

    def load_model(self, label, input_size):
        model = self.build_model(input_size)
        model_file = os.path.join(self.output_dir, '{}.pickle'.format(label))
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
        scores = rankdata(scores, method='min') / len(scores)
        return scores

    def load_input_models(self):
        models = []
        for name in self.params['models']:
            model = Bagging(name)
            # model.main()
            models.append(model)
        return models

    def load_data(self, input_models, dataset):
        model = input_models[0]
        csv_file = model.train_output if dataset == 'train' else model.test_output
        df = pd.read_csv(csv_file, usecols=['id']).sort_values('id')
        ids = df['id']

        for label in common.LABELS:
            logger.info('Loading the %s data for the %s model', dataset, label)
            X = []
            for model in input_models:
                df = pd.read_csv(model.train_output if dataset == 'train' else model.test_output)
                df = df[['id'] + common.LABELS].sort_values('id')
                if self.params['input'] == 'single':
                    values = np.expand_dims(df[label].values, -1)
                elif self.params['input'] == 'all':
                    values = df[common.LABELS].values
                # Calculate the normalized ranks
                for i in range(values.shape[1]):
                    values[:, i] = rankdata(values[:, i], method='min') / values.shape[0]
                X.append(values)
            X = np.hstack(X)
            if dataset == 'train':
                df = pd.read_csv(os.path.join(common.DATA_DIR, 'train.csv'), usecols=['id', label])
                y = df.sort_values('id')[label].values
                yield label, ids, X, y
            else:
                yield label, ids, X


if __name__ == '__main__':
    params = {
        'input': 'all',
        'models': ['rnn', 'cnn', 'mlp', 'xgb'],
        'hidden_layers': 2,
        'hidden_units_factor': 3,
        'hidden_nonlinearily': 'relu',
        'input_dropout': 0.0,
        'hidden_dropout': 0.3,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'patience': 10,
    }
    model = Stacking(params, random_seed=42, debug=False)
    model.main()
