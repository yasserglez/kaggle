import os
import sys
import pprint
import tempfile
import itertools
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
import pandas as pd
import torch
from torch import nn, FloatTensor, LongTensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..clean_data import Products
from ..models import FitModel, PredictModel


products = Products.read()
PRODUCT_ID_TO_AISLE_ID = dict(zip(products.product_id, products.aisle_id))
PRODUCT_ID_TO_DEPT_ID = dict(zip(products.product_id, products.department_id))
del products


@contextmanager
def open_shuffled(file_path):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        subprocess.call(['shuf', file_path, '-o', f.name])
        yield open(f.name)


class RNNv4(object):

    product_history = luigi.IntParameter(default=91)
    scoring_dim = luigi.IntParameter(default=10)
    hidden_layers = luigi.IntParameter(default=2)
    hidden_nonlinearily = luigi.Parameter(default='leaky_relu')
    dropout = luigi.FloatParameter(default=0.0)

    random_seed = luigi.IntParameter(default=3996193, significant=False)
    global_orders_ratio = luigi.FloatParameter(default=0.00001, significant=False)
    validation_orders_ratio = luigi.FloatParameter(default=0.1, significant=False)
    target_orders_ratio = luigi.FloatParameter(default=0.1, significant=False)
    epochs = luigi.IntParameter(default=1000, significant=False)

    @property
    def model_name(self):
        params = [
            self.product_history,
            self.scoring_dim,
            self.hidden_layers,
            self.hidden_nonlinearily,
            self.dropout,
        ]
        model_name = 'rnn_v4_{}'.format('_'.join(str(p).lower() for p in params))
        return model_name

    def _init_random_state(self):
        self.random = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        torch.manual_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

    def _iter_user_data(self, orders_path, shuffle=False):
        with (open_shuffled(orders_path) if shuffle else open(orders_path)) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                yield user_data

    def _generate_example(self, prior_orders, last_order):
        recently_ordered = set()

        product_history = []
        for order_num, order in enumerate(prior_orders):
            weekday = order['day_of_week']
            hour_sin = np.sin(2 * np.pi * order['hour_of_day'] / 23)
            hour_cos = np.cos(2 * np.pi * order['hour_of_day'] / 23)
            hour = np.array([hour_sin, hour_cos])

            for product in order['products']:
                product_history.append({
                    'weekday': weekday,
                    'hour': hour,
                    'department': int(PRODUCT_ID_TO_DEPT_ID[product['product_id']] - 1),
                    'aisle': int(PRODUCT_ID_TO_AISLE_ID[product['product_id']] - 1),
                    'product': int(product['product_id'] - 1),
                })
                recently_ordered.add(product['product_id'])

        weekday = last_order['day_of_week']
        hour_sin = np.sin(2 * np.pi * last_order['hour_of_day'] / 23)
        hour_cos = np.cos(2 * np.pi * last_order['hour_of_day'] / 23)
        hour = np.array([hour_sin, hour_cos])

        next_products = []
        next_products_targets = []

        reordered = set()
        for product in last_order['products']:
            if product['reordered'] and product['product_id'] in recently_ordered:
                reordered.add(product['product_id'])

        for product_id in recently_ordered:
            next_products.append({
                'weekday': weekday,
                'hour': hour,
                'department': int(PRODUCT_ID_TO_DEPT_ID[product_id] - 1),
                'aisle': int(PRODUCT_ID_TO_AISLE_ID[product_id] - 1),
                'product': int(product_id - 1),
            })
            next_products_targets.append(int(product_id in reordered))

        return product_history, next_products, next_products_targets

    def _generate_examples(self, orders_path, target_orders=None, shuffle=False):
        for user_data in self._iter_user_data(orders_path, shuffle=shuffle):
            user_orders = user_data['prior_orders'].copy()
            user_orders.append(user_data['last_order'])

            # Determine the number of target orders to include for this user.
            user_target_orders = target_orders
            if not user_target_orders:
                user_target_orders = int(np.ceil(self.target_orders_ratio * len(user_orders)))

            for last_order_index in reversed(range(1, len(user_orders))):
                last_order = user_orders[last_order_index]
                prior_orders = []
                days_count = last_order['days_since_prior_order']
                for order in reversed(user_orders[:last_order_index]):
                    prior_orders.insert(0, order)
                    if order['days_since_prior_order'] is not None:
                        # There is at least another order, stop if it will go over the limit
                        days_count += order['days_since_prior_order']
                        if days_count >= self.product_history:
                            break
                yield self._generate_example(prior_orders, last_order)
                user_target_orders -= 1
                if user_target_orders == 0:
                    break

    def _format_as_tensors(self, product_history, next_products, next_products_targets):

        def create_tensor(tensor_type, orders, field):
            return Variable(tensor_type([p[field] for p in orders]), requires_grad=False)

        product_history_tensor = {
            'weekday': create_tensor(LongTensor, product_history, 'weekday').view(1, -1),
            'hour': create_tensor(FloatTensor, product_history, 'hour').view(1, -1, 2),
            'department': create_tensor(LongTensor, product_history, 'department').view(1, -1),
            'aisle': create_tensor(LongTensor, product_history, 'aisle').view(1, -1),
            'product': create_tensor(LongTensor, product_history, 'product').view(1, -1),
        }

        next_products_tensor = {
            'weekday': create_tensor(LongTensor, next_products, 'weekday').view(-1, 1),
            'hour': create_tensor(FloatTensor, next_products, 'hour').view(-1, 2),
            'department': create_tensor(LongTensor, next_products, 'department').view(-1, 1),
            'aisle': create_tensor(LongTensor, next_products, 'aisle').view(-1, 1),
            'product': create_tensor(LongTensor, next_products, 'product').view(-1, 1),
        }

        next_products_targets_tensor = Variable(FloatTensor(next_products_targets), requires_grad=False)

        return product_history_tensor, next_products_tensor, next_products_targets_tensor

    def _load_model(self):

        model = Model(
            weekday_dim=2,
            department_dim=3,
            aisle_dim=5,
            product_dim=10,
            scoring_dim=10,
            hidden_layers=self.hidden_layers,
            hidden_nonlinearily=self.hidden_nonlinearily,
            dropout=self.dropout)

        return model


def layer_units(input_size, output_size, num_layers=0):
    units = np.linspace(input_size, output_size, num_layers + 2)
    units = list(map(int, np.round(units, 0)))
    return units


class Projection(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers, nonlinearity, dropout):
        super(Projection, self).__init__()

        nonlinearities = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
        }

        layers = []
        units = layer_units(input_size, output_size, hidden_layers)
        for in_size, out_size in zip(units, units[1:]):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nonlinearities[nonlinearity])
        self.projection = nn.Sequential(*layers)

        # Weight initialization.
        for layer in layers:
            if isinstance(layer, nn.Linear):
                gain = 1.0
                if nonlinearity in {'tanh', 'relu', 'leaky_relu'}:
                    gain = nn.init.calculate_gain(nonlinearity)
                nn.init.xavier_uniform(layer.weight, gain=gain)
                nn.init.constant(layer.bias, 0.0)

    def forward(self, x):
        return self.projection(x)


class Model(nn.Module):

    def __init__(self, weekday_dim, department_dim, aisle_dim, product_dim, scoring_dim,
                 hidden_layers, hidden_nonlinearily, dropout):
        super(Model, self).__init__()

        self._init_embeddings(weekday_dim, department_dim, aisle_dim, product_dim)

        # Transformation of the product orders into the scoring space.
        self.product_order_proj = Projection(
            weekday_dim + 2 + department_dim + aisle_dim + product_dim,
            scoring_dim, hidden_layers=hidden_layers, nonlinearity=hidden_nonlinearily, dropout=dropout)

        self._init_lstm(scoring_dim)

        # Transformation of the LSTM output into the scoring space.
        self.lstm_output_proj = Projection(2 * scoring_dim, scoring_dim,
            hidden_layers=hidden_layers, nonlinearity=hidden_nonlinearily, dropout=dropout)

        self.probabilities = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def _init_embeddings(self, weekday_dim, department_dim, aisle_dim, product_dim):
        """
        Embeddings for the product order features.
        """
        self.weekday_embedding = nn.Embedding(7, weekday_dim)
        self.department_embedding = nn.Embedding(21, department_dim)
        self.product_embedding = nn.Embedding(49688, product_dim)
        self.aisle_embedding = nn.Embedding(134, aisle_dim)

        for emb in (self.weekday_embedding, self.department_embedding, self.aisle_embedding, self.product_embedding):
            nn.init.uniform(emb.weight, -0.1, 0.1)

    def _init_lstm(self, scoring_dim):
        """
        Bidirectional LSTM layers.
        """
        self.lstm = nn.LSTM(scoring_dim, scoring_dim, batch_first=True, bidirectional=True)

        # Weight initialization.
        for direction in range(2):
            suffix = '_reverse' if direction == 1 else ''
            param_name = 'weight_ih_l{}{}'.format(0, suffix)
            nn.init.xavier_uniform(getattr(self.lstm, param_name), gain=nn.init.calculate_gain('tanh'))
            param_name = 'weight_hh_l{}{}'.format(0, suffix)
            nn.init.orthogonal(getattr(self.lstm, param_name), gain=nn.init.calculate_gain('tanh'))
            for param_name in ['bias_ih_l{}{}', 'bias_hh_l{}{}']:
                param_name = param_name.format(0, suffix)
                nn.init.constant(getattr(self.lstm, param_name), 1)

    def forward(self, product_history, next_products):
        # Transform the sequence of prior product orders.
        weekday_vector = self.weekday_embedding(product_history['weekday'])
        department_vector = self.department_embedding(product_history['department'])
        aisle_vector = self.aisle_embedding(product_history['aisle'])
        product_vector = self.product_embedding(product_history['product'])
        order_features = [
            weekday_vector,
            product_history['hour'],
            department_vector,
            aisle_vector,
            product_vector,
        ]
        orders = torch.cat(order_features, dim=2)
        orders = self.product_order_proj(orders.squeeze(0)).unsqueeze(0)
        # orders.size() == (1, seq_len, scoring_dim)

        # Process the sequence of prior product orders with the recurrent layer.
        lstm_output, _ = self.lstm(orders)
        orders_vector = self.lstm_output_proj(lstm_output[:, -1])
        # orders_vector.size() == (1, scoring_dim)

        # Transform the next product orders.
        weekday_vector = self.weekday_embedding(next_products['weekday']).squeeze(1)
        department_vector = self.department_embedding(next_products['department']).squeeze(1)
        aisle_vector = self.aisle_embedding(next_products['aisle']).squeeze(1)
        product_vector = self.product_embedding(next_products['product']).squeeze(1)
        order_features = [
            weekday_vector,
            next_products['hour'],
            department_vector,
            aisle_vector,
            product_vector,
            # product_name_vector,
        ]
        product_vectors = torch.cat(order_features, dim=1)
        product_vectors = self.product_order_proj(product_vectors)
        # product_vectors.size() == (num_products, scoring_dim)

        # Calculate the similarity between the orders vector and each product vector.
        scores = torch.matmul(orders_vector, product_vectors.unsqueeze(2)).squeeze(2)

        # Squash the similarities into probabilities.
        probabilities = self.probabilities(scores).squeeze()

        return probabilities


class FitRNNv4(RNNv4, FitModel):

    def _split_orders(self, orders_path):
        validation_fd = tempfile.NamedTemporaryFile(mode='w+', delete=True)
        training_fd = tempfile.NamedTemporaryFile(mode='w+', delete=True)
        with open(orders_path) as orders_fd:
            for line in orders_fd:
                if self.global_orders_ratio >= 1 or self.random.uniform() <= self.global_orders_ratio:
                    if self.random.uniform() <= self.validation_orders_ratio:
                        validation_fd.write(line)
                    else:
                        training_fd.write(line)
        validation_fd.flush()
        training_fd.flush()
        return validation_fd, training_fd

    def _train_model(self, model, optimizer, orders_path):
        loss_sum = num_examples = 0
        for example in self._generate_examples(orders_path, shuffle=True):
            product_history, next_products, targets = self._format_as_tensors(*example)
            optimizer.zero_grad()
            outputs = model(product_history, next_products)
            loss = nn.functional.binary_cross_entropy(outputs, targets, size_average=True)
            loss.backward()
            optimizer.step()
            # Aggregate losses to calculate a global average
            loss_sum += loss.data[0] * outputs.size(0)
            num_examples += outputs.size(0)
        loss_avg = loss_sum / num_examples
        return loss_avg

    def _evaluate_model(self, model, orders_path):
        loss_sum = num_examples = 0
        for example in self._generate_examples(orders_path, target_orders=1, shuffle=False):
            product_history, next_products, targets = self._format_as_tensors(*example)
            outputs = model(product_history, next_products)
            loss = nn.functional.binary_cross_entropy(outputs, targets, size_average=False)
            # Aggregate losses to calculate a global average
            loss_sum += loss.data[0]
            num_examples += outputs.size()[0]
        loss_avg = loss_sum / num_examples
        return loss_avg

    def run(self):
        self._init_random_state()

        orders_path = self.requires()['orders'].output().path
        validation_fd, training_fd = self._split_orders(orders_path)

        model = self._load_model()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(parameters, lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-5, factor=0.1, patience=10)

        early_stopping_count = 0
        early_stopping_patience = 100
        val_loss = np.inf

        for epoch in range(1, self.epochs + 1):
            t_start = datetime.now()
            model.train()
            loss = self._train_model(model, optimizer, training_fd.name)
            model.eval()
            curr_val_loss = self._evaluate_model(model, validation_fd.name)
            t_end = datetime.now()

            curr_lr = None
            for group in optimizer.param_groups:
                curr_lr = group['lr']
            scheduler.step(curr_val_loss, epoch)

            if curr_val_loss < val_loss:
                # torch.save(model.state_dict(), self.output().path)
                val_loss = curr_val_loss
                early_stopping_count = 0

            print('Epoch {:04d}/{:04d}: loss: {:.6g} - val_loss: {:.6g} - lr: {:.6g} - time: {}'.format(
                epoch, self.epochs, loss, val_loss, curr_lr, str(t_end - t_start).split('.')[0]))

            if curr_val_loss > val_loss:
                early_stopping_count += 1
                if early_stopping_count == early_stopping_patience:
                    break

        validation_fd.close()
        training_fd.close()


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
