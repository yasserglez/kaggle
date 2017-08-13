import sys
import pprint
import bisect
import tempfile
from datetime import datetime

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
import pandas as pd
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

import torch
if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..models import FitModel, PredictModel


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda(async=True)
    return Variable(x)


class RNNv5(object):

    product_history = luigi.IntParameter(default=91)
    embedding_dim = luigi.IntParameter(default=10)
    lstm_size = luigi.IntParameter(default=25)
    lstm_layers = luigi.IntParameter(default=2)
    hidden_layers = luigi.IntParameter(default=2)
    hidden_nonlinearily = luigi.Parameter(default='leaky_relu')
    dropout = luigi.FloatParameter(default=0.2)

    random_seed = luigi.IntParameter(default=3996193, significant=False)
    global_orders_ratio = luigi.FloatParameter(default=1.0, significant=False)
    validation_orders_ratio = luigi.FloatParameter(default=0.2, significant=False)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    negative_factor = luigi.IntParameter(default=2, significant=False)
    epochs = luigi.IntParameter(default=1000, significant=False)

    @property
    def model_name(self):
        params = [
            self.product_history,
            self.embedding_dim,
            self.lstm_size,
            self.lstm_layers,
            self.hidden_layers,
            self.hidden_nonlinearily,
            self.dropout,
        ]
        model_name = 'rnn_v5_{}'.format('_'.join(str(p).lower() for p in params))
        return model_name

    def _init_random_state(self):
        self.random = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        torch.manual_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

    def _build_model(self):
        model = Model(
            embedding_dim=self.embedding_dim,
            lstm_size=self.lstm_size,
            lstm_layers=self.lstm_layers,
            hidden_layers=self.hidden_layers,
            hidden_nonlinearily=self.hidden_nonlinearily,
            dropout=self.dropout)

        if torch.cuda.is_available():
            model = model.cuda()

        print(model)

        return model


class OrdersDataset(Dataset):

    def __init__(self, orders_path, product_history):
        super(OrdersDataset, self).__init__()

        self._num_examples = 0
        self._order_index = []
        self.order_ids = []
        self.product_histories = []
        self.next_products = []
        self.targets = []
        for example in self._generate_examples(orders_path, product_history):
            order_id, product_history, next_products, next_product_targets = example
            self._num_examples += len(next_products)
            self.order_ids.append(order_id)
            self.product_histories.append(product_history)
            self._order_index.append(self._num_examples)
            self.next_products.extend(next_products)
            self.targets.extend(next_product_targets)

    def __len__(self):
        return self._num_examples

    def __getitem__(self, example_index):
        k = bisect.bisect_right(self._order_index, example_index)
        order_id = self.order_ids[k]
        product_history = self.product_histories[k]
        next_product = self.next_products[example_index]
        target = self.targets[example_index]
        example = {
            'order_id': order_id,
            'product_history': product_history,
            'next_product': next_product,
            'target': target,
        }
        return example

    def _generate_examples(self, orders_path, product_history):
        for user_data in self._iter_user_data(orders_path):
            user_orders = user_data['prior_orders'].copy()
            user_orders.append(user_data['last_order'])

            # Number of target orders to include for each user.
            user_target_orders = 1
            for last_order_index in reversed(range(1, len(user_orders))):
                last_order = user_orders[last_order_index]
                prior_orders = []
                days_count = last_order['days_since_prior_order']
                for order in reversed(user_orders[:last_order_index]):
                    prior_orders.insert(0, order)
                    if order['days_since_prior_order'] is not None:
                        # There is at least another order, stop if it will go over the limit.
                        days_count += order['days_since_prior_order']
                        if days_count >= product_history:
                            break
                yield self._generate_example(prior_orders, last_order)
                user_target_orders -= 1
                if user_target_orders == 0:
                    break

    def _iter_user_data(self, orders_path):
        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                yield user_data

    def _generate_example(self, prior_orders, last_order):
        product_history = []
        next_products = []
        targets = []

        recently_ordered = set()
        for order_num, order in enumerate(prior_orders):
            for product in order['products']:
                product_history.append(product['product_id'] - 1)
                recently_ordered.add(product['product_id'])

        reordered = set()
        if last_order['products']:
            for product in last_order['products']:
                if product['reordered'] and product['product_id'] in recently_ordered:
                    reordered.add(product['product_id'])

        for product_id in recently_ordered:
            next_products.append(int(product_id - 1))
            targets.append(int(product_id in reordered))

        # Make the order ID zero-based for consistency with the product ID's
        return (last_order['order_id'] - 1), product_history, next_products, targets

    @staticmethod
    def collate_fn(input_batch):
        order_id = [d['order_id'] for d in input_batch]
        product_history = [d['product_history'] for d in input_batch]
        product_history_lengths = [len(h) for h in product_history]
        next_product = [d['next_product'] for d in input_batch]
        target = [d['target'] for d in input_batch]

        # Sort the examples following product_history_lengths in reverse order.
        sort_index = np.argsort(product_history_lengths)[::-1]
        reorder = lambda l: [l[i] for i in sort_index]

        order_id = reorder(order_id)
        product_history = reorder(product_history)
        product_history_lengths = reorder(product_history_lengths)
        next_product = reorder(next_product)
        target = reorder(target)

        # Pad the product history sequences.
        pad = lambda h, length: np.pad(h, pad_width=(0, length - len(h)), mode='constant', constant_values=(0, 0))
        product_history = np.array([pad(h, product_history_lengths[0]) for h in product_history])
        product_history = to_var(torch.from_numpy(product_history))

        output_batch = {}
        output_batch['order_id'] = order_id
        output_batch['product_history'] = product_history
        output_batch['product_history_lengths'] = product_history_lengths
        output_batch['next_product'] = to_var(LongTensor(next_product))
        output_batch['target'] = to_var(FloatTensor(target))

        return output_batch


class OrdersSampler(Sampler):

    def __init__(self, orders_dataset, negative_factor=1):
        self._orders_dataset = orders_dataset
        self._negative_factor = negative_factor
        positive = np.array(self._orders_dataset.targets) > 0
        self._positive_indices = np.argwhere(positive).flatten()
        self._negative_indices = np.argwhere(np.logical_not(positive)).flatten()
        assert self._positive_indices.shape[0] <= self._negative_indices.shape[0]

    def __iter__(self):
        positive_perm = torch.randperm(self._positive_indices.shape[0])
        negative_perm = torch.randperm(self._negative_indices.shape[0])
        j = 0
        for i in positive_perm:
            yield self._positive_indices[i]
            for _ in range(self._negative_factor):
                yield self._negative_indices[negative_perm[j]]
                j += 1

    def __len__(self):
        return self._positive_indices.shape[0] * (1 + self._negative_factor)


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

    def __init__(self, embedding_dim, lstm_size, lstm_layers, hidden_layers, hidden_nonlinearily, dropout):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(49688, embedding_dim)
        nn.init.uniform(self.embedding.weight, -0.05, 0.05)

        self.lstm_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embedding_dim, lstm_size, num_layers=lstm_layers, dropout=dropout,
            batch_first=True, bidirectional=True)
        for layer in range(lstm_layers):
            for direction in range(2):
                suffix = '_reverse' if direction == 1 else ''
                param_name = 'weight_ih_l{}{}'.format(layer, suffix)
                nn.init.xavier_uniform(getattr(self.lstm, param_name), gain=nn.init.calculate_gain('tanh'))
                param_name = 'weight_hh_l{}{}'.format(layer, suffix)
                nn.init.orthogonal(getattr(self.lstm, param_name), gain=nn.init.calculate_gain('tanh'))
                for param_name in ['bias_ih_l{}{}', 'bias_hh_l{}{}']:
                    param_name = param_name.format(layer, suffix)
                    nn.init.constant(getattr(self.lstm, param_name), 1.0)

        self.attention = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        nn.init.xavier_uniform(self.attention[0].weight)
        nn.init.constant(self.attention[0].bias, 0.0)

        self.probability = nn.Sequential(
            Projection(2 * lstm_size, 1, hidden_layers=hidden_layers, nonlinearity=hidden_nonlinearily, dropout=dropout),
            nn.Sigmoid())

    def forward(self, product_history, product_history_lengths, next_product):
        # Determine the embeddings for the product history.
        product_history_vectors = self.embedding(product_history)

        # Determine the embedding for the next products.
        next_product_vector = self.embedding(next_product)

        # Calculate the attention vector.
        attention = next_product_vector.unsqueeze(1).expand_as(product_history_vectors)
        attention = nn.functional.cosine_similarity(product_history_vectors, attention, dim=2)
        attention_size = attention.size()
        attention = self.attention(attention.view(-1, 1)).view(attention_size)

        # Applying attention to the product history (after attention).
        attention = attention.unsqueeze(2).expand(product_history_vectors.size())
        product_history_vectors = product_history_vectors * attention

        # Process the product history (after attention) with the recurrent layer.
        product_history_vectors = self.lstm_dropout(product_history_vectors)
        packed_product_history = pack_padded_sequence(product_history_vectors, product_history_lengths, batch_first=True)
        lstm_output, _ = self.lstm(packed_product_history)  # returns a PackedSequence
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)  # unpack it
        # and get the last output for each sequence
        i = LongTensor(range(lstm_output.size(0)))
        j = LongTensor(product_history_lengths) - 1
        lstm_output = lstm_output[i, j, :]

        # Calculate the probabilities.
        probabilities = self.probability(lstm_output).squeeze()

        return probabilities


class FitRNNv5(RNNv5, FitModel):

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

    def _train_model(self, model, optimizer, data_loader):
        loss_sum = num_examples = 0

        for batch in data_loader:
            product_history = batch['product_history']
            product_history_lengths = batch['product_history_lengths']
            next_product = batch['next_product']
            target = batch['target']

            optimizer.zero_grad()
            output = model(product_history, product_history_lengths, next_product)
            loss = nn.functional.binary_cross_entropy(output, target, size_average=True)
            loss.backward()
            # Clip the gradients of the LSTM weights such that their norm is bounded by 1.0
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

            # Aggregate losses to calculate a global average
            loss_sum += loss.data[0] * output.size(0)
            num_examples += output.size(0)

        loss_avg = loss_sum / num_examples
        return loss_avg

    def _evaluate_model(self, model, data_loader):
        loss_sum = num_examples = 0

        for batch in data_loader:
            product_history = batch['product_history']
            product_history_lengths = batch['product_history_lengths']
            next_product = batch['next_product']
            target = batch['target']

            output = model(product_history, product_history_lengths, next_product)
            loss = nn.functional.binary_cross_entropy(output, target, size_average=False)

            # Aggregate losses to calculate a global average
            loss_sum += loss.data[0]
            num_examples += output.size()[0]

        loss_avg = loss_sum / num_examples
        return loss_avg

    def run(self):
        self._init_random_state()

        orders_path = self.requires()['orders'].output().path
        validation_fd, training_fd = self._split_orders(orders_path)

        print('Loading the training examples... ', end='', flush=True)
        training_data = OrdersDataset(training_fd.name, product_history=self.product_history)
        print('{:,} loaded.'.format(len(training_data)))
        training_loader = DataLoader(
            training_data, collate_fn=OrdersDataset.collate_fn,
            sampler=OrdersSampler(training_data, self.negative_factor),
            pin_memory=torch.cuda.is_available(), batch_size=self.batch_size)

        print('Loading the validation examples... ', end='', flush=True)
        validation_data = OrdersDataset(validation_fd.name, product_history=self.product_history)
        print('{:,} loaded.'.format(len(validation_data)))
        validation_loader = DataLoader(
            validation_data, collate_fn=OrdersDataset.collate_fn,
            pin_memory=torch.cuda.is_available(), batch_size=self.batch_size)

        model = self._build_model()

        lr = 0.1
        min_lr = 1e-4
        val_loss = np.inf
        patience = 10
        patience_count = 0

        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, min_lr=min_lr, factor=0.1, patience=patience)

        for epoch in range(1, self.epochs + 1):
            t_start = datetime.now()
            model.train()
            loss = self._train_model(model, optimizer, training_loader)
            model.eval()
            curr_val_loss = self._evaluate_model(model, validation_loader)
            t_end = datetime.now()

            for group in optimizer.param_groups:
                lr = group['lr']
            scheduler.step(curr_val_loss, epoch)

            if curr_val_loss < val_loss:
                torch.save(model.state_dict(), self.output().path)
                val_loss = curr_val_loss
                patience_count = 0
            else:
                if np.isclose(lr, min_lr):
                    # Wait until the learning rate is decreased before considering early stopping
                    patience_count += 1
                if patience_count >= patience:
                    break

            print('Epoch {:04d}/{:04d}: loss: {:.6g} - val_loss: {:.6g} - lr: {:.6g} - time: {}'.format(
                epoch, self.epochs, loss, val_loss, lr, str(t_end - t_start).split('.')[0]))

        validation_fd.close()
        training_fd.close()


class _PredictRNNv5(RNNv5, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitRNNv5(
            mode=self.mode,
            product_history=self.product_history,
            embedding_dim=self.embedding_dim,
            lstm_size=self.lstm_size,
            hidden_layers=self.hidden_layers,
            hidden_nonlinearily=self.hidden_nonlinearily,
            dropout=self.dropout,
            global_orders_ratio=self.global_orders_ratio,
            validation_orders_ratio=self.validation_orders_ratio,
            batch_size=self.batch_size,
            negative_factor=self.negative_factor,
            epochs=self.epochs)
        return req


class PredictRNNv5ReorderSizeKnown(_PredictRNNv5):

    @staticmethod
    def _count_reordered_products(order):
        k = 0
        for product in order['products']:
            if product['reordered']:
                k += 1
        return k

    def _determine_reorder_size(self):
        reorder_size = {}
        orders_path = self.requires()['orders'].output().path
        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                order_id = int(user_data['last_order']['order_id'])
                reorder_size[order_id] = self._count_reordered_products(user_data['last_order'])
        return reorder_size

    def run(self):
        self._init_random_state()

        orders_path = self.requires()['orders'].output().path
        print('Loading the evaluation examples... ', end='', flush=True)
        data = OrdersDataset(orders_path, product_history=self.product_history)
        print('{:,} loaded.'.format(len(data)))
        data_loader = DataLoader(
            data, collate_fn=OrdersDataset.collate_fn,
            batch_size=self.batch_size, pin_memory=torch.cuda.is_available())

        model = self._build_model()
        model.load_state_dict(torch.load(self.input()['model'].path))
        model.eval()

        order_ids = []
        product_ids = []
        scores = []
        for batch_num, batch in enumerate(data_loader):
            next_products = batch['next_product']
            output = model(batch['product_history'], batch['product_history_lengths'], next_products)
            for order_id, product_index, score in zip(batch['order_id'], next_products.data.tolist(), output.data.tolist()):
                order_ids.append(order_id + 1)
                product_ids.append(product_index + 1)
                scores.append(score)
            if batch_num % (len(data_loader) // 100) == 0:
                print('Processed {:,}/{:,} batches...'.format(batch_num + 1, len(data_loader)))
        print('done.')

        scores = pd.DataFrame({'order_id': order_ids, 'product_id': product_ids, 'score': scores})
        reorder_size = self._determine_reorder_size()

        predictions = {}
        for order_id in set(order_ids):
            predictions[order_id] = []
            df = scores[scores.order_id == order_id].nlargest(reorder_size[order_id], 'score')
            for row in df.itertuples(index=False):
                predictions[int(order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizeRNNv5ReorderSizeKnown(luigi.Task):

    choices = {
        'product_history': [91],
        'embedding_dim': [5, 10],
        'lstm_size': [15, 25],
        'lstm_layers': [2],
        'hidden_layers': [2],
        'hidden_nonlinearily': ['leaky_relu'],
        'dropout': [0.2],
    }

    def run(self):
        space = {k: hp.choice(k, v) for k, v in self.choices.items()}
        while True:
            values = sample(space)
            yield PredictRNNv5ReorderSizeKnown(
                mode='evaluation',
                product_history=values['product_history'],
                embedding_dim=values['embedding_dim'],
                lstm_size=values['lstm_size'],
                lstm_layers=values['lstm_layers'],
                hidden_layers=values['hidden_layers'],
                hidden_nonlinearily=values['hidden_nonlinearily'],
                dropout=values['dropout'],
                global_orders_ratio=0.1)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
