import os
import sys
import pprint
import tempfile
import itertools
import subprocess
import multiprocessing
from collections import defaultdict
from contextlib import contextmanager

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from ..clean_data import Products
from ..models import FitModel, PredictModel
from .f1_maximization import maximize_expected_f1


class _MLPv2(object):

    num_orders_per_user = luigi.IntParameter(default=5)
    product_history = luigi.IntParameter(default=91)
    product_embedding = luigi.IntParameter(default=5)
    hidden_layers = luigi.IntParameter(default=3)
    dropout = luigi.FloatParameter(default=0.5)

    random_seed = luigi.IntParameter(default=3996193, significant=False)
    global_orders_ratio = luigi.FloatParameter(default=1.0, significant=False)
    validation_orders_ratio = luigi.FloatParameter(default=0.1, significant=False)
    batch_size = luigi.IntParameter(default=1024, significant=False)
    epochs = luigi.IntParameter(default=1000, significant=False)

    num_products = Products.count()

    @property
    def model_name(self):
        params = [
            self.num_orders_per_user,
            self.product_history,
            self.product_embedding,
            self.hidden_layers,
            self.dropout,
        ]
        model_name = 'mlp_v2_{}'.format('_'.join(str(p).lower() for p in params))
        return model_name

    @staticmethod
    def _count_lines(file_path):
        p = subprocess.Popen(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return int(p.communicate()[0].partition(b' ')[0])

    @contextmanager
    def _open_shuffled(self, file_path):
        with tempfile.NamedTemporaryFile(delete=True) as f:
            subprocess.call(['shuf', file_path, '-o', f.name])
            yield open(f.name)

    def _generate_examples(self, last_order, prior_orders):
        # Collect the products that were ordered each day for the past self.product_history days
        products_by_day = defaultdict(set)
        num_days = last_order['days_since_prior_order']
        for order in reversed(prior_orders):
            for product in order['products']:
                products_by_day[num_days].add(product['product_id'])
            if order['days_since_prior_order'] is not None:
                # There is at least another order, stop if it will go over the limit
                num_days += order['days_since_prior_order']
                if num_days >= self.product_history:
                    break

        # Collect the products that were reordered
        positive_examples = set()
        if last_order['products']:
            for product in last_order['products']:
                if product['reordered']:
                    positive_examples.add(product['product_id'])
        # and previously purchased but not reordered
        negative_examples = set(p for P in products_by_day.values() for p in P)

        # Exclude any reordered product not purchased in the last self.product_history days
        positive_examples -= (positive_examples - negative_examples)
        negative_examples -= positive_examples

        for product_id in (positive_examples | negative_examples):
            # Generate the product_history vector
            product_history = []
            for day_num in range(self.product_history):
                value = 1.0 if product_id in products_by_day[day_num] else -1.0
                product_history.append(value)
            # Return the example
            inputs = {'product': product_id - 1, 'product_history': product_history}
            prediction = float(product_id in positive_examples)
            yield last_order['order_id'], product_id, inputs, prediction

    def _generate_user_examples(self, user_data, num_orders_per_user):
        for values in self._generate_examples(user_data['last_order'], user_data['prior_orders']):
            yield (user_data['user_id'], ) + values
        num_orders_per_user -= 1
        if num_orders_per_user > 0:
            for k in range(len(user_data['prior_orders']) - 1, 0, -1):
                last_order = user_data['prior_orders'][k]
                prior_orders = user_data['prior_orders'][:k]
                for values in self._generate_examples(last_order, prior_orders):
                    yield (user_data['user_id'], ) + values
                num_orders_per_user -= 1
                if num_orders_per_user == 0:
                    break

    def _load_data(self, orders_path, num_orders_per_user=1):
        all_user_ids = []
        all_order_ids = []
        all_product_ids = []
        all_inputs = defaultdict(list)
        all_predictions = []

        def add_example(user_id, order_id, product_id, inputs, prediction):
            all_user_ids.append(user_id)
            all_order_ids.append(order_id)
            all_product_ids.append(product_id)
            for k in inputs.keys():
                all_inputs[k].append(inputs[k])
            all_predictions.append(prediction)

        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                for user_id, order_id, product_id, inputs, prediction in \
                        self._generate_user_examples(user_data, num_orders_per_user=num_orders_per_user):
                    add_example(user_id, order_id, product_id, inputs, prediction)

        # Build the numpy arrays
        for k in all_inputs.keys():
            all_inputs[k] = np.array(all_inputs[k])
        all_predictions = np.array(all_predictions)

        return all_user_ids, all_order_ids, all_product_ids, all_inputs, all_predictions

    def _create_data_generator(self, orders_path, num_orders_per_user, batch_size):
        # Count the number of training examples
        num_examples = 0
        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                for _ in self._generate_user_examples(user_data, num_orders_per_user):
                    num_examples += 1

        batch_sizes = [len(a) for a in np.array_split(range(num_examples), num_examples / batch_size)]
        steps_per_epoch = len(batch_sizes)

        def generator():
            while True:
                current_step = 0
                all_inputs, all_predictions = defaultdict(list), []
                with self._open_shuffled(orders_path) as orders_file:
                    for line in orders_file:
                        user_data = ujson.loads(line)
                        # Generate examples from this user's data
                        for _, _, _, inputs, prediction in self._generate_user_examples(user_data, num_orders_per_user):
                            for k in inputs.keys():
                                all_inputs[k].append(inputs[k])
                            all_predictions.append(prediction)
                        # Return inputs and predictions if we have enough examples
                        while current_step < steps_per_epoch and len(all_predictions) >= batch_sizes[current_step]:
                            b = batch_sizes[current_step]
                            # Return a batch of examples
                            inputs = {}
                            for k in all_inputs.keys():
                                inputs[k] = np.array(all_inputs[k][:b])
                            yield inputs, np.array(all_predictions[:b])
                            # and remove it from the buffer
                            for k in all_inputs.keys():
                                del all_inputs[k][:b]
                            del all_predictions[:b]
                            current_step += 1
                # Flush the rest of the examples
                while current_step < steps_per_epoch:
                    b = batch_sizes[current_step]
                    # Return a batch of examples
                    inputs = {}
                    for k in all_inputs.keys():
                        inputs[k] = all_inputs[k][:b]
                    yield inputs, np.array(all_predictions[:b])
                    # and remove it from the buffer
                    for k in all_inputs.keys():
                        del all_inputs[k][:b]
                    del all_predictions[:b]
                    current_step += 1
                assert current_step == steps_per_epoch
                for k in all_inputs.keys():
                    assert len(all_inputs[k]) == 0
                assert len(all_predictions) == 0

        return generator(), steps_per_epoch

    def _hidden_layer_units(self, num_layers, from_dim, to_dim):
        units = np.linspace(from_dim, to_dim, num_layers + 2)[1:-1]
        units = np.round(units, 0).astype(np.int)
        return units

    def _build_model(self):
        product = layers.Input(shape=(1, ), dtype='int32', name='product')
        product_history = layers.Input(shape=(self.product_history, ), name='product_history')

        # Calculate the embedding vector of the product
        product_vector = layers.Embedding(name='product_embedding',
            input_dim=self.num_products, output_dim=self.product_embedding, input_length=1)(product)
        product_vector = layers.Flatten(name='product_vector')(product_vector)

        # Summarize the product purchase history
        input_size = self.product_history
        output_size = input_size // 2
        product_history_vector = layers.Dense((input_size + output_size) // 2, activation='tanh', name='product_history_hidden')(product_history)
        product_history_vector = layers.Dropout(rate=self.dropout)(product_history_vector)
        product_history_vector = layers.Dense(output_size, activation='tanh', name='product_history_vector')(product_history_vector)

        # Concatenate all the feature vectors
        features = [product_vector, product_history_vector]
        feature_vector = layers.concatenate(features, name='feature_vector')
        feature_vector_dim = self.product_embedding + (self.product_history // 2)

        # Hidden layers and prediction
        hidden = layers.Dropout(rate=self.dropout)(feature_vector)
        layer_units = self._hidden_layer_units(self.hidden_layers, feature_vector_dim, 1)
        for k, units in enumerate(layer_units):
            hidden_name = 'hidden_{}'.format(k + 1)
            hidden = layers.Dense(units, activation='relu', name=hidden_name)(hidden)
            hidden = layers.Dropout(rate=self.dropout)(hidden)
        prediction = layers.Dense(1, activation='sigmoid', name='prediction')(hidden)

        inputs = [product, product_history]
        model = models.Model(inputs=inputs, outputs=prediction)
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model


class FitMLPv2(_MLPv2, FitModel):

    def run(self):
        self.random = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        tf.set_random_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

        # Split the orders into training and validation sets and write them to separate files
        orders_path = self.requires()['orders'].output().path
        training_fd = tempfile.NamedTemporaryFile(mode='w+', delete=True)
        validation_fd = tempfile.NamedTemporaryFile(mode='w+', delete=True)
        with open(orders_path) as input_fd:
            for line in input_fd:
                if self.global_orders_ratio >= 1 or self.random.uniform() <= self.global_orders_ratio:
                    if self.random.uniform() <= self.validation_orders_ratio:
                        validation_fd.write(line)
                    else:
                        training_fd.write(line)
        validation_fd.flush()
        training_fd.flush()

        _, _, _, validation_inputs, validation_predictions = self._load_data(validation_fd.name)
        training_generator, steps_per_epoch = self._create_data_generator(
            training_fd.name, num_orders_per_user=self.num_orders_per_user, batch_size=self.batch_size)

        model = self._build_model()
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(os.path.abspath(self.output().path), verbose=1, save_weights_only=True, save_best_only=True),
        ]
        model.fit_generator(training_generator, steps_per_epoch,
                            validation_data=(validation_inputs, validation_predictions),
                            callbacks=callbacks, epochs=self.epochs, verbose=2)

        validation_fd.close()
        training_fd.close()


class _PredictMLPv2(_MLPv2, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitMLPv2(
            mode=self.mode,
            num_orders_per_user=self.num_orders_per_user,
            product_history=self.product_history,
            product_embedding=self.product_embedding,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            global_orders_ratio=self.global_orders_ratio,
            validation_orders_ratio=self.validation_orders_ratio,
            batch_size=self.batch_size,
            epochs=self.epochs)
        return req


class PredictMLPv2ReorderSizeKnown(_PredictMLPv2):

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
        self.random = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        tf.set_random_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

        orders_path = self.requires()['orders'].output().path
        _, order_ids, product_ids, inputs, _ = self._load_data(orders_path)
        reorder_size = self._determine_reorder_size()

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=self.batch_size, verbose=0).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': product_ids, 'score': scores})

        predictions = {}
        for order_id in set(order_ids):
            predictions[order_id] = []
            df = scores[scores.order_id == order_id].nlargest(reorder_size[order_id], 'score')
            for row in df.itertuples(index=False):
                # ujson fails when it tries to serialize the numpy int values
                predictions[int(order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizeMLPv2ReorderSizeKnown(luigi.Task):

    choices = {
        'num_orders_per_user': [3, 5, 7],
        'product_history': [91],
        'product_embedding': [5],
        'hidden_layers': [3],
        'dropout': [0.5],
    }

    def run(self):
        space = {k: hp.choice(k, v) for k, v in self.choices.items()}
        while True:
            values = sample(space)
            yield PredictMLPv2ReorderSizeKnown(
                mode='evaluation',
                num_orders_per_user=values['num_orders_per_user'],
                product_history=values['product_history'],
                product_embedding=values['product_embedding'],
                hidden_layers=values['hidden_layers'],
                dropout=values['dropout'],
                global_orders_ratio=0.25)


class PredictMLPv2Threshold(_PredictMLPv2):

    threshold = luigi.FloatParameter(default=0.173)

    @property
    def model_name(self):
        model_name = super().model_name
        model_name += '_threshold_{}'.format(self.threshold)
        return model_name

    def run(self):
        self.random = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        tf.set_random_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

        orders_path = self.requires()['orders'].output().path
        _, order_ids, product_ids, inputs, _ = self._load_data(orders_path)

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=self.batch_size, verbose=0).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': product_ids, 'score': scores})
        scores = scores[scores.score > self.threshold].sort_values('score', ascending=False)

        predictions = {}
        for order_id in set(order_ids):
            predictions[order_id] = []
        for row in scores.itertuples(index=False):
            # ujson fails when it tries to serialize the numpy int values
            predictions[int(row.order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizePredictMLPv2Threshold(luigi.Task):

    def run(self):
        for i in range(100):
            threshold = np.round(np.random.uniform(0.1, 0.2), 3)
            yield PredictMLPv2Threshold(threshold=threshold)


class PredictMLPv2ThresholdVariable(_PredictMLPv2):

    @property
    def model_name(self):
        model_name = super().model_name
        model_name += '_threshold_variable'
        return model_name

    def _determine_reorder_thresholds(self, model, scores):
        orders_path = self.requires()['orders'].output().path
        all_user_ids, all_order_ids, all_product_ids, all_inputs, all_targets = \
            self._load_data(orders_path, num_orders_per_user=self.num_orders_per_user)

        target_order_ids = set(scores.order_id)
        user_id_to_target_order_id = {}
        for i in range(len(all_user_ids)):
            if all_order_ids[i] in target_order_ids:
                user_id_to_target_order_id[all_user_ids[i]] = all_order_ids[i]
        mask = np.array([order_id not in target_order_ids for order_id in all_order_ids])

        for k in all_inputs.keys():
            all_inputs[k] = all_inputs[k][mask]
        all_predictions = model.predict(all_inputs, batch_size=self.batch_size, verbose=0).flatten()

        results = pd.DataFrame({
            'user_id': list(itertools.compress(all_user_ids, mask)),
            'order_id': list(itertools.compress(all_order_ids, mask)),
            'product_id': list(itertools.compress(all_product_ids, mask)),
            'prediction': all_predictions,
            'target': all_targets[mask],
        })

        # Find the best threshold value for each previous order by each user
        best_thresholds = defaultdict(list)
        grouped = results.groupby(['user_id', 'order_id'])
        for (user_id, order_id), group in grouped:
            product_ids = np.array(group['product_id'])
            reordered = set(product_ids[np.array(group['target']) > 0])
            probability = np.array(group['prediction'])
            if not reordered:
                best_threshold = probability.max()
            else:
                best_threshold, best_f1 = None, None
                for threshold in probability:
                    predicted = set(product_ids[probability >= threshold])
                    tp = len(predicted & reordered)
                    precision = tp / len(predicted)
                    recall = tp / len(reordered)
                    f1 = 2.0 * (precision * recall) / (precision + recall) if precision or recall else 0.0
                    if best_f1 is None or f1 > best_f1:
                        best_threshold = threshold
                        best_f1 = f1
            best_thresholds[user_id].append(best_threshold)

        # Select the average threshold for each user
        reorder_thresholds = {}
        for user_id in user_id_to_target_order_id:
            order_id = user_id_to_target_order_id[user_id]
            reorder_thresholds[order_id] = np.mean(best_thresholds[user_id])

        return reorder_thresholds

    def run(self):
        self.random = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        tf.set_random_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        orders_path = self.requires()['orders'].output().path
        user_ids, order_ids, product_ids, inputs, _ = self._load_data(orders_path)
        scores = model.predict(inputs, batch_size=self.batch_size, verbose=0).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': product_ids, 'score': scores})

        reorder_thresholds = self._determine_reorder_thresholds(model, scores)

        predictions = {}
        for order_id in set(order_ids):
            predictions[order_id] = []
            df = scores[scores.order_id == order_id]
            df = df[df.score >= reorder_thresholds[order_id]]
            for row in df.itertuples(index=False):
                # ujson fails when it tries to serialize the numpy int values
                predictions[int(order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizeMLPv2ThresholdVariable(luigi.Task):

    choices = {
        'num_orders_per_user': [3, 5, 7],
        'product_history': [91],
        'product_embedding': [5],
        'hidden_layers': [3],
        'dropout': [0.5],
    }

    def run(self):
        space = {k: hp.choice(k, v) for k, v in self.choices.items()}
        while True:
            values = sample(space)
            yield PredictMLPv2ThresholdVariable(
                mode='evaluation',
                num_orders_per_user=values['num_orders_per_user'],
                product_history=values['product_history'],
                product_embedding=values['product_embedding'],
                hidden_layers=values['hidden_layers'],
                dropout=values['dropout'],
                global_orders_ratio=0.25)


class PredictMLPv2EMU(_PredictMLPv2):

    @property
    def model_name(self):
        model_name = super().model_name
        model_name += '_emu'
        return model_name

    def _determine_reorder_size(self, scores):
        reorder_size = {}
        grouped = scores.groupby('order_id')
        P_values = grouped['score'].apply(list)
        with multiprocessing.Pool() as pool:
            results = pool.map(maximize_expected_f1, P_values)
        for order_id, (best_k, max_f1) in zip(grouped.groups.keys(), results):
            reorder_size[order_id] = best_k
        return reorder_size

    def run(self):
        self.random = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        tf.set_random_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

        orders_path = self.requires()['orders'].output().path
        _, order_ids, product_ids, inputs, _ = self._load_data(orders_path)

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=self.batch_size, verbose=0).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': product_ids, 'score': scores})
        reorder_size = self._determine_reorder_size(scores)

        predictions = {}
        for order_id in set(order_ids):
            predictions[order_id] = []
            df = scores[scores.order_id == order_id].nlargest(reorder_size[order_id], 'score')
            for row in df.itertuples(index=False):
                # ujson fails when it tries to serialize the numpy int values
                predictions[int(order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
