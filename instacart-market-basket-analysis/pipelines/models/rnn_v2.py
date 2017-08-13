import os
import pprint
import subprocess
import tempfile
from collections import defaultdict
from contextlib import contextmanager

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from ..models import FitModel, PredictModel
from ..clean_data import Products
from ..config import OUTPUT_DIR


class RNNv2(object):

    max_prior_orders = luigi.IntParameter(default=3)
    max_days = luigi.IntParameter(default=31)
    max_products_per_day = luigi.IntParameter(default=20)
    product_embedding_dim = luigi.IntParameter(default=10)
    days_attention_layers = luigi.IntParameter(default=3)
    days_attention_activation = luigi.Parameter(default='relu')
    lstm_units = luigi.IntParameter(default=10)
    hidden_layers = luigi.IntParameter(default=3)
    hidden_layers_activation = luigi.Parameter(default='relu')
    optimizer = luigi.Parameter(default='adam')

    random_seed = luigi.IntParameter(default=3996193, significant=False)
    global_orders_ratio = luigi.FloatParameter(default=1.0, significant=False)
    validation_orders_ratio = luigi.FloatParameter(default=0.1, significant=False)
    users_per_batch = luigi.IntParameter(default=32, significant=False)
    epochs = luigi.IntParameter(default=1000)

    num_products = Products.count()

    @property
    def model_name(self):
        params = [
            self.max_prior_orders,
            self.max_days,
            self.max_products_per_day,
            self.product_embedding_dim,
            self.days_attention_layers,
            self.days_attention_activation,
            self.lstm_units,
            self.hidden_layers,
            self.hidden_layers_activation,
            self.optimizer,
        ]
        model_name = 'rnn_v2_{}'.format('_'.join(str(p) for p in params))
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

    def _generate_order_examples(self, last_order, prior_orders):
        assert self.max_days >= 31, 'max_days should be >= 31, since at least one prior order is needed'

        # Collect the sequence of products that was ordered each day for the last max_days days
        # orders[0] corresponds to orders placed the same day
        orders = [[] for _ in range(self.max_days)]
        num_days = last_order['days_since_prior_order']
        for order in reversed(prior_orders):
            for product in order['products']:
                orders[num_days].append(product['product_id'])
            if order['days_since_prior_order'] is not None:
                # There is at least another order, stop if it will go over the limit
                num_days += order['days_since_prior_order']
                if num_days >= self.max_days:
                    break

        ordered_within_max_days = set(p for d in orders for p in d)
        orders = pad_sequences(orders, maxlen=self.max_products_per_day, padding='post', truncating='post')

        # Return the positive examples
        positive_examples = set()
        if last_order['products']:
            for product in last_order['products']:
                product_id = product['product_id']
                if product['reordered'] and product_id in ordered_within_max_days:
                    yield last_order['order_id'], product_id, orders, 1.0
                    positive_examples.add(product_id)

        # Return the negative examples
        for product_id in (ordered_within_max_days - positive_examples):
            yield last_order['order_id'], product_id, orders, 0.0

    def _generate_user_examples(self, user_data, max_prior_orders=1):
        yield from self._generate_order_examples(user_data['last_order'], user_data['prior_orders'])
        max_prior_orders -= 1
        if max_prior_orders > 0:
            for k in range(len(user_data['prior_orders']) - 1, 0, -1):
                last_order = user_data['prior_orders'][k]
                prior_orders = user_data['prior_orders'][:k]
                yield from self._generate_order_examples(last_order, prior_orders)
                max_prior_orders -= 1
                if max_prior_orders == 0:
                    break

    def _load_data(self, orders_path):
        order_ids = []
        inputs = defaultdict(list)
        predictions = []

        def add_example(order_id, product, orders, prediction):
            order_ids.append(order_id)
            inputs['product'].append(product)
            inputs['orders'].append(orders)
            predictions.append(prediction)

        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                for order_id, product, orders, prediction in \
                        self._generate_user_examples(user_data, max_prior_orders=1):
                    add_example(order_id, product, orders, prediction)

        # Build the numpy arrays
        inputs['product'] = np.array(inputs['product'])
        inputs['orders'] = np.array(inputs['orders'])
        predictions = np.array(predictions)

        return order_ids, inputs, predictions

    def _create_data_generator(self, orders_path, users_per_batch, max_prior_orders):
        num_users = self._count_lines(orders_path)
        batch_sizes = [len(a) for a in np.array_split(range(num_users), num_users / users_per_batch)]

        def generator():
            while True:
                with self._open_shuffled(orders_path) as orders_file:
                    current_batch_num = 0
                    current_batch_size = 0
                    product_inputs, orders_inputs, predictions = [], [], []
                    for line in orders_file:
                        user_data = ujson.loads(line)
                        # Generate examples from this user's data
                        for order_id, product, orders, prediction in \
                                self._generate_user_examples(user_data, max_prior_orders):
                            product_inputs.append(product)
                            orders_inputs.append(orders)
                            predictions.append(prediction)
                        current_batch_size += 1
                        # Return inputs and predictions if the batch is complete
                        if current_batch_size == batch_sizes[current_batch_num]:
                            inputs = {'product': np.array(product_inputs), 'orders': np.array(orders_inputs)}
                            predictions = np.array(predictions)
                            yield inputs, predictions
                            # Reset the current batch
                            product_inputs, orders_inputs, predictions = [], [], []
                            current_batch_size = 0
                            current_batch_num += 1

        return generator(), len(batch_sizes)

    def _hidden_layer_units(self, num_layers, from_dim, to_dim):
        units = np.linspace(from_dim, to_dim, num_layers + 2)[1:-1]
        units = np.round(units, 0).astype(np.int)
        return units

    def _build_product_embedding_submodel(self):
        product = layers.Input(shape=(1, ), dtype='int32', name='product')
        product_emb = layers.Embedding(self.num_products + 1, self.product_embedding_dim, name='product_emb')(product)
        product_emb = layers.Reshape((self.product_embedding_dim, ), name='product_emb_reshaped')(product_emb)
        product_emb = models.Model(inputs=product, outputs=product_emb, name='product_emb')
        # product_emb.summary()
        return product_emb

    def _build_days_attention_submodel(self):
        product_emb = layers.Input(shape=(self.product_embedding_dim, ), name='product_emb')
        hidden = product_emb
        layer_units = self._hidden_layer_units(self.days_attention_layers, self.product_embedding_dim, self.max_days)
        for k, units in enumerate(layer_units):
            hidden_name = 'days_attn_hidden_{}'.format(k + 1)
            hidden = layers.Dense(units, activation=self.days_attention_activation, name=hidden_name)(hidden)
        attn = layers.Dense(self.max_days, activation='softmax', name='attn')(hidden)
        days_attn = models.Model(inputs=product_emb, outputs=attn, name='days_attn')
        # days_attn.summary()
        return days_attn

    def _build_model(self):
        # Inputs:
        # - product: query product
        # - orders: products ordered in the past max_days days
        product = layers.Input(shape=(1, ), dtype='int32', name='product')
        orders = layers.Input(shape=(self.max_days, self.max_products_per_day), name='orders')

        # Submodels
        product_emb_model = self._build_product_embedding_submodel()
        days_attn_model = self._build_days_attention_submodel()

        # Compute the embedding for the query product
        product_emb = product_emb_model(product)

        # Flatten the orders
        orders_emb = layers.Reshape((self.max_days * self.max_products_per_day, 1))(orders)

        # Compute the embedding for the previously ordered products
        orders_emb = layers.Masking(mask_value=0.0, name='mask_zero')(orders_emb)
        orders_emb = layers.TimeDistributed(product_emb_model, name='orders_emb')(orders_emb)

        # Calculate the dot product between the query product and each previously ordered product
        # (see https://github.com/fchollet/keras/issues/6151 for a batch_dot example)
        f = lambda x: K.batch_dot(x[0], x[1], axes=(1, 2))
        sim = layers.Lambda(f, name='sim')([product_emb, orders_emb])

        # Reshape it back into a sequence with one element per day
        sim = layers.Reshape((self.max_days, self.max_products_per_day))(sim)

        # Compute the attention vector with one entry for each day
        days_attn = days_attn_model(product_emb)
        rep = int(self.max_products_per_day)  # Fixes serialization issues
        f = lambda x: K.repeat_elements(K.expand_dims(x), rep, 2)
        repeated_days_attn = layers.Lambda(f, name='repeated_days_attn')(days_attn)

        # Scale each element of the sequence by the attention value
        scaled_sim = layers.multiply([sim, repeated_days_attn], name='scaled_sim')

        lstm = layers.Bidirectional(layers.LSTM(self.lstm_units), merge_mode='concat', name='lstm')(scaled_sim)

        hidden = lstm
        layer_units = self._hidden_layer_units(self.hidden_layers, 2 * self.lstm_units, 1)
        for k, units in enumerate(layer_units):
            hidden_name = 'hidden_{}'.format(k + 1)
            hidden = layers.Dense(units, activation=self.hidden_layers_activation, name=hidden_name)(hidden)

        prediction = layers.Dense(1, activation='sigmoid', name='prediction')(hidden)

        model = models.Model(inputs=[product, orders], outputs=prediction)
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        # model.summary()

        return model


class FitRNNv2(RNNv2, FitModel):

    def run(self):
        self.random = RandomState(self.random_seed)

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

        _, validation_inputs, validation_predictions = self._load_data(validation_fd.name)
        training_generator, training_steps_per_epoch = \
            self._create_data_generator(training_fd.name, self.users_per_batch, max_prior_orders=self.max_prior_orders)

        model = self._build_model()
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(os.path.abspath(self.output().path), verbose=1, save_weights_only=True, save_best_only=True),
        ]
        if self.mode == 'evaluation':
            log_dir = os.path.join(OUTPUT_DIR, 'tensorboard', self.mode, self.model_name)
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=False))

        model.fit_generator(training_generator, training_steps_per_epoch,
                            validation_data=(validation_inputs, validation_predictions),
                            epochs=self.epochs, verbose=2, callbacks=callbacks)

        validation_fd.close()
        training_fd.close()


class _PredictRNNv2(RNNv2, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitRNNv2(
            mode=self.mode,
            max_prior_orders=self.max_prior_orders,
            max_days=self.max_days,
            max_products_per_day=self.max_products_per_day,
            product_embedding_dim=self.product_embedding_dim,
            days_attention_layers=self.days_attention_layers,
            days_attention_activation=self.days_attention_activation,
            lstm_units=self.lstm_units,
            hidden_layers=self.hidden_layers,
            hidden_layers_activation=self.hidden_layers_activation,
            optimizer=self.optimizer,
            global_orders_ratio=self.global_orders_ratio,
            validation_orders_ratio=self.validation_orders_ratio,
            users_per_batch=self.users_per_batch,
            epochs=self.epochs)
        return req


class PredictRNNv2ReorderSizeKnown(_PredictRNNv2):

    @staticmethod
    def _count_reordered_products(order):
        k = 0
        for product in order['products']:
            if product['reordered']:
                k += 1
        return k

    def _determine_reorder_size(self):
        assert self.mode == 'evaluation'
        num_reordered = {}
        orders_path = self.requires()['orders'].output().path
        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                order_id = int(user_data['last_order']['order_id'])
                num_reordered[order_id] = self._count_reordered_products(user_data['last_order'])
        return num_reordered

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        order_ids, inputs, _ = self._load_data(orders_path)

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=1024, verbose=1).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': inputs['product'], 'score': scores})

        reorder_size = self._determine_reorder_size()

        predictions = {}
        for order_id in set(order_ids):
            predictions[order_id] = []
            df = scores[scores.order_id == order_id].nlargest(reorder_size[order_id], 'score')
            for row in df.itertuples(index=False):
                # ujson fails when it tries to serialize the numpy int values
                predictions[int(order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizeRNNv2ReorderSizeKnown(luigi.Task):

    choices = {
        'max_prior_orders': [1],
        'max_days': [31, 91],
        'max_products_per_day': [10, 20],
        'product_embedding_dim': [10, 30],
        'days_attention_layers': [1, 2, 3],
        'days_attention_activation': ['relu'],
        'lstm_units': [10, 25, 50],
        'hidden_layers': [1, 2, 3],
        'hidden_layers_activation': ['relu'],
        'optimizer': ['adam'],
    }

    def run(self):
        from hyperopt import hp
        from hyperopt.pyll.stochastic import sample
        space = {k: hp.choice(k, v) for k, v in self.choices.items()}
        while True:
            values = sample(space)
            yield PredictRNNv2ReorderSizeKnown(
                mode='evaluation',
                max_days=values['max_days'],
                max_products_per_day=values['max_products_per_day'],
                product_embedding_dim=values['product_embedding_dim'],
                days_attention_layers=values['days_attention_layers'],
                days_attention_activation=values['days_attention_activation'],
                lstm_units=values['lstm_units'],
                hidden_layers=values['hidden_layers'],
                hidden_layers_activation=values['hidden_layers_activation'],
                optimizer=values['optimizer'],
                global_orders_ratio=0.25,
                validation_orders_ratio=0.1,
                users_per_batch=8,
                epochs=10)


class PredictRNNv2ReorderSizePercentile(PredictRNNv2ReorderSizeKnown):

    percentile = luigi.IntParameter(default=50)

    @property
    def model_name(self):
        model_name = super().model_name
        model_name += '_percentile_{}'.format(self.percentile)
        return model_name

    def _determine_reorder_size(self):
        reorder_size = {}
        orders_path = self.requires()['orders'].output().path
        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                order_id = int(user_data['last_order']['order_id'])
                num_reordered_values = []
                for order in user_data['prior_orders']:
                    num_reordered = self._count_reordered_products(order)
                    num_reordered_values.append(num_reordered)
                reorder_size[order_id] = int(np.percentile(num_reordered_values, self.percentile))
        return reorder_size


class OptimizePredictRNNv2ReorderSizePercentile(luigi.Task):

    def run(self):
        for percentile in [50, 60, 70, 80, 90]:
            yield PredictRNNv2ReorderSizePercentile(percentile=percentile)


class PredictRNNv2Threshold(_PredictRNNv2):

    threshold = luigi.FloatParameter(default=0.2)

    @property
    def model_name(self):
        model_name = super().model_name
        model_name += '_threshold_{}'.format(self.threshold)
        return model_name

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        order_ids, inputs, _ = self._load_data(orders_path)

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=1024, verbose=1).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': inputs['product'], 'score': scores})
        scores = scores[scores.score > self.threshold].sort_values('score', ascending=False)

        predictions = {}
        for order_id in set(order_ids):
            predictions[order_id] = []
        for row in scores.itertuples(index=False):
            # ujson fails when it tries to serialize the numpy int values
            predictions[int(row.order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizePredictRNNv2Threshold(luigi.Task):

    def run(self):
        for i in range(20):
            threshold = np.round(np.random.uniform(0.0, 0.5), 2)
            yield PredictRNNv2Threshold(threshold=threshold)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
