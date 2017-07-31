import os
import sys
import pprint
import tempfile

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from ..models import FitModel, PredictModel
from ..clean_data import Products

from .utils import ExamplesGenerator, hidden_layer_units


# class ProductEmbedding(layers.Embedding):
#
#     def __init__(self, num_products, embedding_dim, **kwargs):
#         super().__init__(num_products, embedding_dim, **kwargs)
#         self.supports_masking = False
#
#     def compute_mask(self, inputs, mask=None):
#         return None
#
#     def compute_output_shape(self, input_shape):
#         output_shape = (None, self.output_dim)
#         return output_shape
#
#     def call(self, inputs):
#         # Get the embedding vector for each product ID
#         x = K.maximum(inputs - 1, 0)
#         emb = K.squeeze(super().call(x), 1)
#         # Return an all-zeros vector if the product ID was 0, otherwise return the embedding vector
#         out = K.cast(K.not_equal(inputs, 0), K.floatx()) * emb
#         return out


class _RNNv3(ExamplesGenerator):

    max_days = luigi.IntParameter(default=91)
    max_products_per_day = luigi.IntParameter(default=15)
    max_prior_orders = luigi.IntParameter(default=3)
    embedding_dim = luigi.IntParameter(default=10)
    lstm_layers = luigi.IntParameter(default=2)
    lstm_units = luigi.IntParameter(default=10)
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
            self.max_days,
            self.max_products_per_day,
            self.max_prior_orders,
            self.embedding_dim,
            self.lstm_layers,
            self.lstm_units,
            self.hidden_layers,
            self.dropout,
        ]
        model_name = 'rnn_v3_{}'.format('_'.join(str(p).lower() for p in params))
        return model_name

    def _generate_examples(self, last_order, prior_orders):
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

    def _build_model(self):
        # Inputs:
        # - product: query product
        # - orders: products ordered in the past max_days days
        product = layers.Input(shape=(1,), dtype='int32', name='product')
        orders = layers.Input(shape=(self.max_days, self.max_products_per_day), name='orders')

        # Define the product embedding layer (used in two places below)
        # product_embedding = ProductEmbedding(self.num_products, self.embedding_dim, name='product_vector')
        product_id = layers.Input(shape=(1,), dtype='int32')
        product_vector = layers.Embedding(self.num_products + 1, self.embedding_dim)(product_id)
        product_vector = layers.Flatten()(product_vector)
        product_embedding = models.Model(inputs=product_id, outputs=product_vector, name='product_vector')

        # Compute the embedding for the query product
        product_vector = product_embedding(product)

        # Flatten the orders and compute the embedding for the previously ordered products
        order_vectors = layers.Reshape((self.max_days * self.max_products_per_day, 1))(orders)
        order_vectors = layers.TimeDistributed(product_embedding, name='order_vectors')(order_vectors)

        # Calculate the dot product between the query product and each previously ordered product
        f = lambda x: K.batch_dot(x[0], x[1], axes=(1, 2))
        similarities = layers.Lambda(f, name='similarities')([product_vector, order_vectors])

        # Apply batch normalization
        similarities = layers.Reshape((1, self.max_days * self.max_products_per_day))(similarities)
        similarities = layers.BatchNormalization(axis=-2, name='batch_norm')(similarities)

        # Reshape the similarities back into a sequence with one element per day
        similarities = layers.Reshape((self.max_days, self.max_products_per_day))(similarities)

        # Recurrent layers
        lstm = similarities
        for k in range(self.lstm_layers - 1):
            layer = layers.LSTM(self.lstm_units, return_sequences=True)
            lstm = layers.Bidirectional(layer, merge_mode='concat', name='lstm_{}'.format(k + 1))(lstm)
        layer = layers.LSTM(self.lstm_units)
        lstm = layers.Bidirectional(layer, merge_mode='concat', name='lstm_{}'.format(self.lstm_layers))(lstm)

        hidden = layers.Dropout(rate=self.dropout)(lstm)
        layer_units = hidden_layer_units(self.hidden_layers, 2 * self.lstm_units, 1)
        for k, units in enumerate(layer_units):
            hidden_name = 'hidden_{}'.format(k + 1)
            hidden = layers.Dense(units, activation='relu', name=hidden_name)(hidden)
            hidden = layers.Dropout(rate=self.dropout)(hidden)

        prediction = layers.Dense(1, activation='sigmoid', name='prediction')(hidden)

        model = models.Model(inputs=[product, orders], outputs=prediction)
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model


class FitRNNv3(_RNNv3, FitModel):

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

        _, validation_inputs, validation_predictions = self._load_data(validation_fd.name)
        training_generator, training_steps_per_epoch = \
            self._create_data_generator(training_fd.name, self.max_prior_orders, self.batch_size)

        model = self._build_model()
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(os.path.abspath(self.output().path), verbose=1, save_weights_only=True, save_best_only=True),
        ]
        class_weight = compute_class_weight('balanced', [0, 1], validation_predictions)
        class_weight = dict(enumerate(class_weight))
        model.fit_generator(training_generator, training_steps_per_epoch,
                            validation_data=(validation_inputs, validation_predictions),
                            callbacks=callbacks, class_weight=class_weight,
                            epochs=self.epochs, verbose=1)

        validation_fd.close()
        training_fd.close()


class _PredictRNNv3(_RNNv3, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitRNNv3(
            mode=self.mode,
            max_days=self.max_days,
            max_products_per_day=self.max_products_per_day,
            max_prior_orders=self.max_prior_orders,
            embedding_dim=self.embedding_dim,
            lstm_layers=self.lstm_layers,
            lstm_units=self.lstm_units,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            global_orders_ratio=self.global_orders_ratio,
            validation_orders_ratio=self.validation_orders_ratio,
            batch_size=self.batch_size,
            epochs=self.epochs)
        return req


class PredictRNNv3ReorderSizeKnown(_PredictRNNv3):

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
        np.random.seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))
        tf.set_random_seed(int.from_bytes(self.random.bytes(4), byteorder=sys.byteorder))

        orders_path = self.requires()['orders'].output().path
        order_ids, inputs, _ = self._load_data(orders_path)

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=self.batch_size, verbose=0).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': inputs['product'], 'score': scores})

        reorder_size = self._determine_reorder_size()

        predictions = {}
        for order_id in order_ids:
            predictions[order_id] = []
            df = scores[scores.order_id == order_id].nlargest(reorder_size[order_id], 'score')
            for row in df.itertuples(index=False):
                # ujson fails when it tries to serialize the numpy int values
                predictions[int(order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizeRNNv3ReorderSizeKnown(luigi.Task):

    choices = {
        'max_days': [31, 91],
        'max_products_per_day': [20],
        'max_prior_orders': [2],
        'embedding_dim': [10],
        'lstm_layers': [1, 2],
        'lstm_units': [10],
        'hidden_layers': [2, 3],
        'dropout': [0.5],
    }

    def run(self):
        space = {k: hp.choice(k, v) for k, v in self.choices.items()}
        while True:
            values = sample(space)
            yield PredictRNNv3ReorderSizeKnown(
                mode='evaluation',
                max_days=values['max_days'],
                max_products_per_day=values['max_products_per_day'],
                max_prior_orders=values['max_prior_orders'],
                embedding_dim=values['embedding_dim'],
                lstm_layers=values['lstm_layers'],
                lstm_units=values['lstm_units'],
                hidden_layers=values['hidden_layers'],
                dropout=values['dropout'],
                global_orders_ratio=0.25,
                validation_orders_ratio=0.1,
                batch_size=1024,
                epochs=10)


class PredictRNNv3Threshold(_PredictRNNv3):

    threshold = luigi.FloatParameter(default=0.29)

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
        order_ids, inputs, _ = self._load_data(orders_path)

        model = self._build_model()
        model.load_weights(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=self.batch_size, verbose=1).flatten()
        scores = pd.DataFrame({'order_id': order_ids, 'product_id': inputs['product'], 'score': scores})
        scores = scores[scores.score > self.threshold].sort_values('score', ascending=False)

        predictions = {}
        for order_id in order_ids:
            predictions[order_id] = []
        for row in scores.itertuples(index=False):
            # ujson fails when it tries to serialize the numpy int values
            predictions[int(row.order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


class OptimizePredictRNNv3Threshold(luigi.Task):

    def run(self):
        for i in range(20):
            threshold = np.round(np.random.uniform(0.0, 1.0), 2)
            yield PredictRNNv3Threshold(threshold=threshold)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
