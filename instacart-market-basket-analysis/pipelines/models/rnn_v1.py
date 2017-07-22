import os
import subprocess
import tempfile
from contextlib import contextmanager

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Masking, Reshape, TimeDistributed, Dense, Bidirectional, LSTM, dot
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from ..models import FitModel, PredictModel
from ..clean_data import Products
from ..config import OUTPUT_DIR


# https://github.com/fireeye/tf_rl_tutorial/blob/master/tf_rl_tutorial/models.py#L52
# The model should output alternating positive/negative pairs: [pos, neg, pos, neg, ...]
def hinge_loss(y_true, y_pred, margin=1.0):
    y_pairs = tf.reshape(y_pred, [-1, 2])
    pos_scores, neg_scores = tf.split(y_pairs, 2, 1)
    losses = tf.nn.relu(margin - pos_scores + neg_scores)
    return tf.reduce_mean(losses)


class _RNNv1(object):

    embedding_dim = luigi.IntParameter(default=10)
    max_products_per_user = luigi.IntParameter(default=100)
    users_per_batch = luigi.IntParameter(default=32)
    negative_products_ratio = luigi.IntParameter(default=2)

    random_seed = luigi.IntParameter(default=3996193, significant=False)

    num_products = Products.count()

    @property
    def model_name(self):
        params = [
            self.embedding_dim,
            self.max_products_per_user,
            self.users_per_batch,
            self.negative_products_ratio,
        ]
        model_name = 'rnn_v1_{}'.format('_'.join(str(p) for p in params))
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

    def _generate_order_samples(self, last_order, prior_orders):
        # Collect the indices of the reordered products
        pos_products = set()
        for product in last_order['products']:
            if product['reordered']:
                pos_products.add(product['product_id'])

        neg_products = set()
        previously_ordered = []
        for order in prior_orders:
            for product in order['products']:
                # Collect the sequence of previously ordered products
                previously_ordered.append(product['product_id'])
                # Collect the previously ordered products that were not reordered
                if product['product_id'] not in pos_products:
                    neg_products.add(product['product_id'])
        neg_products = tuple(neg_products)

        if pos_products and neg_products:
            # Take the most recently ordered product
            user_input = pad_sequences([previously_ordered],
                    maxlen=self.max_products_per_user, padding='pre', truncating='pre')[0]
            user_input = np.expand_dims(user_input, axis=1)

            # Add alternating positive/negative examples, sampling a number of
            # negative products for each positive product
            for pos_product_input in pos_products:
                for i in range(self.negative_products_ratio):
                    neg_product_input = self.random.choice(neg_products)
                    yield user_input, pos_product_input
                    yield user_input, neg_product_input

    def _generate_samples(self, user_data, include_prior_orders=False):
        if include_prior_orders:
            for k in range(1, len(user_data['prior_orders'])):
                last_order = user_data['prior_orders'][k]
                prior_orders = user_data['prior_orders'][:k]
                yield from self._generate_order_samples(last_order, prior_orders)
        yield from self._generate_order_samples(user_data['last_order'], user_data['prior_orders'])

    def create_generator(self, orders_path, users_per_batch, shuffle=False, include_prior_orders=False):

        num_users = self._count_lines(orders_path)
        batch_sizes = [len(a) for a in np.array_split(range(num_users), num_users / users_per_batch)]

        def generator():
            while True:
                with (self._open_shuffled(orders_path) if shuffle else open(orders_path)) as orders_file:
                    current_batch_num = 0
                    current_batch_size = 0
                    user_inputs, product_inputs = [], []
                    for line in orders_file:
                        user_data = ujson.loads(line)
                        # Generate samples from this user's data
                        for user_input, product_input in self._generate_samples(user_data, include_prior_orders):
                            user_inputs.append(user_input)
                            product_inputs.append(product_input)
                        current_batch_size += 1
                        # Yield inputs and targets if the batch is complete
                        if current_batch_size == batch_sizes[current_batch_num]:
                            user_inputs = np.array(user_inputs)
                            product_inputs = np.array(product_inputs)
                            inputs = {'user_input': user_inputs, 'product_input': product_inputs}
                            target = np.zeros(user_inputs.shape[0])  # Not used!
                            yield inputs, target
                            # Reset the current batch
                            user_inputs, product_inputs = [], []
                            current_batch_size = 0
                            current_batch_num += 1

        return generator(), len(batch_sizes)


class FitRNNv1(_RNNv1, FitModel):

    def _build_model(self):
        product = Input(shape=(1, ))
        product_embedded = Embedding(input_dim=self.num_products + 1, output_dim=self.embedding_dim, input_length=1)(product)
        product_embedded = Reshape(target_shape=(self.embedding_dim, ))(product_embedded)
        product_embedding = Model(inputs=product, outputs=product_embedded, name='product_embedding')

        user_input = Input(shape=(self.max_products_per_user, 1), name='user_input')
        user_embedded = Masking(name='masking')(user_input)
        user_embedded = TimeDistributed(product_embedding, name='user_embedded')(user_embedded)
        user_embedded = Bidirectional(LSTM(self.embedding_dim), merge_mode='concat', name='lstm')(user_embedded)
        user_embedded = Dense(self.embedding_dim, activation='relu', name='hidden')(user_embedded)

        product_input = Input(shape=(1, ), name='product_input')
        product_embedded = product_embedding(product_input)

        score = dot([user_embedded, product_embedded], axes=[-1, -1], name='score')

        model = Model(inputs=[product_input, user_input], outputs=score)

        model.compile(loss=hinge_loss, optimizer='adam')

        return model

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as training_fd:
            with tempfile.NamedTemporaryFile(mode='w+', delete=True) as validation_fd:
                # Split the orders file into training and validation
                with open(orders_path) as input_fd:
                    for line in input_fd:
                        if self.random.uniform() <= 0.1:
                            validation_fd.write(line)
                        else:
                            training_fd.write(line)
                validation_fd.flush()
                training_fd.flush()
                # Fit the model
                training_generator, training_steps_per_epoch = \
                    self.create_generator(training_fd.name, users_per_batch=self.users_per_batch,
                                          shuffle=True, include_prior_orders=True)
                validation_generator, validation_steps_per_epoch = \
                    self.create_generator(validation_fd.name, users_per_batch=self.users_per_batch,
                                          shuffle=False, include_prior_orders=False)

                model = self._build_model()
                model.summary()

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10),
                    ModelCheckpoint(self.output().path, verbose=1, save_best_only=True),
                    TensorBoard(log_dir=os.path.join(OUTPUT_DIR, 'tensorboard', self.model_name), write_graph=False),
                ]
                model.fit_generator(training_generator, training_steps_per_epoch,
                                    validation_data=validation_generator,
                                    validation_steps=validation_steps_per_epoch,
                                    epochs=1000, verbose=1, callbacks=callbacks)

                # Best model saved by the ModelCheckpoint callback
                # model.save(self.output().path)


class PredictRNNv1(_RNNv1, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitRNNv1(
            mode=self.mode,
            embedding_dim=self.embedding_dim,
            max_products_per_user=self.max_products_per_user,
            users_per_batch=self.users_per_batch,
            negative_products_ratio=self.negative_products_ratio)
        return req

    def run(self):
        assert self.mode == 'evaluation'
        self.random = RandomState(self.random_seed)

        predictions = {}

        model = load_model(self.input()['model'].path, custom_objects={'hinge_loss': hinge_loss})
        model.summary()

        orders_path = self.requires()['orders'].output().path
        with open(orders_path) as f:
            for line in f:
                user_data = ujson.loads(line)

                order_id = int(user_data['last_order']['order_id'])
                predictions[order_id] = []

                num_reordered = 0
                for product in user_data['last_order']['products']:
                    if product['reordered']:
                        num_reordered += 1

                # Define the user input and collect all the previously ordered products
                previously_ordered = []
                for order in user_data['prior_orders']:
                    for product in order['products']:
                        previously_ordered.append(product['product_id'])
                product_inputs = np.array(tuple(set(previously_ordered)))
                user_input = pad_sequences([previously_ordered],
                        maxlen=self.max_products_per_user, padding='pre', truncating='pre')[0]
                user_input = np.expand_dims(user_input, axis=1)
                user_inputs = np.repeat(user_input[np.newaxis, :], product_inputs.shape[0], axis=0)
                inputs = {'user_input': user_inputs, 'product_input': product_inputs}

                # Compute the score for each previously ordered product
                scores = model.predict(inputs).flatten()
                df = pd.DataFrame({'product_id': product_inputs, 'score': scores})
                df = df.nlargest(num_reordered, 'score')
                for row in df.itertuples(index=False):
                    # ujson fails when it tries to serialize the numpy int values
                    predictions[order_id].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
