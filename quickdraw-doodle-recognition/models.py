import tensorflow as tf

import numpy as np
from tqdm import tqdm

import common


def get_model_wrapper(_config):
    if _config['model'] == 'convnet':
        return ConvNetWrapper(_config)
    else:
        raise ValueError("Invalid model '{}'".format(_config['model']))


class _BaseModelWrapper(object):

    def __init__(self, _config):
        self._config = _config
        self._global_step = tf.train.get_or_create_global_step()

    def train(self, optimizer, dataset):
        raise NotImplementedError

    def validate(self, dataset):
        raise NotImplementedError

    def predict(self, dataset):
        raise NotImplementedError

    # MAP@K (assuming only one positive example)
    def calculate_map(self, scores, labels, k=3):
        # Find the top k predictions
        _, predicted = tf.nn.top_k(scores, k)
        # Compare each prediction with the correct label
        actual = tf.tile(tf.expand_dims(labels, -1), [1, k])
        # Find the indices of the correct matches, calculate precision and
        # average across all the examples
        positives = tf.where(tf.equal(predicted, actual)).numpy()
        value = (1 / (positives[:, 1] + 1)).sum() / labels.shape[0].value
        return value


class CustomFlatten(tf.keras.Model):

    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pool = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding='valid', name='max_pool')
        self.avg_pool = tf.keras.layers.AvgPool2D(
            pool_size=3, strides=2, padding='valid', name='avg_pool')
        self.concat = tf.keras.layers.Concatenate(name='concat')
        self.bn = tf.keras.layers.BatchNormalization(name='bn')
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1,
            padding='same', activation='relu', name='conv')
        self.flatten = tf.keras.layers.Flatten(name='flatten')

    def call(self, input_tensor, training=False):
        concat_pooling = self.concat(
            [self.max_pool(input_tensor), self.avg_pool(input_tensor)])
        output = self.flatten(self.conv(self.bn(concat_pooling)))
        return output


class ConvNetWrapper(_BaseModelWrapper):

    def __init__(self, _config):
        super().__init__(_config)
        input_shape = (_config['image_size'], _config['image_size'], 1)
        num_classes = len(common.WORD2LABEL)
        self.model = tf.keras.Sequential([
            # Block A
            tf.keras.layers.Conv2D(
                filters=_config['base_filters'], kernel_size=7, strides=3,
                padding='same', activation='relu', name='conv-a',
                input_shape=input_shape),
            tf.keras.layers.MaxPool2D(
                pool_size=3, strides=2, padding='valid', name='maxpool-a'),
            # Block B
            tf.keras.layers.BatchNormalization(name='bn-b'),
            tf.keras.layers.Conv2D(
                filters=2 * _config['base_filters'], kernel_size=5,
                padding='same', activation='relu', name='conv-b'),
            tf.keras.layers.MaxPool2D(
                pool_size=3, strides=2, padding='valid', name='maxpool-b'),
            # Block C
            tf.keras.layers.BatchNormalization(name='bn-c1'),
            tf.keras.layers.Conv2D(
                filters=4 * _config['base_filters'], kernel_size=3,
                padding='same', activation='relu', name='conv-c1'),
            tf.keras.layers.BatchNormalization(name='bn-c2'),
            tf.keras.layers.Conv2D(
                filters=4 * _config['base_filters'], kernel_size=3,
                padding='same', activation='relu', name='conv-c2'),
            tf.keras.layers.BatchNormalization(name='bn-c3'),
            tf.keras.layers.Conv2D(
                filters=4 * _config['base_filters'], kernel_size=3,
                padding='same', activation='relu', name='conv-c3'),
            # Block D
            CustomFlatten(filters=128, name='flatten'),
            tf.keras.layers.Dropout(_config['dropout'], name='dropout-1'),
            tf.keras.layers.Dense(1024, activation='relu', name='fc-1'),
            tf.keras.layers.Dropout(_config['dropout'], name='dropout-2'),
            tf.keras.layers.Dense(1024, activation='relu', name='fc-2'),
            tf.keras.layers.Dense(num_classes, name='softmax')
        ])

    def calculate_loss(self, logits, labels):
        value = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
        return value

    def train(self, optimizer, dataset):
        total_loss = total_metric = 0
        examples_per_class = self._config['train_examples_per_class']
        batch_size = self._config['batch_size']
        total_batches = (len(common.WORD2LABEL) * examples_per_class) // batch_size
        pbar = tqdm(dataset, desc='train', total=total_batches, ncols=79)

        for batch_num, batch in enumerate(pbar, start=1):
            key_ids, image, labels = batch

            with tf.GradientTape() as tape:
                logits = self.model(image, training=True)
                loss = self.calculate_loss(logits, labels)
                total_loss += loss
                metric = self.calculate_map(logits, labels)
                total_metric += metric
            grad = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(
                zip(grad, self.model.trainable_variables),
                self._global_step)

            if batch_num == total_batches:
                break
        pbar.close()

        return (total_loss / total_batches,
                total_metric / total_batches)

    def validate(self, dataset):
        total_loss = total_metric = 0

        examples_per_class = self._config['val_examples_per_class']
        batch_size = self._config['batch_size']
        total_batches = (len(common.WORD2LABEL) * examples_per_class) // batch_size
        pbar = tqdm(dataset, desc='val', total=total_batches, ncols=79)
        for batch_num, batch in enumerate(pbar, start=1):
            key_ids, image, labels = batch
            logits = self.model(image, training=False)
            loss = self.calculate_loss(logits, labels)
            total_loss += loss
            metric = self.calculate_map(logits, labels)
            total_metric += metric
            if batch_num == total_batches:
                break
        pbar.close()

        return (total_loss / total_batches,
                total_metric / total_batches)

    def predict(self, dataset):
        key_ids, logits = [], []

        total_batches = int(np.ceil(112199 / self._config['batch_size']))
        pbar = tqdm(dataset, desc='test', total=total_batches, ncols=79)
        for batch_num, batch in enumerate(pbar, start=1):
            batch_key_ids, image = batch
            batch_logits = self.model(image, training=False)
            key_ids.append(batch_key_ids)
            logits.append(batch_logits)
        pbar.close()

        key_ids = tf.concat(key_ids, axis=0)
        logits = tf.concat(logits, axis=0)
        return key_ids, logits
