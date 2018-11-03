import tensorflow as tf

import common


def get_model():
    input_shape = (common.IMAGE_SIZE, common.IMAGE_SIZE, 1)
    output_shape = len(common.WORD2LABEL)
    model = tf.keras.Sequential([
        # Block A
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=7, strides=3, padding='same',
            activation='relu', name='conv-a', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding='valid', name='maxpool-a'),
        # Block B
        tf.keras.layers.BatchNormalization(name='bn-b'),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=5, padding='same',
            activation='relu', name='conv-b'),
        tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding='valid', name='maxpool-b'),
        # Block C
        tf.keras.layers.BatchNormalization(name='bn-c1'),
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding='same',
            activation='relu', name='conv-c1'),
        tf.keras.layers.BatchNormalization(name='bn-c2'),
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding='same',
            activation='relu', name='conv-c2'),
        tf.keras.layers.BatchNormalization(name='bn-c3'),
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding='same',
            activation='relu', name='conv-c3'),
        # Block D
        CustomFlatten(filters=128, name='flatten'),
        tf.keras.layers.Dropout(0.1, name='dropout-1'),
        tf.keras.layers.Dense(1024, activation='relu', name='fc-1'),
        tf.keras.layers.Dropout(0.1, name='dropout-2'),
        tf.keras.layers.Dense(1024, activation='relu', name='fc-2'),
        tf.keras.layers.Dense(output_shape, activation='softmax', name='softmax')
    ])
    return model


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
