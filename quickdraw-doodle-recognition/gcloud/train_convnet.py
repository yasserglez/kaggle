import os
import random

import tensorflow as tf
import numpy as np
from onecycle import OneCycleLR

import common
import convnet


def bin_generator(bin_file):
    with open(bin_file, 'rb') as f:
        yield from common.unpack_examples(f)


def get_train_generator(batch_size):
    input_dir = os.path.abspath('output/train_simplified')
    bin_files = ['{}/{}'.format(input_dir, bin_file)
                 for bin_file in os.listdir(input_dir)
                 if bin_file.endswith('.bin')]
    while True:
        random.shuffle(bin_files)
        generators = list(map(bin_generator, bin_files))
        images, labels = [], []
        for example in common.roundrobin(generators):
            images.append(example['image'])
            labels.append(example['label'])
            if len(images) == batch_size:
                images = np.stack(images)
                labels = np.array(labels, dtype=np.int32)
                yield images, labels
                images, labels = [], []


def main():
    random.seed(common.RANDOM_SEED)
    np.random.seed(common.RANDOM_SEED)
    tf.set_random_seed(common.RANDOM_SEED)

    num_epochs = 20
    num_samples = 49707579
    steps_per_epoch = num_samples // common.BATCH_SIZE
    generator = get_train_generator(common.BATCH_SIZE)

    model = convnet.get_model()
    optimizer = tf.keras.optimizers.SGD(nesterov=True)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer)
    model.summary()

    lr_scheduler = OneCycleLR(
        num_samples, num_epochs, common.BATCH_SIZE, max_lr=1e-2,
        maximum_momentum=0.95, minimum_momentum=0.85)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'output/convnet/epoch-{epoch}.h5',
        monitor='loss', save_weights_only=True)

    model.fit_generator(
        generator, max_queue_size=8,
        epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        callbacks=[lr_scheduler, checkpoint], verbose=2)
    model.save_weights('output/convnet/final.h5')


if __name__ == '__main__':
    main()
