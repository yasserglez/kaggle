import os
import csv
import random

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import pandas as pd

import common
import drawing
import convnet


def get_test_generator(batch_size, augmentation=False):
    csv_file = os.path.abspath('input/test_simplified.csv')
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        images = []
        for example in reader:
            image = drawing.process_drawing(
                example['drawing'], common.IMAGE_SIZE, augmentation)
            images.append(image)
            if len(images) == batch_size:
                images = np.stack(images)
                yield images
                images = []
        if len(images) > 0:
            images = np.stack(images)
            yield images


def generate_predictions(model, augmented_images):
    steps = int(np.ceil(112199 / common.BATCH_SIZE))
    # Predictions on the original images.
    generator = get_test_generator(common.BATCH_SIZE, augmentation=False)
    predictions = model.predict_generator(generator, steps)
    # Test time augmentation (TTA).
    for i in range(augmented_images):
        generator = get_test_generator(common.BATCH_SIZE, augmentation=True)
        augmented_predictions = model.predict_generator(generator, steps)
        predictions += augmented_predictions
    if augmented_images > 0:
        predictions /= (1 + augmented_images)
    return predictions


def write_submission(predictions, output_file):
    key_ids = pd.read_csv('input/test_simplified.csv',
                          usecols=['key_id'], dtype=str).iloc[:, 0]
    labels = tf.nn.top_k(predictions, k=3)[1].numpy().tolist()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, ['key_id', 'word'])
        writer.writeheader()
        for key_id, labels in zip(key_ids, labels):
            words = [common.LABEL2WORD[i].replace(' ', '_') for i in labels]
            row = {'key_id': key_id, 'word': ' '.join(words)}
            writer.writerow(row)


def main():
    random.seed(common.RANDOM_SEED)
    np.random.seed(common.RANDOM_SEED)
    tf.set_random_seed(common.RANDOM_SEED)

    model = convnet.get_model()
    model.load_weights('output/convnet/final.h5')
    predictions = generate_predictions(model, augmented_images=20)
    write_submission(predictions, 'submission.csv')


if __name__ == '__main__':
    main()
