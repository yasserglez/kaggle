import os

import pandas as pd


INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')


def load_train_data():
    df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    images = df.iloc[:, 1:].values.reshape(-1, 28, 28) / 255
    targets = df.iloc[:, 0].values
    return images, targets


def load_test_data():
    df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    images = df.values.reshape(-1, 28, 28) / 255
    return images


def save_predictions(labels, output_file):
    df = pd.DataFrame({'ImageId': range(1, labels.shape[0] + 1), 'Label': labels})
    df.to_csv(output_file, index=False)
