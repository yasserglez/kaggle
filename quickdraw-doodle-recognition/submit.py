import os
import subprocess
import argparse


def submit(experiment_name):
    predictions_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     'predictions',
                     f'{experiment_name}.csv'))

    subprocess.call([
        'kaggle', 'competitions', 'submit',
        '-c', 'quickdraw-doodle-recognition',
        '-f', predictions_file,
        '-m', experiment_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    args = parser.parse_args()
    submit(args.experiment_name)
