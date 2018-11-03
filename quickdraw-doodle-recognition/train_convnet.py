import os
import csv
import uuid
import pprint
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

import common
from convnet_data import load_train_val_datasets, load_test_dataset
from models import get_model_wrapper


EXPERIMENT = Experiment()
EXPERIMENT.captured_out_filter = apply_backspaces_and_linefeeds
EXPERIMENT.observers.append(MongoObserver.create())


@EXPERIMENT.config
def config():
    seed = 7198021
    train_examples_per_class = 112640
    val_examples_per_class = 768
    batch_size = 1024
    train_augmentation = True
    test_augmentation = 10
    model = 'convnet'
    image_size = 128
    base_filters = 64
    learning_rate = 0.01
    momentum = 0.9
    dropout = 0.1
    # Number of epochs in initial cycle
    cycle_len = 1
    # Each restart runs for cycle_mult times for epochs
    cycle_mult = 2
    # Maximum number of restarts to perform
    max_cycles = 5


@EXPERIMENT.main
def main(_run, _config, _log):
    experiment_name = _run.experiment_info['name']
    _log.info(f'Starting run {experiment_name}')
    pprint.pprint(_config)

    # Initialize the random number generators
    random.seed(_config['seed'])
    np.random.seed(_config['seed'])
    tf.set_random_seed(_config['seed'])

    # Load training and validation datasets
    train_dataset, val_dataset = load_train_val_datasets(
        _config['image_size'], _config['batch_size'],
        _config['train_augmentation'], _config['seed'])

    # Initialize the model and model checkpointing
    model_wrapper = get_model_wrapper(_config)
    model_wrapper.model.summary()
    checkpoint = tf.train.Checkpoint(model=model_wrapper.model)
    checkpoint_dir = os.path.join(common.MODEL_DIR, experiment_name)
    checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1)

    # Initialize the optimizer
    # https://arxiv.org/abs/1608.03983
    batch_size = _config['batch_size']
    examples_per_class = _config['train_examples_per_class']
    batches_per_epoch = (len(common.WORD2LABEL) * examples_per_class) // batch_size
    global_step = tf.train.get_or_create_global_step()
    global_step.assign(0)
    learning_rate = tf.train.cosine_decay_restarts(
        _config['learning_rate'], global_step,
        first_decay_steps=_config['cycle_len'] * batches_per_epoch,
        t_mul=_config['cycle_mult'])
    # https://stackoverflow.com/a/50778921
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, _config['momentum'], use_nesterov=True)

    epoch_num = 0
    cycle_count = 0
    next_cycle_epoch = _config['cycle_len']
    best_val_metric = 0

    while True:
        epoch_num += 1
        _log.info(f'Starting epoch {epoch_num}')

        train_loss, train_metric = model_wrapper.train(optimizer, train_dataset)
        _log.info('train_loss: {:.6f} train_MAP@3: {:.6f}'
                  .format(train_loss, train_metric))
        _run.log_scalar('train_loss', train_loss.numpy(), epoch_num)
        _run.log_scalar('train_MAP@3', train_metric, epoch_num)

        if epoch_num == next_cycle_epoch:
            cycle_count += 1
            _log.info('Finished cycle {}/{}'.format(cycle_count, _config['max_cycles']))

            # Evaluation on the validation dataset
            val_loss, val_metric = model_wrapper.validate(val_dataset)
            _log.info('val_loss: {:.6f} val_MAP@3: {:.6f}'
                      .format(val_loss, val_metric))
            _run.log_scalar('val_loss', val_loss.numpy(), epoch_num)
            _run.log_scalar('val_MAP@3', val_metric, epoch_num)

            if val_metric <= best_val_metric:
                # Early stopping
                _log.warning('val_MAP@3 did not improve {:.6f} <= {:.6f}'
                             .format(val_metric, best_val_metric))
                break
            else:
                _log.info('Model checkpoint. val_MAP@3: {:.6f} > {:.6f}'
                          .format(val_metric, best_val_metric))
                checkpoint_manager.save()
                best_val_metric = val_metric
                if cycle_count < _config['max_cycles']:
                    # Update the target epoch for the new cycle
                    next_cycle_epoch += _config['cycle_mult'] ** cycle_count
                else:
                    # Reached the maximum number of cycles
                    break

    # Generate predictions on the test dataset using the latest checkpoint.
    checkpoint_dir = checkpoint_manager.latest_checkpoint
    status = checkpoint.restore(checkpoint_dir)
    status.assert_consumed()
    test_key_ids, test_scores = generate_predictions(_config, _log, model_wrapper)

    # Generating the submission file.
    test_labels = tf.nn.top_k(test_scores, k=3)[1].numpy().tolist()
    predictions_file = os.path.join(common.PREDICTIONS_DIR, f'{experiment_name}.csv')
    with open(predictions_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, ['key_id', 'word'])
        writer.writeheader()
        for key_id, labels in zip(test_key_ids, test_labels):
            words = [common.LABEL2WORD[i].replace(' ', '_') for i in labels]
            row = {'key_id': key_id.numpy().decode(), 'word': ' '.join(words)}
            writer.writerow(row)
    _run.add_artifact(predictions_file, 'predictions')


def generate_predictions(_config, _log, model_wrapper):
    test_key_ids, test_scores = None, []
    _log.info('Generating predictions')

    # Predictions using the original images.
    test_dataset = load_test_dataset(
        _config['image_size'], _config['batch_size'],
        augmentation=False)
    test_key_ids, scores = model_wrapper.predict(test_dataset)
    test_scores.append(scores)

    # Test time augmentation (TTA).
    for i in range(_config['test_augmentation']):
        _log.info('Test time augmentation [{}/{}]'
                  .format(i + 1, _config['test_augmentation']))
        test_dataset = load_test_dataset(
            _config['image_size'], _config['batch_size'],
            augmentation=True)
        scores = model_wrapper.predict(test_dataset)[1]
        test_scores.append(scores)

    # Averaging the scores.
    final_scores = test_scores[0]
    if _config['test_augmentation'] > 0:
        for scores in test_scores[1:]:
            final_scores += scores
        final_scores = final_scores / (1 + _config['test_augmentation'])

    return test_key_ids, final_scores


if __name__ == '__main__':
    config_updates = {}
    EXPERIMENT.run(
        config_updates=config_updates,
        # https://github.com/IDSIA/sacred/issues/110#issuecomment-293337397
        options={'--name': str(uuid.uuid1())})
