import os
import sys
import pprint
import logging
from datetime import datetime, timedelta

import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchtext.data import Dataset, Field, Example

import common
import preprocessing
import embeddings


logger = logging.getLogger(__name__)


class CommentsDataset(Dataset):

    def __init__(self, df, fields, **kwargs):
        if common.LABELS[0] in df.columns:
            labels = df[common.LABELS].values.astype(np.float)
        else:
            labels = np.full(df.shape[0], np.nan)
        examples = []
        for values in zip(df['text'], labels):
            example = Example.fromlist(values, fields)
            examples.append(example)
        super().__init__(examples, fields, **kwargs)


class BaseModel(object):
    """Base class of all the implemented models.

    Parameters:
        lr: learning rate.
        max_epochs: maximum number of epochs.
        patience: number of epochs with no improvement after which training will be stopped.
    """

    def __init__(self, mode, name, params, random_seed):
        self.mode = mode
        self.name = name
        self.params = params
        self.random_seed = random_seed
        self.model_dir = os.path.join(common.DATA_DIR, self.mode, str(self.random_seed), self.name)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.output_file = os.path.join(self.model_dir, common.params_str(self.params) + '.csv')

    def main(self):
        logger.info(' {} / {} / {} '.format(self.mode, self.name, self.random_seed).center(60, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(self.output_file):
            logger.info('Output file already exists - skipping')
        else:
            self.random_state = RandomState(self.random_seed)
            np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))
            torch.manual_seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))

            text_field = Field(batch_first=True, include_lengths=True, pad_token='<PAD>', unk_token='<UNK>')
            labels_field = Field(sequential=False, use_vocab=False, tensor_type=torch.FloatTensor)
            self.fields = [('text', text_field), ('labels', labels_field)]

            preprocessed_data = preprocessing.load(self.params)

            vectors = []
            if self.params['vectors_glove']:
                vectors.append('glove.42B.300d')
            if self.params['vectors_fasttext']:
                fasttext_params = {
                    'embedding_size': self.params['vectors_fasttext'],
                    'pretrain_model': 'cbow',
                    'pretrain_lr': 0.05,
                    'pretrain_epochs': 50,
                }
                vectors.append(embeddings.load(preprocessed_data, fasttext_params))
            if self.params['vectors_pmi']:
                pmi_params = {
                    'embedding_size': self.params['vectors_pmi'],
                    'pretrain_model': 'pmi',
                    'pretrain_lr': 0.001,
                    'pretrain_epochs': 50,
                }
                vectors.append(embeddings.load(preprocessed_data, pmi_params))
            if self.params['vectors_lm']:
                lm_params = {
                    'embedding_size': self.params['vectors_lm'],
                    'pretrain_model': 'lm',
                    'pretrain_lr': 0.001,
                    'pretrain_epochs': 50,
                }
                vectors.append(embeddings.load(preprocessed_data, lm_params))

            train_df = common.load_data('submission', None, 'train.csv')
            train_df['text'] = train_df['id'].map(preprocessed_data)
            train_dataset = CommentsDataset(train_df, self.fields)
            test_df = common.load_data('submission', None, 'test.csv')
            test_df['text'] = test_df['id'].map(preprocessed_data)
            test_dataset = CommentsDataset(test_df, self.fields)
            text_field.build_vocab(train_dataset, test_dataset, vectors=vectors)
            self.vocab = text_field.vocab

            self.train(preprocessed_data)
            self.predict(preprocessed_data)

    def train(self, preprocessed_data):
        train_iter, val_iter = self.build_training_iterators(preprocessed_data)
        logger.info('Training on {:,} examples, validating on {:,} examples'
                    .format(len(train_iter.dataset), len(val_iter.dataset)))

        model = self.build_model().cuda()
        trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_size = sum([np.prod(p.size()) for p in trainable_parameters])
        logger.info('The model has {:,} parameters:\n{}'.format(model_size, model))

        optimizer = self.build_optimizer(model)
        patience_count = 0
        best_val_auc = 0

        for epoch in range(1, self.params['max_epochs'] + 1):
            t_start = datetime.now()
            loss = self.train_model(model, optimizer, train_iter)
            val_auc = self.evaluate_model(model, val_iter)
            t_end = datetime.now()

            logger.info('Epoch {:04d} - loss: {:.6g}, val_auc: {:.6g}, time: {}'
                        .format(epoch, loss, val_auc, str(t_end - t_start).split('.')[0]))

            if val_auc > best_val_auc:
                logger.info('Saving best model - val_auc: {:.6g}'.format(val_auc))
                self.save_model(model)
                best_val_auc = val_auc
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.params['patience']:
                    logger.info('Stopped - best_val_auc: {:.6g}'.format(best_val_auc))
                    break

            if epoch == self.params['max_epochs']:
                logger.warning('Training reached the maximum number of epochs')

    def predict(self, preprocessed_data):
        pred_id, pred_iter = self.build_prediction_iterator(preprocessed_data)
        logger.info('Generating predictions for {:,} examples'.format(len(pred_iter.dataset)))
        model = self.load_model().cuda()
        output = self.predict_model(model, pred_iter)

        predictions = pd.DataFrame(output, columns=common.LABELS)
        predictions.insert(0, 'id', pred_id)
        predictions.to_csv(self.output_file, index=False)

    def build_training_iterators(self, preprocessed_data):
        raise NotImplementedError

    def build_prediction_iterator(self, preprocessed_data):
        raise NotImplementedError

    def build_optimizer(self, model):
        trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_parameters, lr=self.params['lr'])
        return optimizer

    def build_model(self):
        raise NotImplementedError

    def train_model(self, model, optimizer, train_iter):
        loss_sum = 0
        model.train()
        for batch in train_iter:
            optimizer.zero_grad()
            loss = self.calculate_loss(model, batch)
            loss.backward()
            self.update_parameters(model, optimizer, loss)
            loss_sum += loss.data[0]
        return loss_sum / len(train_iter)

    def calculate_loss(self, model, batch):
        (text, text_lengths), labels = batch.text, batch.labels
        output = model(text, text_lengths)
        loss = F.binary_cross_entropy(output, labels)
        return loss

    def update_parameters(self, model, optimizer, loss):
        optimizer.step()

    def evaluate_model(self, model, batch_iter):
        model.eval()
        labels, predictions = [], []
        for batch in batch_iter:
            (text, text_lengths), _ = batch.text, batch.labels
            labels.append(batch.labels.data.cpu())
            output = model(text, text_lengths)
            predictions.append(output.data.cpu())
        labels = torch.cat(labels).numpy()
        predictions = torch.cat(predictions).numpy()
        auc = roc_auc_score(labels, predictions, average='macro')
        return auc

    def predict_model(self, model, batch_iter):
        model.eval()
        predictions = []
        for batch in batch_iter:
            (text, text_lengths), _ = batch.text, batch.labels
            output = model(text, text_lengths)
            predictions.append(output.data.cpu())
        predictions = torch.cat(predictions).numpy()
        return predictions

    def save_model(self, model):
        file_path = os.path.join(self.model_dir, common.params_str(self.params) + '.pickle')
        torch.save(model.state_dict(), file_path)

    def load_model(self):
        model = self.build_model()
        file_path = os.path.join(self.model_dir, common.params_str(self.params) + '.pickle')
        model.load_state_dict(torch.load(file_path))
        return model


class BaseModule(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        vocab_size, embedding_size = vocab.vectors.shape
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(vocab.vectors)
        for word in ['<PAD>', '<UNK>']:
            self.embedding.weight.data[vocab.stoi[word], :] = 0
        self.embedding.weight.requires_grad = False


class Dense(nn.Module):

    nonlinearities = {
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
    }

    def __init__(self, input_size, output_size, output_nonlinearity=None,
                 hidden_layers=0, hidden_nonlinearity=None, dropout=0):

        super().__init__()

        # Increase/decrease the number of units linearly from input to output
        units = np.linspace(input_size, output_size, hidden_layers + 2)
        units = list(map(int, np.round(units, 0)))

        layers = []
        for in_size, out_size in zip(units, units[1:]):
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_size, out_size))
            if hidden_nonlinearity:
                layers.append(self.nonlinearities[hidden_nonlinearity])
        # Remove the last hidden nonlinearity (if any)
        if hidden_nonlinearity:
            layers.pop()
        # and add the output nonlinearity (if any)
        if output_nonlinearity:
            layers.append(self.nonlinearities[output_nonlinearity])

        self.dense = nn.Sequential(*layers)

        for layer in layers:
            if isinstance(layer, nn.Linear):
                gain = nn.init.calculate_gain(hidden_nonlinearity)
                nn.init.xavier_uniform(layer.weight, gain=gain)
                nn.init.constant(layer.bias, 0.0)

    def forward(self, x):
        return self.dense(x)
