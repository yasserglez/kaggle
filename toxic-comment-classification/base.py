import os
import sys
import math
import pprint
import logging

import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchtext.data import Dataset, Field, Example
from torchtext.vocab import Vectors, pretrained_aliases

import common
import preprocessing


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

            preprocessed_data = preprocessing.load(self.params)
            self.fields, self.vocab = self.build_fields_and_vocab(preprocessed_data)
            self.train(preprocessed_data)
            self.predict(preprocessed_data)

    def build_fields_and_vocab(self, preprocessed_data):
        text_field = Field(pad_token='<PAD>', unk_token=None, batch_first=True, include_lengths=True)
        labels_field = Field(sequential=False, use_vocab=False, tensor_type=torch.FloatTensor)
        fields = [('text', text_field), ('labels', labels_field)]

        # Build the vocabulary
        train_df = common.load_data('submission', None, 'train.csv')
        train_df['text'] = train_df['id'].map(preprocessed_data)
        train_dataset = CommentsDataset(train_df, fields)
        test_df = common.load_data('submission', None, 'test.csv')
        test_df['text'] = test_df['id'].map(preprocessed_data)
        test_dataset = CommentsDataset(test_df, fields)
        text_field.build_vocab(train_dataset, test_dataset)
        vocab = text_field.vocab
        assert vocab.stoi['<PAD>'] == 0

        # Fill in missing words with the mean of the existing vectors
        vectors = pretrained_aliases[self.params['vectors']]()
        vectors_sum = np.zeros((vectors.dim, ))
        vectors_count = 0
        for token in vocab.itos:
            if token in vectors.stoi:
                vectors_sum += vectors[token].numpy()
                vectors_count += 1
        mean_vector = torch.FloatTensor(vectors_sum / vectors_count).unsqueeze(0)

        def getitem(self, token):
            return self.vectors[self.stoi[token]] if token in self.stoi else mean_vector
        Vectors.__getitem__ = getitem

        vocab.load_vectors(vectors)

        return fields, vocab

    def train(self, preprocessed_data):
        train_iter, val_iter = self.build_training_iterators(preprocessed_data)
        logger.info('Training on {:,} examples, validating on {:,} examples'
                    .format(len(train_iter.dataset), len(val_iter.dataset)))

        model = self.build_model()
        # Start with fixed embeddings
        model.embedding.weight.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        model_size = sum([np.prod(p.size()) for p in parameters])
        logger.info('Optimizing {:,} parameters:\n{}'.format(model_size, model))

        # SGD with warm restarts
        run = 0
        t_max = 1
        lr_max = self.params['lr_max']
        lr_min = self.params['lr_min']
        best_val_auc = 0
        while True:
            # New warm-start SGD run
            run += 1
            t_cur, lr = 0, lr_max
            optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
            logger.info('Starting run {} - t_max {}'.format(run, t_max))
            for epoch in range(t_max):
                loss_sum = 0
                model.train()
                t = tqdm(train_iter, ncols=79)
                for batch_num, batch in enumerate(t):
                    # Update the learning rate
                    t_cur = epoch + batch_num / len(train_iter)
                    lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * t_cur / t_max)) / 2
                    t.set_postfix(t_cur='{:.4f}'.format(t_cur), lr='{:.6f}'.format(lr))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    # Forward and backward pass
                    optimizer.zero_grad()
                    loss = self.calculate_loss(model, batch)
                    loss.backward()
                    self.update_parameters(model, optimizer, loss)
                    loss_sum += loss.data[0]
                loss = loss_sum / len(train_iter)
                logger.info('Run {} - t_cur {}/{} - lr {:.6f} - loss {:.6f}'
                            .format(run, int(math.ceil(t_cur)), t_max, lr, loss))

            # Run ended - evaluate early stopping
            model.eval()
            val_auc = self.evaluate_model(model, val_iter)
            if val_auc > best_val_auc:
                logger.info('Saving best model - val_auc {:.6f}'.format(val_auc))
                self.save_model(model)
                best_val_auc = val_auc
                # Double the number of epochs for the next run
                t_max = min(2 * t_max, 16)
            else:
                logger.info('Stopping - val_auc {:.6f}'.format(val_auc))
                break

        if not self.params['lr_min']:
            logger.info('Skipping fine-tuning')
            logger.info('Final model - best_val_auc {:.6f}'.format(best_val_auc))
            return

        # Fine-tuning for one epoch
        model = self.load_model()
        model.embedding.weight.requires_grad = True
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        model_size = sum([np.prod(p.size()) for p in parameters])
        logger.info('Fine-tuning {:,} parameters - best_val_auc {:.6f}'
                    .format(model_size, best_val_auc))

        lr_min = 0
        lr = lr_max = self.params['lr_min']
        t_cur, t_max = 0, 1
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
        model.train()
        t = tqdm(train_iter, ncols=79)
        for batch_num, batch in enumerate(t):
            # Update the learning rate
            t_cur = batch_num / len(train_iter)
            lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * t_cur / t_max)) / 2
            t.set_postfix(t_cur='{:.4f}'.format(t_cur), lr='{:.6f}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # Forward and backward pass
            optimizer.zero_grad()
            loss = self.calculate_loss(model, batch)
            loss.backward()
            self.update_parameters(model, optimizer, loss)

        # Evaluate the final model
        model.eval()
        val_auc = self.evaluate_model(model, val_iter)
        if val_auc > best_val_auc:
            logger.info('Saving best model - val_auc {:.6f}'.format(val_auc))
            self.save_model(model)

        logger.info('Final model - best_val_auc {:.6f}'.format(best_val_auc))

    def predic0t(self, preprocessed_data):
        pred_id, pred_iter = self.build_prediction_iterator(preprocessed_data)
        logger.info('Generating predictions for {:,} examples'.format(len(pred_iter.dataset)))
        model = self.load_model()
        output = self.predict_model(model, pred_iter)

        predictions = pd.DataFrame(output, columns=common.LABELS)
        predictions.insert(0, 'id', pred_id)
        predictions.to_csv(self.output_file, index=False)

    def build_training_iterators(self, preprocessed_data):
        raise NotImplementedError

    def build_prediction_iterator(self, preprocessed_data):
        raise NotImplementedError

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
            text, text_lengths = batch.text
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
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.data[vocab.stoi['<PAD>'], :] = 0


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
