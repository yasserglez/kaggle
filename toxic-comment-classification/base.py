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
from tensorboardX import SummaryWriter

import torch
from torch import nn, optim, autograd
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

    def __init__(self, name, params, random_seed, debug=False):
        self.name = name
        self.params = params
        self.random_seed = random_seed
        self.debug = debug

        self.output_dir = os.path.join(
            common.OUTPUT_DIR,
            self.name,
            str(self.random_seed),
            common.params_str(self.params))
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if self.debug:
            self.writer = SummaryWriter(self.output_dir)

        self.model_output = os.path.join(self.output_dir, 'model.pickle')
        self.train_output = os.path.join(self.output_dir, 'train.csv')
        self.validation_output = os.path.join(self.output_dir, 'validation.csv')
        self.test_output = os.path.join(self.output_dir, 'test.csv')

    def main(self):
        logger.info(' {} / {} '.format(self.name, self.random_seed).center(62, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(self.test_output):
            logger.info('Output already exists - skipping')
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
        datasets = []
        for dataset in ['train', 'validation', 'test']:
            df = common.load_data(self.random_seed, dataset)
            df['text'] = df['id'].map(preprocessed_data)
            datasets.append(CommentsDataset(df, fields))
        text_field.build_vocab(*datasets)
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
        train_iter = self.build_train_iterator(preprocessed_data)
        _, val_iter = self.build_prediction_iterator(preprocessed_data, 'validation')
        logger.info('Training on {:,} examples, validating on {:,} examples'
                    .format(len(train_iter.dataset), len(val_iter.dataset)))

        # Train the model keeping the word embeddings fixed until the validation AUC
        # stops improving, then unfreeze the embeddings and fine-tune the entire
        # model with a lower learning rate. Use SGD with warm restarts.
        model = self.build_model()
        model.embedding.weight.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        model_size = sum([np.prod(p.size()) for p in parameters])
        logger.info('Optimizing {:,} parameters:\n{}'.format(model_size, model))
        run = epoch = 0
        lr_max = self.params['lr_high']
        optimizer = optim.SGD(parameters, lr=lr_max, momentum=0.9)
        t_max = 1
        best_val_auc = 0

        while True:
            run += 1
            # grad_norms = []
            t_cur, lr = 0, lr_max
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            logger.info('Starting run {} - t_max {}'.format(run, t_max))
            for t_index in range(t_max):
                epoch += 1
                loss_sum = 0
                model.train()
                t = tqdm(train_iter, ncols=79)
                for batch_index, batch in enumerate(t):
                    # Update the learning rate
                    t_cur = t_index + batch_index / len(train_iter)
                    lr = lr_max * (1 + math.cos(math.pi * t_cur / t_max)) / 2
                    t.set_postfix(t_cur='{:.4f}'.format(t_cur), lr='{:.6f}'.format(lr))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    # Forward and backward pass
                    optimizer.zero_grad()
                    loss = self.calculate_loss(model, batch)
                    loss.backward()
                    # grad_vector = [p.grad.data.view(-1) for p in parameters]
                    # grad_norms.append(torch.cat(grad_vector).norm())
                    self.update_parameters(model, optimizer, loss)
                    loss_sum += loss.data[0]
                loss = loss_sum / len(train_iter)
                logger.info('Epoch {} - run {} - t_cur {}/{} - lr {:.6f} - loss {:.6f}'
                            .format(epoch, run, int(math.ceil(t_cur)), t_max, lr, loss))

                # https://arxiv.org/abs/1212.0901
                # logger.info('Average norm of the gradient - {:.6f}'.format(np.mean(grad_norms)))
                if self.debug:
                    auc = self.evaluate_model(model, train_iter)
                    val_auc = self.evaluate_model(model, val_iter)
                    self.writer.add_scalars('auc', {'train': auc, 'validation': val_auc}, epoch)

            # Run ended - evaluate early stopping
            val_auc = self.evaluate_model(model, val_iter)
            if round(val_auc - best_val_auc, 4) > 0:  # TODO: consider increasing precision
                logger.info('Saving best model - val_auc {:.6f}'.format(val_auc))
                self.save_model(model)
                best_val_auc = val_auc
                # Double the number of epochs for the next run
                t_max = min(2 * t_max, 16)
            else:
                logger.info('Stopping - val_auc {:.6f}'.format(val_auc))
                if self.params['lr_low'] == 0 or model.embedding.weight.requires_grad:
                    # Fine-tuning not needed or it just finished
                    break
                else:
                    model = self.load_model()
                    model.embedding.weight.requires_grad = True
                    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
                    model_size = sum([np.prod(p.size()) for p in parameters])
                    logger.info('Fine-tuning {:,} parameters - best_val_auc {:.6f}'
                                .format(model_size, best_val_auc))
                    run = 0
                    lr_max = self.params['lr_low']
                    optimizer = optim.SGD(parameters, lr=lr_max, momentum=0.9)
                    t_max = 1

        logger.info('Final model - best_val_auc {:.6f}'.format(best_val_auc))

    def predict(self, preprocessed_data):
        model = self.load_model()
        for dataset in ['train', 'validation', 'test']:
            csv_file = (self.train_output if dataset == 'train' else
                        self.validation_output if dataset == 'validation' else
                        self.test_output)
            logger.info('Generating {} predictions'.format(dataset))
            pred_id, pred_iter = self.build_prediction_iterator(preprocessed_data, dataset)
            output = self.predict_model(model, pred_iter)
            predictions = pd.DataFrame(output, columns=common.LABELS)
            predictions.insert(0, 'id', pred_id)
            predictions.to_csv(csv_file, index=False)

    def build_train_iterator(self, preprocessed_data):
        raise NotImplementedError

    def build_prediction_iterator(self, preprocessed_data, dataset):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

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
        torch.save(model.state_dict(), self.model_output)

    def load_model(self):
        model = self.build_model()
        model.load_state_dict(torch.load(self.model_output))
        return model


class BaseModule(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        vocab_size, embed_size = vocab.vectors.shape
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
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
                 hidden_layers=0, hidden_nonlinearity=None,
                 input_dropout=0, hidden_dropout=0, dropout=0):

        super().__init__()

        if isinstance(hidden_layers, list):
            # The number of hidden units for each layer is explicitly given
            units = [input_size] + hidden_layers + [output_size]
        else:
            # Increase/decrease the number of units linearly from input to output
            units = np.linspace(input_size, output_size, hidden_layers + 2)
            units = list(map(int, np.round(units, 0)))

        if dropout:
            # Use the same dropout for the input and hidden layers
            input_dropout = hidden_dropout = dropout

        layers = []
        for layer_index, (in_size, out_size) in enumerate(zip(units, units[1:])):
            if layer_index == 0:
                if input_dropout:
                    layers.append(nn.Dropout(input_dropout))
            else:
                if hidden_dropout:
                    layers.append(nn.Dropout(hidden_dropout))
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

        # Weight initialization
        for layer in layers:
            if isinstance(layer, nn.Linear):
                gain = nn.init.calculate_gain(hidden_nonlinearity) if hidden_nonlinearity else 1.0
                nn.init.xavier_uniform(layer.weight, gain=gain)
                nn.init.constant(layer.bias, 0.0)
        if output_nonlinearity and output_nonlinearity != hidden_nonlinearity:
            gain = nn.init.calculate_gain(output_nonlinearity)
            nn.init.xavier_uniform(layers[-2].weight, gain=gain)

    def forward(self, x):
        return self.dense(x)
