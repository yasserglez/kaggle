import os
import sys
import pprint
from datetime import datetime
from collections import Counter

import joblib
import spacy
import pandas as pd
import numpy as np
from numpy.random import RandomState
from scipy.sparse import csr_matrix
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchtext.data import Dataset, Field, Example


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
VOCAB_DIR = os.path.join(DATA_DIR, 'vocab')
PMI_DIR = os.path.join(DATA_DIR, 'pmi')

for d in [VOCAB_DIR, PMI_DIR]:
    if not os.path.isdir(d):
        os.makedirs(d)

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
NUM_TOKEN = '<NUM>'

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, NUM_TOKEN]

TARGET_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_data(mode, random_seed, file_name):
    path_parts = [DATA_DIR, mode]
    if mode == 'cross_validation':
        path_parts.append(str(random_seed))
    path_parts.append(file_name)

    df = pd.read_csv(os.path.join(*path_parts))
    df['comment_text'].fillna('', inplace=True)

    # Manually cleanup a problematic example
    if file_name == 'test.csv':
        df.loc[df['id'] == 206058417140, 'comment_text'] = \
            df.loc[df['id'] == 206058417140, 'comment_text'].str.rstrip('!') + '!'

    return df


def split_data(df, test_size, random_state):
    test_df = df.groupby(TARGET_COLUMNS) \
        .apply(lambda x: x.sample(frac=test_size, random_state=random_state))
    train_df = df[~df['id'].isin(test_df['id'])]
    return train_df, test_df


def format_params(params):
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    s = '_'.join(['{}={}'.format(k.replace('_', '-'), v) for k, v in sorted_params])
    return s


class TextField(Field):

    def __init__(self, params, **kwargs):
        self.params = params
        if self.params['token'] == 'word':
            self._english = spacy.load('en')

        super_kwargs = dict(
            batch_first=True,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN)
        super_kwargs.update(kwargs)
        super().__init__(**super_kwargs)

    def preprocess(self, x):
        x = self._preprocessing(x)
        x = self._tokenize(x)
        x = (self._char_postprocessing(x)
             if self.params['token'] == 'char' else
             self._word_postprocessing(x))
        return x

    def _preprocessing(self, text):
        if self.params['lower']:
            text = text.lower()
        return text

    def _tokenize(self, x):
        if self.params['token'] == 'char':
            tokens = list(x)
        else:
            tokens = [token.text for token in self._english.tokenizer(x)]
        return tokens

    def _char_postprocessing(self, tokens):
        final_tokens = tokens[:self.params['max_len']]
        if not final_tokens:
            final_tokens.append(UNK_TOKEN)
        return final_tokens

    def _word_postprocessing(self, tokens):
        final_tokens = []
        for token in tokens:
            if token.isspace():
                continue
            elif token.isnumeric():
                final_tokens.append(NUM_TOKEN)
            else:
                final_tokens.append(token)
        final_tokens = tokens[:self.params['max_len']]
        if not final_tokens:
            final_tokens.append(UNK_TOKEN)
        return final_tokens


class ToxicCommentDataset(Dataset):

    def __init__(self, df, text_field, **kwargs):
        target_field = Field(sequential=False, use_vocab=False, tensor_type=torch.FloatTensor)
        fields = [('text', text_field), ('target', target_field)]
        if TARGET_COLUMNS[0] in df.columns:
            targets = df[TARGET_COLUMNS].values.astype(np.float)
        else:
            targets = np.zeros(df.shape[0], dtype=np.float)
        examples = []
        for values in zip(df['comment_text'], targets):
            example = Example.fromlist(values, fields)
            examples.append(example)
        super().__init__(examples, fields, **kwargs)


def build_vocab(text_field):
    vocab_name = format_params({k: text_field.params[k] for k in ['token', 'lower', 'min_freq']})
    vocab_file = os.path.join(VOCAB_DIR, f'{vocab_name}.pickle')
    if not os.path.exists(vocab_file):
        print(f'Building the vocabulary: {vocab_name}')

        train_df = load_data('submission', None, 'train.csv')
        train_dataset = ToxicCommentDataset(train_df, text_field)
        test_df = load_data('submission', None, 'test.csv')
        test_dataset = ToxicCommentDataset(test_df, text_field)

        text_field.build_vocab(
            train_dataset, test_dataset,
            min_freq=text_field.params['min_freq'])

        joblib.dump(text_field.vocab, vocab_file)

    text_field.vocab = joblib.load(vocab_file)
    return text_field.vocab


# Based on https://github.com/dawenl/cofactor
def build_pmi(text_field, vocab, k=1):
    pmi_name = format_params({k: text_field.params[k] for k in ['token', 'lower', 'min_freq']})
    pmi_file = os.path.join(PMI_DIR, f'{pmi_name}.pickle')
    if not os.path.exists(pmi_file):
        print(f'Building the PMI matrix: {pmi_name}')

        df = load_data('submission', None, 'train.csv')
        dataset = ToxicCommentDataset(df, text_field)
        ignored_tokens = {vocab.stoi[token] for token in SPECIAL_TOKENS}

        # Co-occurrences (i: token, j: label)
        ij_freq = Counter()
        for example in dataset:
            j_values = list(np.flatnonzero(example.target))
            if not j_values:
                continue
            for token in set(example.text):
                i = vocab.stoi[token]
                if i not in ignored_tokens:
                    for j in j_values:
                        ij_freq[(i, j)] += 1

        # Marginals (i: token, j: label)
        i_freq, j_freq = Counter(), Counter()
        for (i, j), count in ij_freq.items():
            i_freq[i] += count
            j_freq[j] += count

        D = sum(ij_freq.values())
        data, row_ind, col_ind = [], [], []
        for i, j in ij_freq.keys():
            value = np.log(D * ij_freq[(i, j)] / (i_freq[i] * j_freq[j]))
            row_ind.append(i)
            col_ind.append(j)
            data.append(value)

        pmi = csr_matrix((data, (row_ind, col_ind)), shape=(len(vocab), 6))
        pmi.data[pmi.data < 0] = 0
        pmi.eliminate_zeros()

        joblib.dump(pmi, pmi_file)

    pmi = joblib.load(pmi_file)
    if k > 1:
        # Apply the parameter controlling the sparsity
        offset = np.log(k)
        pmi.data -= offset
        pmi.data[pmi.data < 0] = 0
        pmi.eliminate_zeros()
    return pmi


class BaseModel(object):

    def __init__(self, mode, params, random_seed, name=None, tag=None):
        self.random_seed = random_seed
        self.random_state = RandomState(random_seed)
        np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))
        torch.manual_seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))

        self.mode = mode
        self.params = params
        self.name = name if name else self.__class__.__name__.lower()
        self.tag = tag if tag else format_params(params)
        print(' {} / {} / {} '.format(self.name, self.mode, self.random_seed).center(80, '='))
        print('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))

        self.model_dir = os.path.join(DATA_DIR, self.mode, str(self.random_seed), self.name)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.output_file = os.path.join(self.model_dir, format_params(self.params) + '.csv')

        self.text_field = TextField(params, include_lengths=True)
        self.vocab = build_vocab(self.text_field)
        print('Considering {:,} tokens'.format(len(self.vocab)))
        if self.params['reg_lambda'] > 0:
            self.pmi = build_pmi(self.text_field, self.vocab, self.params['reg_k'])
            self.reg_lambda = Variable(torch.FloatTensor([self.params['reg_lambda']]).cuda())

    def main(self):
        if not os.path.isfile(self.output_file):
            self.train()
            self.predict()
        else:
            print('Output file already exists - skipping')

    def train(self):
        train_iter, val_iter = self.build_training_iterators()
        print('Training on {:,} examples, validating on {:,} examples'
              .format(len(train_iter.dataset), len(val_iter.dataset)))

        model = self.build_model().cuda()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_size = sum([np.prod(p.size()) for p in model_parameters])
        print('The model has {:,} parameters:\n{}'.format(model_size, model))

        optimizer = self.build_optimizer(model)
        best_val_bce = np.inf
        patience_count = 0

        for epoch in range(1, self.params['max_epochs'] + 1):
            t_start = datetime.now()
            bce, loss = self.train_model(model, optimizer, train_iter)
            val_bce = self.evaluate_model(model, val_iter)
            t_end = datetime.now()

            print('Epoch {:04d}/{:04d} - bce: {:.6g}, loss: {:.6g}, val_bce: {:.6g}, time: {}'
                  .format(epoch, self.params['max_epochs'], bce, loss, val_bce,
                          str(t_end - t_start).split('.')[0]))

            if val_bce < best_val_bce:
                print('Saving best model - val_bce: {:.6g}'.format(val_bce))
                self.save_model(model)
                best_val_bce = val_bce
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.params['patience']:
                    print('Stopped - best_val_bce: {:.6g}'.format(best_val_bce))
                    break

    def predict(self):
        pred_id, pred_iter = self.build_prediction_iterator()
        print('Generating predictions for {:,} examples'.format(len(pred_iter.dataset)))
        model = self.load_model().cuda()
        output = self.predict_model(model, pred_iter)

        predictions = pd.DataFrame(output, columns=TARGET_COLUMNS)
        predictions.insert(0, 'id', pred_id)
        predictions.to_csv(self.output_file, index=False)

    def train_model(self, model, optimizer, train_iter):

        bce_sum = loss_sum = 0
        model.train()
        for batch in train_iter:
            optimizer.zero_grad()
            loss = bce = self.calculate_binary_cross_entropy(model, batch)
            if self.params['reg_lambda'] > 0:
                reg = self.calculate_regularization(model, batch)
                if reg is not None:
                    loss = bce + self.reg_lambda * reg
            loss.backward()
            self.update_parameters(model, optimizer, loss)
            bce_sum += bce.data[0]
            loss_sum += loss.data[0]
        return bce_sum / len(train_iter), loss_sum / len(train_iter)

    def calculate_binary_cross_entropy(self, model, batch):
        (text, text_lengths), target = batch.text, batch.target
        output = model(text, text_lengths)
        bce = F.binary_cross_entropy(output, target)
        return bce

    def calculate_regularization(self, model, batch):
        (text, text_lengths), target = batch.text, batch.target
        text, text_lengths = text.data.cpu(), text_lengths.cpu()
        target = target.data.cpu()

        token_indices = []
        target_indices = []
        pmi_values = []

        # Iterate over all token occurrences with positive labels
        # and collect the corresponding target value from the positive
        # pointwise mutual information (PMI) matrix.
        for example_index, j in target.nonzero():
            for token_index in range(text_lengths[example_index]):
                i = text[example_index, token_index]
                if self.pmi[i, j]:
                    token_indices.append(i)
                    target_indices.append(j)
                    pmi_values.append(self.pmi[i, j])

        if not token_indices:
            # We couldn't find any positive labels
            return None

        token_indices = torch.LongTensor(token_indices).cuda()
        target_indices = torch.LongTensor(target_indices).cuda()
        pmi_values = Variable(torch.FloatTensor(pmi_values).cuda(), requires_grad=False)

        token_vectors = model.token_embedding.weight[token_indices, :].unsqueeze(-2)
        target_vectors = model.target_embedding[target_indices, :].unsqueeze(-1)
        dot_products = torch.bmm(token_vectors, target_vectors).squeeze()
        reg = F.mse_loss(dot_products, pmi_values)
        return reg

    def evaluate_model(self, model, val_iter):
        bce_sum = 0
        model.eval()
        for batch in val_iter:
            bce = self.calculate_binary_cross_entropy(model, batch)
            bce_sum += bce.data[0]
        return bce_sum / len(val_iter)

    def save_model(self, model):
        file_path = os.path.join(self.model_dir, format_params(self.params) + '.pickle')
        torch.save(model.state_dict(), file_path)

    def load_model(self):
        model = self.build_model()
        file_path = os.path.join(self.model_dir, format_params(self.params) + '.pickle')
        model.load_state_dict(torch.load(file_path))
        return model

    def build_optimizer(self, model):
        optimizer = Adam(model.parameters(), lr=self.params['learning_rate'])
        return optimizer

    def update_parameters(self, model, optimizer, loss):
        optimizer.step()

    def predict_model(self, model, pred_iter):
        predictions = []
        model.eval()
        for batch in pred_iter:
            (text, text_lengths), _ = batch.text, batch.target
            output = model(text, text_lengths)
            predictions.append(output.data.cpu())
        predictions = torch.cat(predictions).numpy()
        return predictions

    def build_model(self):
        raise NotImplementedError

    def build_training_iterators(self):
        raise NotImplementedError

    def build_prediction_iterator(self):
        raise NotImplementedError


class BaseModule(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        nn.init.uniform(self.token_embedding.weight, -0.05, 0.05)
        self.target_embedding = nn.Parameter(torch.FloatTensor(6, embedding_size))
        nn.init.uniform(self.target_embedding, -0.05, 0.05)


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
