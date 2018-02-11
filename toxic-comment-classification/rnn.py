import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Iterator

import common
import base


class RNNModule(base.BaseModule):

    def __init__(self, vocab, rnn_cell, rnn_size, rnn_layers, dense_layers, dense_nonlinearily, dense_dropout):
        super().__init__(vocab)

        h0 = torch.zeros(2 * rnn_layers, 1, rnn_size)
        self.rnn_h0 = nn.Parameter(h0, requires_grad=True)

        rnn_kwargs = dict(
                input_size=vocab.vectors.shape[1],
                hidden_size=rnn_size,
                num_layers=rnn_layers,
                bidirectional=True,
                batch_first=True)
        self.rnn = nn.LSTM(**rnn_kwargs) if rnn_cell == 'LSTM' else nn.GRU(**rnn_kwargs)
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_ih_'):
                nn.init.xavier_uniform(param)
            elif name.startswith('weight_hh_'):
                nn.init.orthogonal(param)
            elif name.startswith('bias_'):
                nn.init.constant(param, 0.0)

        self.dense = base.Dense(
            2 * rnn_size, len(common.LABELS),
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity=dense_nonlinearily,
            dropout=dense_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)

        packed_vectors = pack_padded_sequence(vectors, text_lengths.tolist(), batch_first=True)
        h0 = self.rnn_h0.expand(-1, text.shape[0], -1).contiguous()
        if isinstance(self.rnn, nn.LSTM):
            h0 = (h0, Variable(h0.data.new(h0.size()).zero_()))
        packed_rnn_output, _ = self.rnn(packed_vectors, h0)

        rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)
        # Permute to (batch, hidden_size * num_directions, seq_len)
        rnn_output = rnn_output.permute(0, 2, 1)

        # Make sure that the zero padding doesn't interfere with the maximum
        zeros_as_min = rnn_output + rnn_output.min() * (rnn_output == 0).float()
        rnn_output_max = F.max_pool1d(zeros_as_min, rnn_output.shape[-1]).squeeze(-1)

        output = self.dense(rnn_output_max)
        return output


class RNN(base.BaseModel):

    def build_training_iterators(self, preprocessed_data):
        df = common.load_data(self.mode, self.random_seed, 'train.csv')
        df['text'] = df['id'].map(preprocessed_data)
        train_df, val_df = common.split_data(df, test_size=0.1, random_state=self.random_state)

        train_dataset = base.CommentsDataset(train_df, self.fields)
        val_dataset = base.CommentsDataset(val_df, self.fields)

        train_iter, val_iter = Iterator.splits(
            (train_dataset, val_dataset), batch_size=self.params['batch_size'],
            repeat=False, sort_within_batch=True, sort_key=lambda x: len(x.text))

        return train_iter, val_iter

    def build_prediction_iterator(self, preprocessed_data):
        df = common.load_data(self.mode, self.random_seed, 'test.csv')
        df['text'] = df['id'].map(preprocessed_data)
        dataset = base.CommentsDataset(df, self.fields)

        # Reorder the examples (required by pack_padded_sequence)
        sort_indices = sorted(range(len(dataset)), key=lambda i: -len(dataset[i].text))
        pred_id = [df['id'][i] for i in sort_indices]
        dataset.examples = [dataset.examples[i] for i in sort_indices]
        pred_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, shuffle=False, sort=False)

        return pred_id, pred_iter

    def build_model(self):
        model = RNNModule(
            vocab=self.vocab,
            rnn_cell=self.params['rnn_cell'],
            rnn_size=self.params['rnn_size'],
            rnn_layers=self.params['rnn_layers'],
            dense_layers=self.params['dense_layers'],
            dense_nonlinearily='relu',
            dense_dropout=self.params['dense_dropout'])
        return model.cuda()

    def update_parameters(self, model, optimizer, loss):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        clip_grad_norm(parameters, 1.0)
        optimizer.step()
