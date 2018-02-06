import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Iterator

import common
import base


class LSTM(base.BaseModule):

    def __init__(self, vocab, bidirectional, lstm_size, lstm_mean, lstm_max, lstm_last,
                 dense_layers, dense_nonlinearily, dense_dropout):

        super().__init__(vocab)

        # TODO: Train the initial hidden state
        embedding_size = vocab.vectors.shape[1]
        self.lstm = nn.LSTM(embedding_size, lstm_size, bidirectional=bidirectional, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if name.startswith('weight_ih_'):
                nn.init.xavier_uniform(param)
            elif name.startswith('weight_hh_'):
                nn.init.orthogonal(param)
            elif name.startswith('bias_ih_'):
                nn.init.constant(param, 0.0)

        self.bidirectional = bidirectional
        self.lstm_mean, self.lstm_max, self.lstm_last = lstm_mean, lstm_max, lstm_last
        num_directions = 2 if bidirectional else 1
        num_lstm_outputs = sum([lstm_mean, lstm_max, lstm_last])

        self.dense = base.Dense(
            num_directions * num_lstm_outputs * lstm_size,
            len(common.LABELS),
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity=dense_nonlinearily,
            dropout=dense_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)

        packed_vectors = pack_padded_sequence(vectors, text_lengths.tolist(), batch_first=True)
        packed_lstm_output, (h_n, _) = self.lstm(packed_vectors)
        h_n = h_n.permute(1, 0, 2)  # batch_first
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True)
        # Permute to (batch, hidden_size * num_directions, seq_len)
        lstm_output = lstm_output.permute(0, 2, 1)

        dense_input = []
        if self.lstm_mean:
            a = torch.sum(lstm_output, -1) / Variable(text_lengths.unsqueeze(-1)).float()
            dense_input.append(a)
        if self.lstm_max:
            # Make sure that the zero padding doesn't interfere
            x = lstm_output + lstm_output.min() * (lstm_output == 0).float()
            m = F.max_pool1d(x, lstm_output.shape[-1]).squeeze(-1)
            dense_input.append(m)
        if self.lstm_last:
            last_layer_indices = [h_n.shape[1] - 1]
            if self.bidirectional:
                last_layer_indices.append(h_n.shape[1] - 2)
            l = h_n[:, last_layer_indices, :].view(h_n.shape[0], -1)
            dense_input.append(l)
        dense_input = torch.cat(dense_input, -1)

        output = self.dense(dense_input)
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
        model = LSTM(
            vocab=self.vocab,
            bidirectional=self.params['bidirectional'],
            lstm_size=self.params['lstm_size'],
            lstm_mean=self.params['lstm_mean'],
            lstm_max=self.params['lstm_max'],
            lstm_last=self.params['lstm_last'],
            dense_layers=self.params['dense_layers'],
            dense_nonlinearily='relu',
            dense_dropout=self.params['dense_dropout'])
        return model

    def update_parameters(self, model, optimizer, loss):
        # TODO: Implement gradient clipping
        optimizer.step()
