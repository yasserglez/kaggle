import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Iterator

import common
import base


class LSTM(base.BaseModule):

    def __init__(self, vocab, lstm_size, dense_layers, dense_nonlinearily, dense_dropout):

        super().__init__(vocab)

        lstm_layers = 1
        h0 = torch.zeros(2 * lstm_layers, 1, lstm_size)
        self.lstm_h0 = nn.Parameter(h0, requires_grad=True)

        embedding_size = vocab.vectors.shape[1]
        self.lstm = nn.LSTM(embedding_size, lstm_size, lstm_layers, bidirectional=True, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if name.startswith('weight_ih_'):
                nn.init.xavier_uniform(param)
            elif name.startswith('weight_hh_'):
                nn.init.orthogonal(param)
            elif name.startswith('bias_ih_'):
                nn.init.constant(param, 0.0)

        self.dense = base.Dense(
            2 * lstm_size, len(common.LABELS),
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity=dense_nonlinearily,
            dropout=dense_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)

        packed_vectors = pack_padded_sequence(vectors, text_lengths.tolist(), batch_first=True)
        h0 = self.lstm_h0.expand(-1, text.shape[0], -1).contiguous()
        c0 = Variable(h0.data.new(h0.size()).zero_())
        packed_lstm_output, _ = self.lstm(packed_vectors, (h0, c0))

        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True)
        # Permute to (batch, hidden_size * num_directions, seq_len)
        lstm_output = lstm_output.permute(0, 2, 1)

        # Make sure that the zero padding doesn't interfere with the maximum
        zeros_as_min = lstm_output + lstm_output.min() * (lstm_output == 0).float()
        lstm_output_max = F.max_pool1d(zeros_as_min, lstm_output.shape[-1]).squeeze(-1)

        output = self.dense(lstm_output_max)
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
            lstm_size=self.params['lstm_size'],
            dense_layers=self.params['dense_layers'],
            dense_nonlinearily='relu',
            dense_dropout=self.params['dense_dropout'])
        return model

    def update_parameters(self, model, optimizer, loss):
        # TODO: Implement gradient clipping
        optimizer.step()
