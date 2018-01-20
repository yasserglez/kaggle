import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Iterator

import common


class LSTM(common.BaseModule):

    def __init__(self, vocab_size, embedding_size, lstm_size,
                 dense_layers, dense_nonlinearily, dropout):

        super().__init__(vocab_size, embedding_size)

        # TODO: Train the initial hidden state
        self.lstm = nn.LSTM(embedding_size, lstm_size, bidirectional=True, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if name.startswith('weight_ih_'):
                nn.init.xavier_uniform(param)
            elif name.startswith('weight_hh_'):
                nn.init.orthogonal(param)
            elif name.startswith('bias_ih_'):
                nn.init.constant(param, 0.0)

        self.dense = common.Dense(
            2 * lstm_size, 6,
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity=dense_nonlinearily,
            dropout=dropout)

    def forward(self, text, text_lengths):
        vectors = self.token_embedding(text)

        packed_vectors = pack_padded_sequence(vectors, text_lengths.tolist(), batch_first=True)
        packed_lstm_output, _ = self.lstm(packed_vectors)
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True)

        lstm_output = lstm_output.permute(0, 2, 1)
        pooling_output = F.max_pool1d(lstm_output, lstm_output.shape[-1])

        output = self.dense(pooling_output.squeeze(-1))
        return output


class RNN(common.BaseModel):

    def build_training_iterators(self):
        df = common.load_data(self.mode, self.random_seed, 'train.csv')
        train_df, val_df = common.split_data(df, test_size=0.1, random_state=self.random_state)

        train_dataset = common.ToxicCommentDataset(train_df, self.text_field)
        val_dataset = common.ToxicCommentDataset(val_df, self.text_field)

        train_iter, val_iter = Iterator.splits(
            (train_dataset, val_dataset), batch_size=self.params['batch_size'],
            repeat=False, sort_within_batch=True, sort_key=lambda x: len(x.text))

        return train_iter, val_iter

    def build_prediction_iterator(self):
        df = common.load_data(self.mode, self.random_seed, 'test.csv')
        dataset = common.ToxicCommentDataset(df, self.text_field)

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
            len(self.vocab),
            embedding_size=self.params['embedding_size'],
            lstm_size=self.params['lstm_size'],
            dense_layers=self.params['dense_layers'],
            dense_nonlinearily='relu',
            dropout=self.params['dropout'])
        return model

    def update_parameters(self, model, optimizer, loss):
        # TODO: Implement gradient clipping
        optimizer.step()


if __name__ == '__main__':
    params = {
        'token': 'char',
        'lower': False,
        'min_freq': 11,
        'max_len': 1000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'max_epochs': 100,
        'patience': 10,
        'embedding_size': 16,
        'lstm_size': 256,
        'dense_layers': 1,
        'dropout': 0.1,
        'reg_lambda': 0.0,
        'reg_k': 1,
    }
    for mode in ['cross_validation', 'submission']:
        rnn = RNN(mode, params, random_seed=1584965)
        rnn.main()
