import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Iterator

import common
import base


# Based on https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
class WeightDrop(torch.nn.Module):

    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def no_op(*args, **kwargs):
        return

    def _setup(self):
        # Temporary solution to an issue regarding compacting weights re: cuDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.no_op

        for w_name in self.weights:
            w = getattr(self.module, w_name)
            del self.module._parameters[w_name]
            self.module.register_parameter(w_name + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for w_name in self.weights:
            raw_w = getattr(self.module, w_name + '_raw')
            w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, w_name, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class RNNModule(base.BaseModule):

    def __init__(self, vocab, rnn_size, rnn_layers, rnn_dropout,
                 dense_layers, dense_nonlinearily, dense_dropout):
        super().__init__(vocab)

        h0 = torch.zeros(2 * rnn_layers, 1, rnn_size)
        self.rnn_h0 = nn.Parameter(h0, requires_grad=True)

        self.rnn = nn.LSTM(
            input_size=vocab.vectors.shape[1],
            hidden_size=rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True)

        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_ih_'):
                nn.init.xavier_uniform(param)
            elif name.startswith('weight_hh_'):
                nn.init.orthogonal(param)
            elif name.startswith('bias_'):
                nn.init.constant(param, 0.0)

        if rnn_dropout:
            weights = ['weight_hh_l{}'.format(k) for k in range(rnn_layers)]
            self.rnn = WeightDrop(self.rnn, weights, dropout=rnn_dropout)

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
        c0 = Variable(h0.data.new(h0.size()).zero_().contiguous())
        packed_rnn_output, _ = self.rnn(packed_vectors, (h0, c0))

        rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)
        # Permute to (batch, hidden_size * num_directions, seq_len)
        rnn_output = rnn_output.permute(0, 2, 1)

        # Make sure that the zero padding doesn't interfere with the maximum
        zeros_as_min = rnn_output + rnn_output.min() * (rnn_output == 0).float()
        rnn_output_max = F.max_pool1d(zeros_as_min, rnn_output.shape[-1]).squeeze(-1)

        output = self.dense(rnn_output_max)
        return output


class RNN(base.BaseModel):

    def build_train_iterator(self, df):
        dataset = base.CommentsDataset(df, self.fields)
        train_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, sort_within_batch=True, shuffle=True,
            sort_key=lambda x: len(x.text))
        return train_iter

    def build_prediction_iterator(self, df):
        dataset = base.CommentsDataset(df, self.fields)
        # Reorder the examples (required by pack_padded_sequence)
        sort_indices = sorted(range(len(dataset)), key=lambda i: -len(dataset[i].text))
        pred_id = [df['id'].iloc[i] for i in sort_indices]
        dataset.examples = [dataset.examples[i] for i in sort_indices]
        pred_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, shuffle=False, sort=False)
        return pred_id, pred_iter

    def build_model(self):
        model = RNNModule(
            vocab=self.vocab,
            rnn_size=self.params['rnn_size'],
            rnn_layers=1,
            rnn_dropout=self.params['rnn_dropout'],
            dense_layers=self.params['dense_layers'],
            dense_nonlinearily='relu',
            dense_dropout=self.params['dense_dropout'])
        return model.cuda()

    def update_parameters(self, model, optimizer, loss):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        clip_grad_norm(parameters, 1.0)
        optimizer.step()


if __name__ == '__main__':
    params = {
        'vocab_size': 30000,
        'max_len': 300,
        'vectors': 'glove.42B.300d',
        'rnn_size': 500,
        'rnn_dropout': 0.2,
        'dense_layers': 1,
        'dense_dropout': 0.5,
        'batch_size': 128,
        'lr_high': 0.5,
        'lr_low': 0.01,
    }
    model = RNN('rnn', params, random_seed=base.RANDOM_SEED)
    model.main()
