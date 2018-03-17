# Based on https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Iterator

import common
import base


class GRUModule(base.BaseModule):

    def __init__(self, vocab, rnn_size, rnn_dropout, proj_size, proj_layers, proj_dropout,
                 dense_layers, dense_dropout):
        super().__init__(vocab)

        embedding_size = vocab.vectors.shape[1]

        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=rnn_size,
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
            self.rnn = base.WeightDrop(self.rnn, ['weight_hh_l0'], dropout=rnn_dropout)

        self.proj = base.Dense(
            embedding_size + 2 * rnn_size, proj_size,
            output_nonlinearity='relu',
            hidden_layers=proj_layers,
            dropout=proj_dropout)

        self.dense = base.Dense(
            proj_size, len(common.LABELS),
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity='relu',
            dropout=dense_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)

        packed_vectors = pack_padded_sequence(vectors, text_lengths.tolist(), batch_first=True)
        packed_rnn_output, _ = self.rnn(packed_vectors)
        rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)

        batch_size = vectors.shape[0]
        rnn_size = rnn_output.shape[-1] // 2
        padding = Variable(torch.zeros(batch_size, 1, rnn_size), requires_grad=False).cuda()
        if rnn_output.shape[1] > 1:
            context_left = torch.cat([padding, rnn_output[:, 1:, :rnn_size]], 1)
            context_right = torch.cat([rnn_output[:, :-1, rnn_size:], padding], 1)
            context_vectors = torch.cat([context_left, vectors, context_right], -1)
        else:
            context_vectors = torch.cat([padding, vectors, padding], -1)

        proj_vectors = self.proj(context_vectors)
        proj_vectors = proj_vectors.permute(0, 2, 1)
        proj_output = F.max_pool1d(proj_vectors, proj_vectors.shape[-1]).squeeze(-1)

        output = self.dense(proj_output)
        return output


class GRU(base.BaseModel):

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
        model = GRUModule(
            vocab=self.vocab,
            rnn_size=self.params['rnn_size'],
            rnn_dropout=self.params['rnn_dropout'],
            proj_size=self.params['proj_size'],
            proj_layers=self.params['proj_layers'],
            proj_dropout=self.params['proj_dropout'],
            dense_layers=self.params['dense_layers'],
            dense_dropout=self.params['dense_dropout'])
        return model.cuda()

    def update_parameters(self, model, optimizer, loss):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        clip_grad_norm(parameters, 0.2)
        optimizer.step()


if __name__ == '__main__':
    params = {
        'vocab_size': 70000,
        'max_len': 150,
        'vectors': 'glove.42B.300d',
        'rnn_size': 500,
        'rnn_dropout': 0.2,
        'proj_size': 150,
        'proj_layers': 0,
        'proj_dropout': 0.3,
        'dense_layers': 1,
        'dense_dropout': 0.5,
        'batch_size': 128,
        'lr_high': 0.5,
        'lr_low': 0.01,
    }
    model = GRU(params, random_seed=base.RANDOM_SEED)
    model.main()
