# Based on https://arxiv.org/abs/1612.08083

import torch
from torch import nn, autograd
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torchtext.data import Iterator

import common
import base


class GLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.conv_linear = nn.Conv1d(in_channels, out_channels, kernel_size)
        self._init_conv(self.conv_linear)
        self.conv_linear = weight_norm(self.conv_linear)

        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size)
        self._init_conv(self.conv_gate)
        self.conv_gate = weight_norm(self.conv_gate)

    def _init_conv(self, module):
        nn.init.normal(module.weight, mean=0, std=0.01)
        nn.init.constant(module.bias, 0.0)

    def forward(self, x):
        # Zero-pad the beginning of the sequence with kernel_size - 1 elements
        batch_size = x.shape[0]
        padding = torch.zeros(batch_size, self.in_channels, self.kernel_size - 1)
        padding = autograd.Variable(padding, requires_grad=False)
        if torch.cuda.is_available():
            padding = padding.cuda()
        x_padded = torch.cat([padding, x], dim=-1)

        a = self.conv_linear(x_padded)
        b = self.conv_gate(x_padded)
        h = a * F.sigmoid(b)
        return h


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super().__init__()

        self.num_layers = num_layers
        for i in range(1, self.num_layers + 1):
            layer = GLU(in_channels, out_channels, kernel_size)
            setattr(self, f'layer{i}', layer)

    def forward(self, x):
        h = x
        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f'layer{i}')
            h = layer(h)
        output = x + h
        return output


class GCNNModule(base.BaseModule):

    def __init__(self, vocab, num_blocks, num_layers, num_channels, kernel_size,
                 dense_layers, dense_dropout):
        super().__init__(vocab)

        embedding_size = vocab.vectors.shape[1]
        self.glu0 = GLU(embedding_size, num_channels, kernel_size)

        self.num_blocks = num_blocks
        for i in range(1, self.num_blocks + 1):
            block = ResidualBlock(num_channels, num_channels, kernel_size, num_layers)
            setattr(self, f'block{i}', block)

        self.dense = base.Dense(
            num_channels, len(common.LABELS),
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity='relu',
            input_dropout=dense_dropout,
            hidden_dropout=dense_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)
        vectors = vectors.permute(0, 2, 1).contiguous()

        conv_output = self.glu0(vectors)
        for i in range(1, self.num_blocks + 1):
            block = getattr(self, f'block{i}')
            conv_output = block(conv_output)

        pooling_output = F.max_pool1d(conv_output, conv_output.shape[-1])
        output = self.dense(pooling_output.squeeze(-1))

        return output


class GCNN(base.BaseModel):

    def build_train_iterator(self, df):
        dataset = base.CommentsDataset(df, self.fields)
        train_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, shuffle=True)
        return train_iter

    def build_prediction_iterator(self, df):
        dataset = base.CommentsDataset(df, self.fields)
        pred_id = list(df['id'].values)
        pred_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, shuffle=False, sort=False)
        return pred_id, pred_iter

    def build_model(self):
        model = GCNNModule(
            vocab=self.vocab,
            num_blocks=self.params['num_blocks'],
            num_layers=self.params['num_layers'],
            num_channels=self.params['num_channels'],
            kernel_size=self.params['kernel_size'],
            dense_layers=self.params['dense_layers'],
            dense_dropout=self.params['dense_dropout'])
        return model.cuda()

    def update_parameters(self, model, optimizer, loss):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        clip_grad_norm(parameters, 0.1)
        optimizer.step()


if __name__ == '__main__':
    params = {
        'vocab_size': 100000,
        'max_len': 300,
        'vectors': 'glove.42B.300d',
        'num_blocks': 1,
        'num_layers': 2,
        'num_channels': 128,
        'kernel_size': 3,
        'dense_layers': 0,
        'dense_dropout': 0.5,
        'batch_size': 64,
        'lr_high': 1.0,
        'lr_low': 0.2,
    }
    model = GCNN(params, random_seed=base.RANDOM_SEED)
    model.main()
