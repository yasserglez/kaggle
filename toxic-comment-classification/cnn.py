from torch import nn
import torch.nn.functional as F
from torchtext.data import Iterator

import common
import base


class ConvBlock(nn.Module):

    def __init__(self, channels, dropout=0):
        super().__init__()

        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(channels)
        self._init_batchnorm(self.batchnorm1)
        if dropout:
            self.dropout1 = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self._init_conv(self.conv1)

        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(channels)
        self._init_batchnorm(self.batchnorm2)
        if dropout:
            self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self._init_conv(self.conv2)

    def _init_batchnorm(self, module):
        nn.init.constant(module.weight, 1.0)
        nn.init.constant(module.bias, 0.0)

    def _init_conv(self, module):
        nn.init.normal(module.weight, mean=0, std=0.01)
        nn.init.constant(module.bias, 0.0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.batchnorm1(x)
        if hasattr(self, 'dropout1'):
            x = self.dropout1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        if hasattr(self, 'dropout2'):
            x = self.dropout2(x)
        x = self.conv2(x)
        return x


class CNNModule(base.BaseModule):

    def __init__(self, vocab, conv_blocks, conv_dropout, dense_layers, dense_nonlinearily, dense_dropout):
        super().__init__(vocab)

        self.conv_blocks = conv_blocks
        channels = vocab.vectors.shape[1]
        for k in range(1, conv_blocks + 1):
            setattr(self, f'conv_block{k}', ConvBlock(channels, conv_dropout))

        self.dense = base.Dense(
            channels, len(common.LABELS),
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity=dense_nonlinearily,
            input_dropout=dense_dropout,
            hidden_dropout=dense_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)
        vectors = vectors.permute(0, 2, 1).contiguous()

        conv_output = vectors + self.conv_block1(vectors)
        for k in range(2, self.conv_blocks + 1):
            conv_output = F.max_pool1d(conv_output, kernel_size=3, stride=2)
            block = getattr(self, f'conv_block{k}')
            conv_output = conv_output + block(conv_output)

        pooling_output = F.max_pool1d(conv_output, conv_output.shape[-1])
        output = self.dense(pooling_output.squeeze(-1))

        return output


class CNN(base.BaseModel):

    def build_train_iterator(self, preprocessed_data):
        df = common.load_data(self.random_seed, 'train')
        df['text'] = df['id'].map(preprocessed_data)
        dataset = base.CommentsDataset(df, self.fields)
        train_iter = Iterator(dataset, batch_size=self.params['batch_size'], repeat=False)
        return train_iter

    def build_prediction_iterator(self, preprocessed_data, dataset):
        df = common.load_data(self.random_seed, dataset)
        df['text'] = df['id'].map(preprocessed_data)
        dataset = base.CommentsDataset(df, self.fields)
        pred_id = list(df['id'].values)
        pred_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, shuffle=False, sort=False)
        return pred_id, pred_iter

    def build_model(self):
        model = CNNModule(
            vocab=self.vocab,
            conv_blocks=self.params['conv_blocks'],
            conv_dropout=self.params['conv_dropout'],
            dense_layers=self.params['dense_layers'],
            dense_nonlinearily='relu',
            dense_dropout=self.params['dense_dropout'])
        return model.cuda()


if __name__ == '__main__':
    params = {
        'vocab_size': 50000,
        'max_len': 400,
        'vectors': 'glove.42B.300d',
        'conv_blocks': 1,
        'conv_dropout': 0.1,
        'dense_layers': 1,
        'dense_dropout': 0.5,
        'batch_size': 256,
        'lr_high': 0.01,
        'lr_low': 0.001,
    }
    model = CNN('cnn', params, random_seed=42)
    model.main()
