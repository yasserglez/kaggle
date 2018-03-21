import torch
from torch.autograd import Variable
from torchtext.data import Iterator

import common
import base


class MLPModule(base.BaseModule):

    def __init__(self, vocab, hidden_layers, hidden_units, hidden_nonlinearity,
                 input_dropout=0, hidden_dropout=0):
        super().__init__(vocab)

        self.dense = base.Dense(
            vocab.vectors.shape[1], len(common.LABELS),
            output_nonlinearity='sigmoid',
            hidden_layers=[hidden_units] * hidden_layers,
            hidden_nonlinearity=hidden_nonlinearity,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)
        vectors = vectors.permute(0, 2, 1)
        mean_vectors = torch.sum(vectors, -1) / Variable(text_lengths.unsqueeze(-1)).float()
        output = self.dense(mean_vectors)
        return output


class MLP(base.BaseModel):

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
        model = MLPModule(
            vocab=self.vocab,
            hidden_layers=self.params['hidden_layers'],
            hidden_units=self.params['hidden_units'],
            hidden_nonlinearity='relu',
            input_dropout=self.params['input_dropout'],
            hidden_dropout=self.params['hidden_dropout'])
        if torch.cuda.is_available():
            model = model.cuda()
        return model


if __name__ == '__main__':
    params = {
        'vocab_size': 100000,
        'max_len': 600,
        'vectors': 'glove.42B.300d',
        'hidden_layers': 2,
        'hidden_units': 600,
        'input_dropout': 0.1,
        'hidden_dropout': 0.5,
        'batch_size': 512,
        'lr_high': 0.3,
        'lr_low': 0.1,
    }
    model = MLP(params, random_seed=base.RANDOM_SEED)
    model.main()
