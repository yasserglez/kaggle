import common

import torch
from torch.autograd import Variable
from torchtext.data import Iterator


class MLPModule(common.BaseModule):

    def __init__(self, vocab_size, embedding_size, dense_layers, nonlinearity, dropout):
        super().__init__(vocab_size, embedding_size)
        self.dense = common.Dense(
            embedding_size, 6,
            output_nonlinearity='sigmoid',
            hidden_layers=dense_layers,
            hidden_nonlinearity=nonlinearity,
            dropout=dropout)

    def forward(self, text, text_lengths):
        token_vectors = self.token_embedding(text)
        text_lengths = Variable(text_lengths.float().unsqueeze(-1), requires_grad=False)
        mean_vectors = torch.div(torch.sum(token_vectors, dim=1), text_lengths)
        output = self.dense(mean_vectors)
        return output


class MLP(common.BaseModel):

    def build_training_iterators(self):
        df = common.load_data(self.mode, self.random_seed, 'train.csv')
        train_df, val_df = common.split_data(df, test_size=0.1, random_state=self.random_state)

        train_dataset = common.ToxicCommentDataset(train_df, self.text_field)
        val_dataset = common.ToxicCommentDataset(val_df, self.text_field)

        train_iter, val_iter = Iterator.splits(
            (train_dataset, val_dataset),
            batch_size=self.params['batch_size'],
            repeat=False, sort=False)

        return train_iter, val_iter

    def build_prediction_iterator(self):
        df = common.load_data(self.mode, self.random_seed, 'test.csv')
        dataset = common.ToxicCommentDataset(df, self.text_field)

        pred_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, shuffle=False, sort=False)

        return df['id'], pred_iter

    def build_model(self):
        model = MLPModule(
            vocab_size=len(self.vocab),
            embedding_size=self.params['embedding_size'],
            dense_layers=self.params['dense_layers'],
            nonlinearity='relu',
            dropout=self.params['dropout'])
        return model


if __name__ == '__main__':
    params = {
        'token': 'word',
        'lower': True,
        'min_freq': 5,
        'max_len': None,
        'batch_size': 512,
        'learning_rate': 0.001,
        'max_epochs': 100,
        'patience': 10,
        'embedding_size': 256,
        'dense_layers': 1,
        'dropout': 0.1,
        'reg_lambda': 1.0,
        'reg_k': 1,
    }
    for mode in ['cross_validation', 'submission']:
        mlp = MLP(mode, params, random_seed=46432168)
        mlp.main()
