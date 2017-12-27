from datetime import date, datetime

import luigi
import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
import pandas as pd
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

import torch
from torch import nn
from torch.nn import functional as func
from torch import FloatTensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

def create_variable(x, *args, **kwargs):
    if torch.cuda.is_available():
        x = x.cuda(async=True)
    return Variable(x, *args, **kwargs)

from ..config import RANDOM_SEED
from ..models import FitModel, PredictModel
from .common import init_random_state, Projection, MultiTensorDataset, \
    generate_training_data, generate_prediction_data


def smape_loss(input, target):
    return torch.mean(torch.abs(target - input) / (torch.abs(target) + torch.abs(input))) * 200


class Model(nn.Module):

    def __init__(self, lstm_size_factor, hidden_layers, hidden_nonlinearily, hidden_dropout, num_days_target):
        super().__init__()

        self.lstm_size = lstm_size_factor * num_days_target
        self.lstm_h0 = Projection(num_days_target, self.lstm_size, output_nonlinearity='tanh')
        self.lstm = nn.LSTM(1, self.lstm_size, batch_first=True)

        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=nn.init.calculate_gain('tanh'))
        nn.init.orthogonal(self.lstm.weight_hh_l0, gain=nn.init.calculate_gain('tanh'))
        for param_name in ['bias_ih_l0', 'bias_hh_l0']:
            nn.init.constant(getattr(self.lstm, param_name), 1.0)

        self.lstm_output = Projection(
            self.lstm_size, num_days_target,
            hidden_layers=hidden_layers,
            hidden_nonlinearity=hidden_nonlinearily,
            dropout=hidden_dropout)

    def forward(self, last_year, days_before):
        r = days_before.unsqueeze(-1)
        h0 = self.lstm_h0(last_year).unsqueeze(0)
        c0 = Variable(h0.data.new(h0.size()).zero_())
        r, _ = self.lstm(r, (h0, c0))
        r = r[:, -1, :]  # last hidden state
        r = self.lstm_output(r)
        output = last_year + r
        return output


class RNNv1(object):
    num_days_before = luigi.IntParameter(default=30)
    lstm_size_factor = luigi.IntParameter(default=3)
    hidden_layers = luigi.IntParameter(default=2)
    hidden_nonlinearily = luigi.Parameter(default='selu')
    hidden_dropout = luigi.FloatParameter(default=0.5)
    loss = luigi.ChoiceParameter(choices=['smape', 'mse', 'mae', 'smooth_mae'], default='smape')
    learning_rate = luigi.FloatParameter(default=1e-3)
    max_grad_norm = luigi.FloatParameter(default=3.0)

    validation_ratio = luigi.FloatParameter(default=0.1, significant=False)
    batch_size = luigi.IntParameter(default=256, significant=False)
    epochs = luigi.IntParameter(default=1000, significant=False)
    patience = luigi.IntParameter(default=10, significant=False)

    @property
    def model_name(self):
        params = [
            self.num_days_before,
            self.lstm_size_factor,
            self.hidden_layers,
            self.hidden_nonlinearily,
            self.hidden_dropout,
            self.loss,
            '{:g}'.format(self.learning_rate),
            '{:g}'.format(self.max_grad_norm),
        ]
        model_name = 'rnn_v1_{}'.format('_'.join(str(p) for p in params))
        return model_name

    def _build_model(self):
        model = Model(
            lstm_size_factor=self.lstm_size_factor,
            hidden_layers=self.hidden_layers,
            hidden_nonlinearily=self.hidden_nonlinearily,
            hidden_dropout=self.hidden_dropout,
            num_days_target=(self.to_date - self.from_date).days + 1)

        if torch.cuda.is_available():
            model = model.cuda()

        print(model)

        return model


class FitRNNv1(RNNv1, FitModel):

    def _train_model(self, model, optimizer, loader, loss_func):
        model.train()
        loss_sum = num_examples = 0

        for tensors in loader:
            target, last_year, days_before = map(create_variable, tensors)
            optimizer.zero_grad()
            output = model(last_year, days_before)
            loss = loss_func(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), self.max_grad_norm)
            optimizer.step()

            # Aggregate losses to calculate a global average
            loss_sum += loss.data[0] * output.size(0)
            num_examples += output.size(0)

        loss_avg = loss_sum / num_examples
        return loss_avg

    def _evaluate_model(self, model, loader, loss_func):
        model.eval()
        loss_sum = num_examples = 0

        for batch in loader:
            target, last_year, days_before = map(create_variable, batch)
            output = model(last_year, days_before)
            loss = loss_func(output, target)

            # Aggregate losses to calculate a global average
            loss_sum += loss.data[0] * output.size(0)
            num_examples += output.size(0)

        loss_avg = loss_sum / num_examples
        return loss_avg

    def run(self):
        self.random = init_random_state(self.random_seed)

        data = self.requires()['data'].read()
        target, last_year, days_before = generate_training_data(
            data, self.deploy_date, self.from_date, self.to_date, self.num_days_before)

        tensors = train_test_split(
            target, last_year, days_before,
            test_size=self.validation_ratio, random_state=self.random)

        training_tensors = map(FloatTensor, [tensors[0], tensors[2], tensors[4]])
        training_loader = DataLoader(
            MultiTensorDataset(*training_tensors),
            batch_size=self.batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available())
        print('Training samples: {:,}'.format(tensors[0].shape[0]))

        validation_tensors = map(FloatTensor, [tensors[1], tensors[3], tensors[5]])
        validation_loader = DataLoader(
            MultiTensorDataset(*validation_tensors),
            batch_size=self.batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available())
        print('Validation samples: {:,}'.format(tensors[1].shape[0]))

        model = self._build_model()

        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        loss_func = {
            'smape': smape_loss,
            'mse': func.mse_loss,
            'mae': func.l1_loss,
            'smooth_mae': func.smooth_l1_loss,
        }[self.loss]

        best_val_loss = np.inf
        patience_count = 0

        for epoch in range(1, self.epochs + 1):
            t_start = datetime.now()
            loss = self._train_model(model, optimizer, training_loader, loss_func)
            val_loss = self._evaluate_model(model, validation_loader, loss_func)
            t_end = datetime.now()

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), self.output().path)
                best_val_loss = val_loss
                patience_count = 0

            print('Epoch {:04d}/{:04d}: loss: {:.6g} - val_loss: {:.6g} - time: {}'.format(
                epoch, self.epochs, loss, best_val_loss, str(t_end - t_start).split('.')[0]))

            if val_loss != best_val_loss:
                patience_count += 1
                if patience_count >= self.patience:
                    break


class PredictRNNv1(RNNv1, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitRNNv1(
            stage=self.stage,
            imputation=self.imputation,
            sample_ratio=self.sample_ratio,
            deploy_date=self.deploy_date,
            from_date=self.from_date,
            to_date=self.to_date,
            random_seed=self.random_seed,
            num_days_before=self.num_days_before,
            lstm_size_factor=self.lstm_size_factor,
            hidden_layers=self.hidden_layers,
            hidden_nonlinearily=self.hidden_nonlinearily,
            hidden_dropout=self.hidden_dropout,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm)
        return req

    def run(self):
        model = self._build_model()
        model.load_state_dict(torch.load(self.input()['model'].path))
        model.eval()

        data = self.requires()['data'].read()
        pages, last_year, days_before = generate_prediction_data(
            data, self.deploy_date, self.from_date, self.to_date, self.num_days_before)
        print('Prediction samples: {:,}'.format(len(pages)))

        tensors = map(FloatTensor, [np.zeros_like(last_year), last_year, days_before])
        loader = DataLoader(
            MultiTensorDataset(*tensors),
            batch_size=self.batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available())

        outputs = []
        for batch_num, batch in enumerate(loader):
            _, last_year, days_before = map(create_variable, batch)
            output = model(last_year, days_before)
            if torch.cuda.is_available():
                output = output.cpu()
            outputs.append(output.data)
        outputs = torch.cat(outputs, dim=0).numpy()

        dates = pd.date_range(start=self.from_date, end=self.to_date)

        df_parts = []
        for k in range(len(pages)):
            predictions = np.expm1(outputs[k].flatten())
            predictions = np.maximum(np.round(predictions), 0)
            df_part = pd.DataFrame({
                'Page': [pages[k]] * len(dates),
                'Date': dates,
                'Prediction': predictions,
            })
            df_parts.append(df_part)

        df = pd.concat(df_parts)
        df = df[['Page', 'Date', 'Prediction']].sort_values(['Page', 'Date'])
        df.to_csv(self.output().path, index=False)


class OptimizeRNNv1(luigi.Task):
    random_seed = luigi.IntParameter(default=RANDOM_SEED)

    choices = {
            'imputation': ['zeros', 'linear', 'quadratic'],
            'num_days_before': [365, 270, 180, 90, 30],
            'lstm_size_factor': [1, 2, 3],
            'hidden_layers': [1, 2, 3],
            'hidden_nonlinearily': ['tanh', 'relu', 'leaky_relu', 'elu', 'selu'],
            'hidden_dropout': [0.0, 0.5],
            'loss': ['smape', 'mse', 'mae', 'smooth_mae'],
            'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
            'max_grad_norm': [1.0, 3.0, 5.0],
    }

    def run(self):
        rng = RandomState(self.random_seed)
        space = {k: hp.choice(k, v) for k, v in self.choices.items()}
        while True:
            values = sample(space, rng)
            yield PredictRNNv1(
                stage=2,
                imputation=values['imputation'],
                sample_ratio=0.1,
                deploy_date=date(2017, 6, 18),
                from_date=date(2017, 7, 1),
                to_date=date(2017, 8, 31),
                num_days_before=values['num_days_before'],
                lstm_size_factor=values['lstm_size_factor'],
                hidden_layers=values['hidden_layers'],
                hidden_nonlinearily=values['hidden_nonlinearily'],
                hidden_dropout=values['hidden_dropout'],
                loss=values['loss'],
                learning_rate=values['learning_rate'],
                max_grad_norm=values['max_grad_norm'])


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
