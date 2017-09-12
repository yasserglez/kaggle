import sys
import multiprocessing
from functools import partial
from datetime import timedelta

from dateutil.relativedelta import relativedelta
import numpy as np
from numpy.random import RandomState
import torch
from torch import nn
from torch.utils.data import Dataset


def init_random_state(random_seed):
    rng = RandomState(random_seed)
    np.random.seed(int.from_bytes(rng.bytes(4), byteorder=sys.byteorder))
    torch.manual_seed(int.from_bytes(rng.bytes(4), byteorder=sys.byteorder))
    return rng


def layer_units(input_size, output_size, num_layers=0):
    units = np.linspace(input_size, output_size, num_layers + 2)
    units = list(map(int, np.round(units, 0)))
    return units


class Projection(nn.Module):

    def __init__(self, input_size, output_size, output_nonlinearity=None,
                 hidden_layers=0, hidden_nonlinearity=None, dropout=0):

        super(Projection, self).__init__()

        nonlinearities = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
        }

        layers = []
        units = layer_units(input_size, output_size, hidden_layers)
        for in_size, out_size in zip(units, units[1:]):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_size, out_size))
            if hidden_nonlinearity:
                layers.append(nonlinearities[hidden_nonlinearity])
        # Remove the last hidden nonlinearity (if any)
        if hidden_nonlinearity:
            layers.pop()
        # and add the output nonlinearity (if any)
        if output_nonlinearity:
            layers.append(nonlinearities[output_nonlinearity])

        self.projection = nn.Sequential(*layers)

        for layer in layers:
            if isinstance(layer, nn.Linear):
                gain = 1.0
                if hidden_nonlinearity in {'tanh', 'relu', 'leaky_relu'}:
                    gain = nn.init.calculate_gain(hidden_nonlinearity)
                nn.init.xavier_uniform(layer.weight, gain=gain)
                nn.init.constant(layer.bias, 0.0)

    def forward(self, x):
        return self.projection(x)


class MultiTensorDataset(Dataset):

    def __init__(self, target_tensor, *data_tensors):
        for data_tensor in data_tensors:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.target_tensor = target_tensor
        self.data_tensors = data_tensors

    def __getitem__(self, index):
        result = [self.target_tensor[index]]
        result.extend([tensor[index] for tensor in self.data_tensors])
        return result

    def __len__(self):
        return self.target_tensor.size(0)


def generate_training_data(data, deploy_date, from_date, to_date, num_days_before):

    assert deploy_date < from_date
    assert from_date < to_date

    target_values = []
    last_year_values = []
    days_before_values = []

    # Discard any information after deploy_date
    data = data[data['Date'] <= deploy_date]
    min_date = data['Date'].min().date()
    max_date = data['Date'].max().date()
    print('Training data: {} - {}'.format(min_date, max_date))

    # How many days in the future does the prediction window start?
    delta = timedelta(days=(to_date - max_date).days)
    d = deploy_date - delta
    f = from_date - delta
    t = to_date - delta

    # The target window will shift with 50% overlap
    num_days_target = (to_date - from_date).days + 1
    stride = timedelta(days=num_days_target // 2)

    grouped_data = [(page, page_data) for page, page_data in data.groupby('Page')]

    # Generate training samples until the same period last year, keep the weekdays aligned
    while f >= from_date + relativedelta(years=-1, weekday=from_date.weekday()):
        last_year_f = f + relativedelta(years=-1, weekday=f.weekday())
        last_year_t = t + relativedelta(years=-1, weekday=t.weekday())

        print('Generating samples...')
        print('target:      {} - {}'.format(f, t))
        print('last_year:   {} - {}'.format(last_year_f, last_year_t))
        print('days_before: {} - {}'.format(d - timedelta(days=num_days_before - 1), d))

        with multiprocessing.Pool(processes=4) as pool:
            func = partial(generate_training_sample,
                last_year_f=last_year_f, last_year_t=last_year_t, d=d, f=f, t=t,
                num_days_target=num_days_target, num_days_before=num_days_before)
            results = pool.starmap(func, grouped_data, chunksize=1000)
            for result in results:
                if result:
                    target, last_year, days_before = result
                    target_values.append(target)
                    last_year_values.append(last_year)
                    days_before_values.append(days_before)

        # Shift the training window
        d -= stride
        f -= stride
        t -= stride

    target = np.log1p(np.array(target_values))
    last_year = np.log1p(np.array(last_year_values))
    days_before = np.log1p(np.array(days_before_values))

    return target, last_year, days_before


def generate_training_sample(page, page_data,
        last_year_f, last_year_t, d, f, t, num_days_target, num_days_before):

    mask = (page_data['Date'] >= f) & (page_data['Date'] <= t)
    target = page_data[mask].sort_values('Date')['Views'].values
    assert target.shape[0] == num_days_target

    mask = (page_data['Date'] >= last_year_f) & (page_data['Date'] <= last_year_t)
    last_year = page_data[mask].sort_values('Date')['Views'].values
    assert last_year.shape[0] == num_days_target

    mask = page_data['Date'] <= d
    days_before = page_data[mask].nlargest(num_days_before, 'Date').sort_values('Date')['Views'].values
    assert days_before.shape[0] == num_days_before

    if any(last_year > 0) and any(days_before > 0):
        return target, last_year, days_before


def generate_prediction_data(data, deploy_date, from_date, to_date, num_days_before):

    assert deploy_date < from_date
    assert from_date < to_date

    pages = []
    last_year_values = []
    days_before_values = []

    num_days_target = (to_date - from_date).days + 1
    last_year_from_date = from_date + relativedelta(years=-1, weekday=from_date.weekday())
    last_year_to_date = to_date + relativedelta(years=-1, weekday=to_date.weekday())

    print('Generating samples...')
    print('last_year:   {} - {}'.format(last_year_from_date, last_year_to_date))
    print('days_before: {} - {}'.format(deploy_date - timedelta(days=num_days_before - 1), deploy_date))

    grouped_data = [(page, page_data) for page, page_data in data.groupby('Page')]

    with multiprocessing.Pool(processes=4) as pool:
        func = partial(generate_prediction_sample,
            last_year_from_date=last_year_from_date, last_year_to_date=last_year_to_date,
            deploy_date=deploy_date, num_days_target=num_days_target, num_days_before=num_days_before)
        results = pool.starmap(func, grouped_data, chunksize=1000)
        for result in results:
            page, last_year, days_before = result
            pages.append(page)
            last_year_values.append(last_year)
            days_before_values.append(days_before)

    last_year = np.log1p(np.array(last_year_values))
    days_before = np.log1p(np.array(days_before_values))

    return pages, last_year, days_before


def generate_prediction_sample(page, page_data,
        last_year_from_date, last_year_to_date, deploy_date, num_days_target, num_days_before):

    mask = (page_data['Date'] >= last_year_from_date) & (page_data['Date'] <= last_year_to_date)
    last_year = page_data[mask].sort_values('Date')['Views'].values
    assert last_year.shape[0] == num_days_target

    mask = page_data['Date'] <= deploy_date
    days_before = page_data[mask].nlargest(num_days_before, 'Date').sort_values('Date')['Views'].values
    assert days_before.shape[0] == num_days_before

    return page, last_year, days_before
