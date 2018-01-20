import traceback

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from mlp import MLP


choices = {
    'token': ['word'],
    'lower': [True],
    'min_freq': [5],
    'max_len': [500],
    'batch_size': [512],
    'learning_rate': [0.001],
    'max_epochs': [100],
    'patience': [10],
    'embedding_size': [32, 64, 128, 256],
    'dense_layers': [1],
    'reg_lambda': [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'reg_k': [1],
    'dropout': [0.1],
}

space = {k: hp.choice(k, v) for k, v in choices.items()}

while True:
    params = sample(space)
    try:
        rnn = MLP('cross_validation', params, 46432168)
        rnn.main()
    except Exception:
        traceback.print_exc()
