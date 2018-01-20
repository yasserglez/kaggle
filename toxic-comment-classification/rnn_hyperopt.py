import traceback

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from rnn import RNN


choices = {
    'token': ['word'],
    'lower': [True],
    'min_freq': [3],
    'max_len': [10000],
    'batch_size': [64],
    'learning_rate': [0.001],
    'max_epochs': [100],
    'patience': [10],
    'embedding_size': [32, 64, 128],
    'lstm_size': [64, 128, 256],
    'dense_layers': [1],
    'reg_alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    'reg_k': [1],
    'dropout': [0.0],
}

space = {k: hp.choice(k, v) for k, v in choices.items()}

while True:
    params = sample(space)
    if params['lstm_size'] != 2 * params['embedding_size']:
        continue
    try:
        rnn = RNN('cross_validation', params, 432789)
        rnn.main()
    except Exception:
        traceback.print_exc()
