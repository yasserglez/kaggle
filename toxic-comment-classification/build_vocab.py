import multiprocessing as mp

import common


PARAMS = []

for min_freq in [1, 3, 5, 7, 11]:
    PARAMS.extend([
        {'token': 'word', 'lower': True, 'min_freq': min_freq},
    ])

def process(params):
    text_field = common.TextField(params, pad_token=None)
    common.build_vocab(text_field)

with mp.Pool(mp.cpu_count()) as pool:
    pool.map(process, PARAMS)
