import os
import struct
import logging
import multiprocessing

import ujson
import numpy as np


logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


CPU_COUNT = multiprocessing.cpu_count()


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'predictions')

for d in [MODEL_DIR, PREDICTIONS_DIR]:
    if not os.path.isdir(d):
        os.makedirs(d)


# Assign an integer to each word to be predicted.
WORD2LABEL = {}
for label, f in enumerate(sorted(os.listdir(f'{DATA_DIR}/train_simplified/'))):
    WORD2LABEL[f[:-4]] = label
LABEL2WORD = dict((v, k) for k, v in WORD2LABEL.items())


# Binary format based on https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py

def pack_example(example, fout):
    fout.write(struct.pack('Q', int(example['key_id'])))
    fout.write(struct.pack('H', WORD2LABEL[example['word']]))
    strokes = ujson.loads(example['drawing'])
    num_strokes = len(strokes)
    fout.write(struct.pack('H', num_strokes))
    for i in range(num_strokes):
        num_points = len(strokes[i][0])
        fout.write(struct.pack('H', num_points))
        format = str(num_points) + 'B'
        fout.write(struct.pack(format, *strokes[i][0]))
        fout.write(struct.pack(format, *strokes[i][1]))

def unpack_example(fin):
    key_id, = struct.unpack('Q', fin.read(8))
    label, = struct.unpack('H', fin.read(2))
    strokes = []
    num_strokes, = struct.unpack('H', fin.read(2))
    for i in range(num_strokes):
        num_points, = struct.unpack('H', fin.read(2))
        format = str(num_points) + 'B'
        x = list(struct.unpack(format, fin.read(num_points)))
        y = list(struct.unpack(format, fin.read(num_points)))
        strokes.append([x, y])
    strokes = [np.array(s).T for s in strokes]
    return {
        'key_id': str(key_id),
        'strokes': strokes,
        'label': label,
    }

def unpack_examples(fin):
    while True:
        try:
            yield unpack_example(fin)
        except struct.error:
            break
