from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import attr


@attr.s(auto_attribs=True)
class Config(object):
    random_seed: int
    num_epochs: int
    batch_size: int
    # Adam parameters.
    lr: float
    lr_warmup: float
    beta1: float
    beta2: float
    eps: float
    # Regularization.
    weight_decay: float
    dropout: float
    # Pretrained BERT model. Choices: 'bert-base-uncased',
    # 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased'.
    bert_model: str
    # Embedding size for the distance between the pronoun and the mentions.
    dist_size: int
    # Size of the network processing the pronoun and mention pairs.
    coref_layers: int
    coref_size: int
    # Size of the final hidden layers.
    hidden_layers: int
    hidden_size: int

    def save(self, file_path: Path) -> None:
        d = attr.asdict(self)
        with file_path.open('w') as f:
            f.write(json.dumps(d, indent=4) + '\n')

    @classmethod
    def load(self, file_path: Path) -> Config:
        with file_path.open() as f:
            d = json.load(f)
        if 'random_seed' not in d:
            d['random_seed'] = int.from_bytes(os.urandom(4), sys.byteorder)
        config = Config(**d)
        return config
