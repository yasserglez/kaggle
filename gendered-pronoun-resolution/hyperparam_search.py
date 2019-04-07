import os
import sys
import json
import shutil
import argparse
import logging
from datetime import datetime
from typing import Dict
from pathlib import Path

import torch
from tqdm import tqdm
from hyperopt import hp, fmin, rand, tpe, STATUS_OK

from config import Config
from train import train


ALGORITHMS = {'rand': rand, 'tpe': tpe}


def objective(config_dict: Dict, device: torch.device,
              progress_bar: tqdm, base_output_dir: Path) -> Dict:
    # Allows to use quniform, qloguniform, etc. which returns a float
    int_fields = [
        "num_epochs",
        "batch_size",
        "dist_size",
        "coref_layers",
        "coref_size",
        "hidden_layers",
        "hidden_size",
    ]
    for k in int_fields:
        config_dict[k] = int(config_dict[k])
    if 'random_seed' not in config_dict:
        config_dict['random_seed'] = int.from_bytes(os.urandom(4), sys.byteorder)
    config = Config(**config_dict)
    output_dir = (base_output_dir / 'trial_{}'.format(progress_bar.n + 1)).resolve()
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Logging initialization
    root = logging.getLogger()
    if root.handlers:
        assert len(root.handlers) == 1
        handler = root.handlers.pop()
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(output_dir / 'train.log', mode='w')])
    logging.getLogger('pytorch_pretrained_bert').setLevel(logging.WARNING)

    val_loss = train(config, device, output_dir)
    progress_bar.update()

    return {
        'status': STATUS_OK,
        'config': config_dict,
        'loss': val_loss,
    }


def load_config_space(config_space_file: Path) -> Dict:
    with config_space_file.open() as f:
        config_space = json.load(f)
    space = {}
    for k, v in config_space.items():
        if isinstance(v, list):
            hp_func = getattr(hp, v[0])
            hp_func_args = v[1:]
            space[k] = hp_func(k, *hp_func_args)
        else:
            space[k] = v
    return space


def main(args: argparse.Namespace) -> None:
    config_space_file, output_dir = map(
        lambda s: Path(s).resolve(), [args.config_space, args.output])
    device = torch.device(args.device)

    progress_bar = tqdm(total=args.max_trials, ncols=80, unit='trial')
    space = load_config_space(config_space_file)
    algorithm = ALGORITHMS[args.algorithm]
    fmin(lambda x: objective(x, device, progress_bar, output_dir),
         space=space, algo=algorithm.suggest,
         max_evals=args.max_trials)
    progress_bar.close()

    shutil.copyfile(config_space_file, output_dir / 'config_space.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space', metavar='JSON_FILE',
        help='Model configuration search space.', required=True)
    parser.add_argument('--device', metavar='DEVICE', default='cuda',
        help='PyTorch device to use for computation.')
    parser.add_argument('--algorithm',
        help='hyperopt optimization algorithm.',
        choices=list(ALGORITHMS.keys()), default='tpe')
    parser.add_argument('--max_trials', type=int, metavar='N', default=100,
        help='Maximum number of configurations to evaluate.')
    parser.add_argument('--output', metavar='OUTPUT_DIR',
        help='Output directory. A timestamped folder '
             'will be created if not specified.')
    args = parser.parse_args()
    if args.output is None:
        dirname = datetime.now().strftime('untitled_%Y-%m-%d_%H-%M-%S')
        args.output = 'output/{}'.format(dirname)

    main(args)
