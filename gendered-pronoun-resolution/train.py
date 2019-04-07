import sys
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.random import RandomState
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import Config
from data import load_train_val_examples, GAPDataset
from model import Model
from optim import Adam


logger = logging.getLogger(__name__)


def train(config: Config, device: torch.device, output_dir: Path) -> float:
    # Save the configuration file
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    config.save(output_dir / 'config.json')

    # Initialize random number generators
    random_state = RandomState(config.random_seed)
    seed = lambda: int.from_bytes(random_state.bytes(4), byteorder=sys.byteorder)
    random.seed(seed())
    np.random.seed(seed())
    torch.manual_seed(seed())

    # Load training and validation data.
    train_examples, val_examples = load_train_val_examples(config.random_seed)
    logger.info('Using %d training examples', len(train_examples))
    logger.info('Using %d validation examples', len(val_examples))

    train_dataset = GAPDataset(train_examples, flip_prob=0.5)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        collate_fn=lambda x: x, shuffle=True, drop_last=True)

    val_dataset = GAPDataset(val_examples)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        collate_fn=lambda x: x)

    # Model and optimizer initialization
    model = Model(config, device).to(device)
    total_steps = (len(train_dataset) // config.batch_size) * config.num_epochs
    optimizer = Adam(model.named_parameters(),
                     lr=config.lr, lr_warmup=config.lr_warmup,
                     beta1=config.beta1, beta2=config.beta2, eps=config.eps,
                     weight_decay=config.weight_decay, max_grad_norm=5.0,
                     lr_cooldown='linear', total_steps=total_steps)

    logger.info('Training for %d epochs (%d updates)',
                config.num_epochs, total_steps)
    tensorboard = SummaryWriter(str(output_dir / 'tensorboard'))
    progress_bar = lambda x: tqdm(x, unit='batch', ncols=80, leave=False)
    best_val_loss = np.inf

    for epoch in range(1, config.num_epochs + 1):
        t_start = datetime.now()
        train_loss, val_loss = 0, 0

        # Training
        model.train()
        for batch in progress_bar(train_loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(batch, reduction='mean')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Evaluation
        model.eval()
        with torch.no_grad():
            for batch in progress_bar(val_loader):
                loss = model.calculate_loss(batch, reduction='sum')
                val_loss += loss.item()

        t_elapsed = datetime.now() - t_start
        train_loss /= len(train_loader)
        val_loss /= len(val_dataset)
        tensorboard.add_scalars('loss',
            {'train': train_loss, 'val': val_loss}, epoch)
        logger.info('Epoch %d/%d (train = %g, val = %g, time = %s)',
            epoch, config.num_epochs, train_loss, val_loss, t_elapsed)

        # Save the model each time the validation GAP loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = str(output_dir / 'model.pickle')
            logger.info('Saving model to %s', model_path)
            torch.save(model.state_dict(), model_path)

    with (output_dir / 'val_loss.txt').open('w') as f:
        f.write('%g\n' % best_val_loss)

    return best_val_loss


def main(args: argparse.Namespace) -> None:
    # Load arguments
    config_file, output_dir = map(
        lambda s: Path(s).resolve(), [args.config, args.output])
    config = Config.load(config_file)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    # Logging initialization
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(output_dir / 'train.log', mode='w'),
            logging.StreamHandler(),
        ])
    logging.getLogger('pytorch_pretrained_bert').setLevel(logging.WARNING)

    train(config, device, output_dir)
    logger.info('Output written to %s', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='JSON_FILE',
        help='Model configuration file.', required=True)
    parser.add_argument('--device', metavar='DEVICE', default='cuda',
        help='PyTorch device to use for computation.')
    parser.add_argument('--output', metavar='OUTPUT_DIR',
        help='Output directory. A timestamped folder '
             'will be created if not specified.')
    args = parser.parse_args()
    if args.output is None:
        dirname = datetime.now().strftime('untitled_%Y-%m-%d_%H-%M-%S')
        args.output = 'output/{}'.format(dirname)
    main(args)
