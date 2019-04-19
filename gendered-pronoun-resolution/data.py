import csv
import logging
import random
import re
from collections import OrderedDict
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, List

import attr
from torch.utils.data import Dataset
from syntok import segmenter
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


DATA_DIR = Path(__file__).parent / 'data'


PRONOUNS = {'she', 'her', 'hers', 'he', 'him', 'his'}

PRONOUNS_GENDER = {'she': 'F', 'her': 'F', 'hers': 'F', 'he': 'M', 'him': 'M', 'his': 'M'}


class GAPLabel(IntEnum):
    A, B, NEITHER = 0, 1, 2


@attr.s(auto_attribs=True)
class GAPExample(object):
    id: str
    url: str
    tokens: List[str]
    pronoun_index: int
    a_start: int
    a_end: int  # exclusive
    b_start: int
    b_end: int  # exclusive
    label: Optional[GAPLabel]


def load_train_val_examples(random_seed, train_size=0.9) -> Tuple[List[GAPExample], List[GAPExample]]:
    examples = []
    for tsv_file in ('gap-development.tsv', 'gap-validation.tsv', 'gap-test.tsv'):
        examples.extend(_load_gap(DATA_DIR / tsv_file))
    examples_gender = [PRONOUNS_GENDER[e.tokens[e.pronoun_index].lower()] for e in examples]
    train_examples, val_examples = train_test_split(
        examples, random_state=random_seed, train_size=train_size,
        shuffle=True, stratify=examples_gender)
    return train_examples, val_examples


def load_test_examples(tsv_path: Path = DATA_DIR / 'test_stage_2.tsv') -> List[GAPExample]:
    examples = _load_gap(tsv_path)
    return examples


def _load_gap(tsv_path: Path) -> List[GAPExample]:
    examples: List[GAPExample] = []
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            examples.append(_create_example(row))
    logger.info('Loaded %d examples from %s', len(examples), tsv_path)
    return examples


def _create_example(row: OrderedDict):
    label = None
    a_coref, b_coref = map(lambda x: row.get(f'{x}-coref', '').upper(), 'AB')
    if a_coref == 'TRUE' and b_coref == 'FALSE':
        label = GAPLabel.A
    elif b_coref == 'TRUE' and a_coref == 'FALSE':
        label = GAPLabel.B
    elif a_coref == 'FALSE' and b_coref == 'FALSE':
        label = GAPLabel.NEITHER

    tokens = _word_tokenizer(row['Text'])

    pronoun_index = _char_to_token_offset(
        row['Text'], row['Pronoun'], int(row['Pronoun-offset']), tokens)
    assert tokens[pronoun_index].lower() in PRONOUNS

    a_start = _char_to_token_offset(
        row['Text'], row['A'], int(row['A-offset']), tokens)
    a_end = a_start + len(_word_tokenizer(row['A']))  # exclusive

    b_start = _char_to_token_offset(
        row['Text'], row['B'], int(row['B-offset']), tokens)
    b_end = b_start + len(_word_tokenizer(row['B']))  # exclusive

    example = GAPExample(
        id=row['ID'],
        url=row['URL'],
        tokens=tokens,
        pronoun_index=pronoun_index,
        a_start=a_start,
        a_end=a_end,
        b_start=b_start,
        b_end=b_end,
        label=label)
    return example


def _word_tokenizer(text: str) -> List[str]:
    tokens: List[str] = []
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            for token in sentence:
                # Split tokens on additional characters not handled by syntok
                token_value = token.value
                for c in ('/', r'\*', "'", r'\.', '--', ':'):
                    token_value = re.sub(rf'({c})', r' \1 ', token_value)
                tokens.extend(token_value.split())
    return tokens


def _char_to_token_offset(
        text: str,
        mention: str,
        char_offset: int,
        text_tokens: List[str]) -> int:
    char_index = token_index = 0
    while char_index < char_offset:
        if text[char_index:].startswith(text_tokens[token_index]):
            char_index += len(text_tokens[token_index])
            token_index += 1
        else:
            char_index += 1  # whitespace
    return token_index


class GAPDataset(Dataset):

    def __init__(self, examples: List[GAPExample], flip_prob: float = 0.0) -> None:
        super().__init__()
        self._examples = examples
        assert 0.0 <= flip_prob <= 1.0
        self._flip_prob = flip_prob

    def __getitem__(self, index: int) -> GAPExample:
        example = self._examples[index]
        if (self._flip_prob == 1.0 or
            (self._flip_prob > 0.0 and
             random.random() <= self._flip_prob)):
            example = self._flip_example(example)
        return example

    def __len__(self) -> int:
        return len(self._examples)

    def _flip_example(self, example: GAPExample) -> GAPExample:
        new_label = example.label
        if example.label == GAPLabel.A:
            new_label = GAPLabel.B
        elif example.label == GAPLabel.B:
            new_label = GAPLabel.A
        new_example = GAPExample(
            id=example.id,
            url=example.url,
            tokens=example.tokens,
            pronoun_index=example.pronoun_index,
            a_start=example.b_start,
            a_end=example.b_end,
            b_start=example.a_start,
            b_end=example.a_end,
            label=new_label)
        return new_example
