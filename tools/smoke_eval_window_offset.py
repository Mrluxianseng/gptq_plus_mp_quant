"""Verify fixed-window indexing without model loading or network access."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils import data_utils


def main():
    original = data_utils._get_dataset
    data_utils._get_dataset = lambda *args: ['unused fixture text']

    class Encoding(dict):
        @property
        def input_ids(self):
            return self['input_ids']

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return Encoding(input_ids=torch.arange(64).reshape(1, 64))

    try:
        tokenizer = Tokenizer()
        first = data_utils.get_loaders(
            'wikitext2', 'test', tokenizer, seq_len=4, num_samples=0,
            max_samples=2, start_sample=0,
        )['input_ids']
        heldout = data_utils.get_loaders(
            'wikitext2', 'test', tokenizer, seq_len=4, num_samples=0,
            max_samples=2, start_sample=3,
        )['input_ids']
        assert torch.equal(first, torch.arange(8).reshape(1, 8))
        assert torch.equal(heldout, torch.arange(12, 20).reshape(1, 8))
        try:
            data_utils.get_loaders(
                'wikitext2', 'test', tokenizer, seq_len=4, num_samples=0,
                max_samples=2, start_sample=-1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError('negative evaluation window index was accepted')
        print('window offset smoke passed: windows [0,2) and [3,5) select disjoint token ranges')
    finally:
        data_utils._get_dataset = original


if __name__ == '__main__':
    main()
