from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.efficientqat_compare.compat import (
    BlockForwardContext,
    assert_model_vocab_unchanged,
    call_decoder_block,
    ensure_fast_eos_tokenizer,
    extract_hidden_states,
)
from experiments.efficientqat_compare.data import (
    build_block_ap_trainloader,
    build_e2e_fixed_dataset,
    load_exact_token_cache,
    validate_exact_tokens,
)


def _tokens(count: int = 3, seq_len: int = 4) -> list[torch.Tensor]:
    return [
        torch.arange(index * seq_len, (index + 1) * seq_len, dtype=torch.int64)
        for index in range(count)
    ]


def test_load_exact_token_cache_and_block_ap_view(tmp_path):
    path = tmp_path / "tokens.pt"
    source = _tokens()
    torch.save(source, path)

    loaded = load_exact_token_cache(
        path,
        expected_samples=3,
        expected_seq_len=4,
    )
    loader = build_block_ap_trainloader(loaded)

    assert len(loader) == 3
    input_ids, labels = loader[1]
    assert input_ids.shape == (1, 4)
    assert input_ids.dtype == torch.int64
    assert torch.equal(input_ids[0], source[1])
    assert labels.tolist() == [[-100, -100, -100, 7]]


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (torch.zeros((3, 4), dtype=torch.int64), TypeError, "Python list"),
        (_tokens(count=2), ValueError, "sample count"),
        (
            [torch.zeros(4, dtype=torch.int32) for _ in range(3)],
            TypeError,
            "torch.int64",
        ),
        (
            [torch.zeros((1, 4), dtype=torch.int64) for _ in range(3)],
            ValueError,
            "shape",
        ),
    ],
)
def test_validate_exact_tokens_rejects_normalizable_but_invalid_cache(
    value, error_type, message
):
    with pytest.raises(error_type, match=message):
        validate_exact_tokens(
            value,
            expected_samples=3,
            expected_seq_len=4,
        )


def test_build_e2e_fixed_dataset_preserves_exact_rows_and_labels():
    source = _tokens()
    dataset = build_e2e_fixed_dataset(source)

    assert len(dataset) == 3
    assert dataset.column_names == ["input_ids", "labels"]
    row = dataset[2]
    assert row["input_ids"].dtype == torch.int64
    assert row["input_ids"].shape == (4,)
    assert torch.equal(row["input_ids"], source[2])
    assert torch.equal(row["labels"], source[2])


class _FakeTokenizer:
    is_fast = True
    eos_token = "<eos>"
    eos_token_id = 7

    def __init__(self, *, size: int = 16, add_on_pad: bool = False) -> None:
        self._size = size
        self._pad_token = None
        self.pad_token_id = None
        self._add_on_pad = add_on_pad

    def __len__(self) -> int:
        return self._size

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self.pad_token_id = self.eos_token_id
        if self._add_on_pad:
            self._size += 1


def test_ensure_fast_eos_tokenizer_aliases_without_resize():
    tokenizer = _FakeTokenizer()
    result = ensure_fast_eos_tokenizer(tokenizer, expected_vocab_size=16)

    assert result is tokenizer
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert len(tokenizer) == 16


def test_ensure_fast_eos_tokenizer_fails_if_assignment_adds_vocab():
    tokenizer = _FakeTokenizer(add_on_pad=True)
    with pytest.raises(RuntimeError, match="vocabulary size"):
        ensure_fast_eos_tokenizer(tokenizer, expected_vocab_size=16)


class _FakeEmbedding:
    def __init__(self, rows: int) -> None:
        self.weight = torch.empty((rows, 2))


class _FakeModel:
    def __init__(self, rows: int, *, tied: bool = False) -> None:
        self.config = SimpleNamespace(
            vocab_size=rows,
            pad_token_id=None,
            tie_word_embeddings=tied,
        )
        self._input = _FakeEmbedding(rows)
        self._output = self._input if tied else _FakeEmbedding(rows)

    def get_input_embeddings(self):
        return self._input

    def get_output_embeddings(self):
        return self._output


def test_assert_model_vocab_unchanged_sets_only_config_pad_id():
    tokenizer = ensure_fast_eos_tokenizer(
        _FakeTokenizer(size=16),
        expected_vocab_size=16,
    )
    model = _FakeModel(16)

    assert_model_vocab_unchanged(
        model,
        tokenizer,
        expected_vocab_size=16,
    )

    assert model.config.vocab_size == 16
    assert model.config.pad_token_id == tokenizer.eos_token_id
    assert model.get_input_embeddings().weight.shape[0] == 16
    assert model.get_output_embeddings().weight.shape[0] == 16


def test_assert_model_vocab_unchanged_accepts_frozen_padded_model_vocab():
    tokenizer = ensure_fast_eos_tokenizer(
        _FakeTokenizer(size=15),
        expected_vocab_size=15,
    )
    model = _FakeModel(16)

    assert_model_vocab_unchanged(
        model,
        tokenizer,
        expected_vocab_size=16,
        expected_tokenizer_vocab_size=15,
    )

    assert model.config.vocab_size == 16
    assert model.config.pad_token_id == tokenizer.eos_token_id
    assert len(tokenizer) == 15
    assert model.get_input_embeddings().weight.shape[0] == 16
    assert model.get_output_embeddings().weight.shape[0] == 16


def test_assert_model_vocab_unchanged_requires_declared_tie():
    tokenizer = ensure_fast_eos_tokenizer(
        _FakeTokenizer(size=16),
        expected_vocab_size=16,
    )
    model = _FakeModel(16)
    model.config.tie_word_embeddings = True

    with pytest.raises(RuntimeError, match="no longer shares"):
        assert_model_vocab_unchanged(
            model,
            tokenizer,
            expected_vocab_size=16,
        )

    tied_model = _FakeModel(16, tied=True)
    assert_model_vocab_unchanged(
        tied_model,
        tokenizer,
        expected_vocab_size=16,
    )


class _ModernTensorBlock:
    def __init__(self) -> None:
        self.seen = None

    def __call__(
        self,
        hidden_states,
        *,
        attention_mask,
        position_ids,
        cache_position,
        position_embeddings,
        use_cache,
    ):
        self.seen = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "use_cache": use_cache,
        }
        return hidden_states + 1


def _modern_context(mask_batch: int = 1) -> BlockForwardContext:
    return BlockForwardContext.capture(
        {
            "attention_mask": torch.zeros((mask_batch, 1, 4, 4)),
            "position_ids": torch.arange(4).unsqueeze(0),
            "cache_position": torch.arange(4),
            "position_embeddings": (
                torch.ones((1, 4, 2)),
                torch.zeros((1, 4, 2)),
            ),
        },
        to_cpu=True,
    )


def test_modern_block_context_preserves_batch_and_tensor_output():
    hidden = torch.zeros((2, 4, 8))
    block = _ModernTensorBlock()

    result = call_decoder_block(block, hidden, _modern_context(mask_batch=1))

    assert result.shape == hidden.shape
    assert torch.equal(result, hidden + 1)
    assert block.seen["attention_mask"].shape == (2, 1, 4, 4)
    assert block.seen["use_cache"] is False
    assert len(block.seen["position_embeddings"]) == 2


def test_modern_block_context_does_not_repeat_existing_batch_mask():
    hidden = torch.zeros((2, 4, 8))
    kwargs = _modern_context(mask_batch=2).for_hidden_states(hidden)
    assert kwargs["attention_mask"].shape == (2, 1, 4, 4)


def test_block_context_requires_position_embeddings():
    with pytest.raises(KeyError, match="position_embeddings"):
        BlockForwardContext.capture(
            {
                "attention_mask": torch.zeros((1, 1, 4, 4)),
                "position_ids": torch.arange(4).unsqueeze(0),
            }
        )


def test_extract_hidden_states_supports_legacy_tuple_without_indexing_tensor():
    hidden = torch.randn((2, 4, 8))
    assert extract_hidden_states(hidden) is hidden
    assert extract_hidden_states((hidden, "ignored")) is hidden
