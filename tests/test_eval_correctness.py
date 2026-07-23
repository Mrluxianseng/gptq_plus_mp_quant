from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import eval_utils


class _TinyEvalModel(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=dtype))
        self.lm_head = nn.Linear(3, 5, bias=False, dtype=dtype)


def _eval_fixture(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"tiny"}\n')
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text('{"version":"1"}\n')
    model = _TinyEvalModel()
    tokenizer = SimpleNamespace(
        name_or_path=str(tokenizer_path),
        vocab_size=5,
    )
    analyzer = SimpleNamespace(model=model, tokenizer=tokenizer)
    args = SimpleNamespace(
        cache_dir=str(tmp_path / "cache"),
        model_name="tiny",
        model=str(model_path),
        eval_seq_len=4,
        rotate=False,
        rotation_seed=0,
        optimized_rotation_path=None,
    )
    loader = SimpleNamespace(input_ids=torch.tensor([[0, 1, 2, 3]]))
    return args, analyzer, loader


def test_reference_cache_keys_exact_tokens_artifacts_and_dtype(
    tmp_path,
    monkeypatch,
):
    args, analyzer, loader = _eval_fixture(tmp_path)
    calls = []

    def fake_get_logits(_args, current, current_loader, dev):
        assert current is analyzer
        assert current_loader is loader
        assert dev.type == "cuda"
        calls.append(current_loader.input_ids.clone())
        return (
            torch.full((1, 4, 3), len(calls), dtype=torch.bfloat16),
            current_loader.input_ids,
        )

    monkeypatch.setattr(eval_utils, "_get_logits", fake_get_logits)
    monkeypatch.setattr(eval_utils.dist_utils, "is_main", lambda: True)
    monkeypatch.setattr(
        eval_utils.memory_utils, "cleanup_memory", lambda **_kwargs: None
    )

    first, _ = eval_utils.get_ref_logits(
        args, analyzer, "wikitext2", loader
    )
    second, _ = eval_utils.get_ref_logits(
        args, analyzer, "wikitext2", loader
    )
    assert len(calls) == 1
    assert torch.equal(first, second)

    loader.input_ids[0, -1] = 4
    changed_tokens, _ = eval_utils.get_ref_logits(
        args, analyzer, "wikitext2", loader
    )
    assert len(calls) == 2
    assert not torch.equal(first, changed_tokens)

    config_path = Path(args.model) / "config.json"
    config_path.write_text('{"model_type":"mutated"}\n')
    changed_artifact, _ = eval_utils.get_ref_logits(
        args, analyzer, "wikitext2", loader
    )
    assert len(calls) == 3
    assert not torch.equal(changed_tokens, changed_artifact)

    cache_files = list((Path(args.cache_dir) / "ref_logits").glob("*.cache"))
    assert len(cache_files) == 3
    payload = torch.load(cache_files[0], weights_only=True)
    assert payload["metadata"]["schema_version"] == 2
    assert payload["metadata"]["model_dtype"] == "torch.bfloat16"


def test_reference_cache_atomic_write_preserves_old_file_on_failure(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "ref.cache"
    target.write_bytes(b"old-complete")
    temporary_paths = []

    def fail_save(_payload, path):
        temporary = Path(path)
        temporary_paths.append(temporary)
        temporary.write_bytes(b"partial")
        raise OSError("injected")

    monkeypatch.setattr(eval_utils.torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        eval_utils._atomic_torch_save({"x": torch.ones(1)}, str(target))

    assert target.read_bytes() == b"old-complete"
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_paper_qa_tasks_require_every_task_and_accuracy_fallback():
    available = set(eval_utils.PAPER_QA_TASKS)

    def exact_match(patterns, all_tasks):
        return [pattern for pattern in patterns if pattern in all_tasks]

    assert eval_utils._resolve_paper_qa_tasks(
        exact_match, available
    ) == list(eval_utils.PAPER_QA_TASKS)

    available.remove("social_iqa")
    with pytest.raises(RuntimeError, match="social_iqa"):
        eval_utils._resolve_paper_qa_tasks(exact_match, available)

    def ambiguous(patterns, all_tasks):
        if patterns == ["ceval-valid"]:
            return ["ceval-valid", "ceval-valid-expanded"]
        return exact_match(patterns, all_tasks)

    with pytest.raises(RuntimeError, match="exactly one"):
        eval_utils._resolve_paper_qa_tasks(
            ambiguous, set(eval_utils.PAPER_QA_TASKS)
        )

    # Presence of normalized accuracy must not eagerly access a missing raw
    # accuracy key (the old dict.get(..., result["acc,none"]) did).
    assert eval_utils._task_accuracy(
        "hellaswag", {"acc_norm,none": 0.625}
    ) == 62.5
    assert eval_utils._task_accuracy("boolq", {"acc,none": 0.75}) == 75.0
    with pytest.raises(RuntimeError, match="neither"):
        eval_utils._task_accuracy("broken", {})


def test_kl_ppl_uses_fp32_distribution_math(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the production evaluator")

    torch.manual_seed(0)
    model = _TinyEvalModel()
    analyzer = SimpleNamespace(model=model)
    original_head = nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    original_head.load_state_dict(model.lm_head.state_dict())
    student_hidden = torch.tensor(
        [[[0.25, -0.5, 0.75], [0.1, 0.2, -0.3], [1.0, -0.2, 0.4]]],
        dtype=torch.bfloat16,
    )
    reference_hidden = student_hidden.clone()
    reference_hidden[0, 1, 0] += torch.tensor(0.03125, dtype=torch.bfloat16)
    input_ids = torch.tensor([[0, 1, 2]])

    monkeypatch.setattr(
        eval_utils,
        "_get_logits",
        lambda *_args, **_kwargs: (student_hidden, input_ids),
    )
    monkeypatch.setattr(
        eval_utils.memory_utils, "cleanup_memory", lambda **_kwargs: None
    )
    args = SimpleNamespace(kl_topk=-1)

    ppl, kl = eval_utils._kl_ppl_eval(
        args,
        analyzer,
        original_head,
        object(),
        reference_hidden,
    )
    assert ppl > 0
    assert kl >= 0

    device = torch.device("cuda")
    with torch.no_grad():
        student_logits = model.lm_head.to(device)(
            student_hidden[0].to(device)
        ).float()
        teacher_logits = original_head.to(device)(
            reference_hidden[0].to(device)
        ).float()
        expected = eval_utils.tokenwise_kl_from_logits(
            student_logits, teacher_logits
        ).mean()
        expected_ppl = torch.exp(
            F.cross_entropy(
                student_logits[:-1],
                input_ids[0, 1:].to(device),
                reduction="mean",
            )
        )
    assert kl == pytest.approx(float(expected.cpu()), abs=1e-8)
    assert ppl == pytest.approx(float(expected_ppl.cpu()), rel=1e-7)
