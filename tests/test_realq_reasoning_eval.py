from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import pytest

from realq.benchmarks import data, generation, runner, scoring
from realq.config import Config, parse_cli


class FakeTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[-1]['content']}</user><assistant>"
        )


def _write_jsonl(path: Path, rows, *, gzip_file=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_file else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fixture_data(root: Path):
    _write_jsonl(
        root / "gsm8k" / "test.jsonl",
        [{"question": "What is 40 + 2?", "answer": "work #### 42"}],
    )
    _write_jsonl(
        root / "math_500" / "test.jsonl",
        [{"unique_id": "m0", "problem": "Compute 1+1.", "answer": "2"}],
    )
    _write_jsonl(
        root / "humaneval_plus" / "HumanEvalPlus.jsonl.gz",
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def answer():\n    \"\"\"Return 42.\"\"\"\n",
                "entry_point": "answer",
            }
        ],
        gzip_file=True,
    )
    _write_jsonl(
        root / "livecodebench_lite" / "release_v6.jsonl",
        [
            {
                "question_id": "lcb0",
                "question_content": "Print 42.",
                "starter_code": "",
            }
        ],
    )


def test_config_cli_and_chat_thinking_switch():
    cfg = parse_cli(
        [
            "--reasoning_eval",
            "true",
            "--reasoning_tasks",
            "gsm8k",
            "math-500",
            "--skip_kl_ppl_eval",
            "true",
        ]
    )
    assert cfg.reasoning_eval
    assert cfg.reasoning_tasks == ["gsm8k", "math-500"]
    assert cfg.skip_kl_ppl_eval

    tokenizer = FakeTokenizer()
    rendered, applied = generation.render_prompt(
        tokenizer,
        "question",
        apply_chat_template=True,
        enable_thinking=False,
        system_prompt="system",
    )
    assert applied
    assert rendered.endswith("<assistant>")
    assert tokenizer.calls[0][1]["enable_thinking"] is False


def test_materialized_loaders_cover_all_four_tasks(tmp_path):
    _fixture_data(tmp_path)
    expected_ids = {
        "gsm8k": "0",
        "math_500": "m0",
        "humaneval_plus": "HumanEval/0",
        "livecodebench_lite": "lcb0",
    }
    for task, expected_id in expected_ids.items():
        examples, source = data.load_examples(
            task,
            data_dir=str(tmp_path),
            lcb_release="release_v6",
        )
        assert [example.sample_id for example in examples] == [expected_id]
        assert source["source"] == "materialized_jsonl"
        assert source["sha256"]


def test_evalplus_mapping_key_is_preserved_as_task_id():
    rows = data._human_eval_plus_rows(
        {
            "HumanEval/7": {
                "prompt": "def f():\n    pass\n",
                "entry_point": "f",
            }
        }
    )
    assert rows[0]["task_id"] == "HumanEval/7"


def test_resumable_generation_uses_fixed_chunks(tmp_path):
    examples = [
        data.BenchmarkExample("gsm8k", str(index), f"q{index}", "#### 1")
        for index in range(3)
    ]
    tokenizer = FakeTokenizer()
    requests = generation.make_requests(
        examples,
        tokenizer,
        num_samples=1,
        apply_chat_template=True,
        enable_thinking=True,
        system_prompt="system",
    )
    cfg = Config(
        reasoning_batch_size=2,
        reasoning_resume=True,
        reasoning_do_sample=False,
    )
    calls = []

    def fake_generate(chunk, kwargs):
        calls.append([request.example.sample_id for request in chunk])
        return ["Final answer: 1"] * len(chunk)

    destination = tmp_path / "generations.jsonl"
    first = generation.generate_resumable(
        task="gsm8k",
        requests=requests,
        destination=destination,
        cfg=cfg,
        generate=fake_generate,
    )
    second = generation.generate_resumable(
        task="gsm8k",
        requests=requests,
        destination=destination,
        cfg=cfg,
        generate=fake_generate,
    )
    assert len(first) == len(second) == 3
    assert calls == [["0", "1"], ["2"]]

    cfg.reasoning_max_new_tokens = 17
    generation.generate_resumable(
        task="gsm8k",
        requests=requests,
        destination=destination,
        cfg=cfg,
        generate=fake_generate,
    )
    assert calls == [["0", "1"], ["2"], ["0", "1"], ["2"]]


def test_end_to_end_four_task_fixture(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "results"
    _fixture_data(data_root)

    math_verify = types.ModuleType("math_verify")
    math_verify.parse = lambda value: str(value).replace("\\boxed{", "").replace("}", "")
    math_verify.verify = lambda gold, prediction: gold.strip() in prediction
    monkeypatch.setitem(sys.modules, "math_verify", math_verify)

    class FakeGenerator:
        def __init__(self, model, tokenizer, batch_size):
            pass

        def __call__(self, requests, kwargs):
            outputs = []
            for request in requests:
                task = request.example.task
                outputs.append(
                    {
                        "gsm8k": "Reasoning. Final answer: 42",
                        "math_500": "Final answer: \\\\boxed{2}",
                        "humaneval_plus": "```python\ndef answer():\n    return 42\n```",
                        "livecodebench_lite": "```python\nprint(42)\n```",
                    }[task]
                )
            return outputs

    monkeypatch.setattr(runner, "HFLMGenerator", FakeGenerator)
    cfg = Config(
        model="fixture",
        reasoning_eval=True,
        reasoning_data_dir=str(data_root),
        reasoning_output_dir=str(output_root),
        reasoning_batch_size=2,
        reasoning_do_sample=False,
        reasoning_limit=-1,
    )
    manifest = runner.run_reasoning_eval(object(), FakeTokenizer(), cfg)
    assert manifest["status"] == "completed"
    summaries = {value["task"]: value for value in manifest["results"]}
    assert summaries["gsm8k"]["pass_at_1"] == 1.0
    assert summaries["math_500"]["pass_at_1"] == 1.0
    assert summaries["humaneval_plus"]["status"] == "generated_unscored"
    assert summaries["humaneval_plus"]["official_dataset"].endswith(
        "HumanEvalPlus.jsonl.gz"
    )
    assert summaries["livecodebench_lite"]["status"] == "generated_unscored"
    assert summaries["livecodebench_lite"]["release_version"] == "release_v6"

    humaneval_rows = [
        json.loads(line)
        for line in (
            output_root / "humaneval_plus" / "evalplus_samples.jsonl"
        ).read_text().splitlines()
    ]
    assert humaneval_rows[0]["task_id"] == "HumanEval/0"
    assert "def answer" in humaneval_rows[0]["solution"]
    livecode = json.loads(
        (
            output_root
            / "livecodebench_lite"
            / "livecodebench_custom_outputs.json"
        ).read_text()
    )
    assert livecode == [{"question_id": "lcb0", "code_list": ["print(42)"]}]


def test_gsm8k_numeric_equivalence_and_code_fence_extraction():
    correct, gold, prediction = scoring.gsm8k_correct(
        "work #### 1,250",
        "Thus, Final answer: 1250.",
    )
    assert correct
    assert gold == "1,250"
    assert prediction == "1250"
    assert scoring.extract_python_code("text\n```python\nprint(1)\n```") == "print(1)"
    assert (
        scoring.extract_python_code(
            "<think>I should print one.</think>\nprint(1)"
        )
        == "print(1)"
    )


def test_official_eval_refuses_unsandboxed_execution(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("{}\n")
    from realq.benchmarks import official_eval

    with pytest.raises(RuntimeError, match="Refusing"):
        official_eval.main(
            ["--task", "humaneval_plus", "--samples", str(samples)]
        )


def test_official_lcb_command_pins_release_and_local_data(monkeypatch, tmp_path):
    samples = tmp_path / "samples.json"
    dataset = tmp_path / "release_v6.official.jsonl"
    source = tmp_path / "LiveCodeBench"
    (source / "lcb_runner").mkdir(parents=True)
    samples.write_text("[]\n")
    dataset.write_text("{}\n")
    from realq.benchmarks import official_eval

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(official_eval.subprocess, "run", fake_run)
    official_eval.main(
        [
            "--task",
            "livecodebench_lite",
            "--samples",
            str(samples),
            "--lcb-data",
            str(dataset),
            "--lcb-release",
            "release_v6",
            "--lcb-source",
            str(source),
            "--i-understand-generated-code-will-run",
        ]
    )
    command = captured["command"]
    assert "realq.benchmarks.lcb_local_eval" in command
    assert command[command.index("--release-version") + 1] == "release_v6"
    assert command[command.index("--dataset-file") + 1] == str(dataset)
    assert command[command.index("--source-dir") + 1] == str(source)
    assert captured["kwargs"]["env"]["REALQ_ALLOW_UNTRUSTED_CODE"] == "1"
