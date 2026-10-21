"""Chat rendering, batched generation, and resumable JSONL persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from realq.benchmarks.schema import (
    BenchmarkExample,
    GenerationRequest,
)


def render_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    apply_chat_template: bool,
    enable_thinking: bool,
    system_prompt: str,
) -> tuple[str, bool]:
    """Render one user prompt, including Qwen3's thinking switch when present."""

    chat_template = getattr(tokenizer, "chat_template", None)
    if not apply_chat_template or not chat_template:
        prefix = f"{system_prompt.strip()}\n\n" if system_prompt.strip() else ""
        return prefix + prompt, False

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Non-Qwen tokenizers may expose a strict method signature. They still
        # receive their native chat template, just without a thinking switch.
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return str(rendered), True


def make_requests(
    examples: Iterable[BenchmarkExample],
    tokenizer: Any,
    *,
    num_samples: int,
    apply_chat_template: bool,
    enable_thinking: bool,
    system_prompt: str,
) -> list[GenerationRequest]:
    requests: list[GenerationRequest] = []
    for example in examples:
        rendered, _ = render_prompt(
            tokenizer,
            example.prompt,
            apply_chat_template=apply_chat_template,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        for sample_index in range(num_samples):
            requests.append(
                GenerationRequest(
                    example=example,
                    sample_index=sample_index,
                    rendered_prompt=rendered,
                )
            )
    return requests


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _request_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["task"]),
        str(record["sample_id"]),
        int(record["sample_index"]),
    )


def read_generation_records(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Read a possibly resumed JSONL file; the last complete duplicate wins."""

    if not path.is_file():
        return {}
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = _request_key(record)
            except Exception as exc:
                logging.warning(
                    "Ignoring incomplete generation record %s:%d: %s",
                    path,
                    line_number,
                    exc,
                )
                continue
            records[key] = record
    return records


def _chunked(values: list[GenerationRequest], size: int):
    for start in range(0, len(values), size):
        yield start // size, values[start : start + size]


def _chunk_seed(base_seed: int, task: str, chunk_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{task}:{chunk_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generation_kwargs(cfg: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_gen_toks": cfg.reasoning_max_new_tokens,
        "until": ["<|im_end|>", "<|endoftext|>"],
        "do_sample": bool(cfg.reasoning_do_sample),
    }
    if cfg.reasoning_do_sample:
        kwargs.update(
            {
                "temperature": float(cfg.reasoning_temperature),
                "top_p": float(cfg.reasoning_top_p),
                "top_k": int(cfg.reasoning_top_k),
            }
        )
    return kwargs


def generation_config_sha256(
    task: str,
    cfg: Any,
    kwargs: dict[str, Any],
) -> str:
    """Fingerprint every setting that can change a persisted completion."""

    payload = {
        "task": task,
        "model": str(cfg.model),
        "load_qmodel_path": cfg.load_qmodel_path,
        "generation_kwargs": kwargs,
        "seed": int(cfg.reasoning_seed),
        "batch_size": int(cfg.reasoning_batch_size),
        "num_samples": int(cfg.reasoning_num_samples),
        "protocol": str(cfg.reasoning_protocol),
        "quantization": {
            name: getattr(cfg, name)
            for name in (
                "w_bits",
                "w_groupsize",
                "a_bits",
                "a_groupsize",
                "k_bits",
                "k_groupsize",
                "v_bits",
                "v_groupsize",
                "rotate",
            )
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HFLMGenerator:
    """Thin adapter around lm-eval's tested preloaded-Transformers backend."""

    def __init__(self, model: Any, tokenizer: Any, batch_size: int):
        try:
            from lm_eval.models.huggingface import HFLM
        except ImportError as exc:
            raise RuntimeError(
                "Reasoning evaluation requires `lm-eval==0.4.4`, matching "
                "the existing RealQ QA evaluator."
            ) from exc
        self._model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    def __call__(
        self,
        requests: list[GenerationRequest],
        kwargs: dict[str, Any],
    ) -> list[str]:
        from lm_eval.api.instance import Instance

        instances = [
            Instance(
                request_type="generate_until",
                doc={},
                arguments=(request.rendered_prompt, kwargs),
                idx=index,
                metadata=(request.example.task, index, 1),
            )
            for index, request in enumerate(requests)
        ]
        return self._model.generate_until(instances)


def generate_resumable(
    *,
    task: str,
    requests: list[GenerationRequest],
    destination: Path,
    cfg: Any,
    generate: Callable[[list[GenerationRequest], dict[str, Any]], list[str]],
) -> list[dict[str, Any]]:
    """Generate fixed chunks so sampled runs remain reproducible after resume."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = read_generation_records(destination) if cfg.reasoning_resume else {}
    kwargs = generation_kwargs(cfg)
    config_sha256 = generation_config_sha256(task, cfg, kwargs)

    with destination.open("a", encoding="utf-8") as handle:
        for chunk_index, chunk in _chunked(requests, cfg.reasoning_batch_size):
            reusable = []
            for request in chunk:
                record = existing.get(request.key)
                reusable.append(
                    record is not None
                    and record.get("prompt_sha256")
                    == prompt_sha256(request.rendered_prompt)
                    and record.get("generation_config_sha256") == config_sha256
                )
            if all(reusable):
                continue

            seed = _chunk_seed(cfg.reasoning_seed, task, chunk_index)
            _seed_everything(seed)
            outputs = generate(chunk, kwargs)
            if len(outputs) != len(chunk):
                raise RuntimeError(
                    f"Generation backend returned {len(outputs)} outputs for "
                    f"{len(chunk)} requests."
                )
            for request_index, (request, output) in enumerate(zip(chunk, outputs)):
                if (
                    existing.get(request.key) is not None
                    and reusable[request_index]
                ):
                    continue
                record = {
                    "schema_version": 1,
                    "task": task,
                    "sample_id": request.example.sample_id,
                    "sample_index": request.sample_index,
                    "prompt_sha256": prompt_sha256(request.rendered_prompt),
                    "generation_config_sha256": config_sha256,
                    "chunk_index": chunk_index,
                    "chunk_seed": seed,
                    "output": str(output),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing[request.key] = record
            handle.flush()
            os.fsync(handle.fileno())

    return [
        existing[request.key]
        for request in requests
        if request.key in existing
        and existing[request.key].get("prompt_sha256")
        == prompt_sha256(request.rendered_prompt)
    ]
