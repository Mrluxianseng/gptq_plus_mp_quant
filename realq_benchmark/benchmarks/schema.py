"""Small, serialization-friendly schemas shared by benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkExample:
    task: str
    sample_id: str
    prompt: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRequest:
    example: BenchmarkExample
    sample_index: int
    rendered_prompt: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (
            self.example.task,
            self.example.sample_id,
            self.sample_index,
        )
