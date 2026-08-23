from pathlib import Path

from experiments.realq_backend_label_diagnostic_20260822 import compare_adapter


def test_path_adapter_accepts_str_and_path(tmp_path):
    source = tmp_path / "value"
    source.write_bytes(b"fixed")
    calls = []

    def original(path):
        calls.append(path)
        return path.read_text(encoding="utf-8")

    adapted = compare_adapter._path_adapter(original)
    assert adapted(str(source)) == "fixed"
    assert adapted(source) == "fixed"
    assert calls == [Path(source), Path(source)]
