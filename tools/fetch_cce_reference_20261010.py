"""Archive the versioned CCE primary paper and provenance for the thesis."""
import hashlib
import json
from pathlib import Path
import urllib.request

from pypdf import PdfReader

root = Path(__file__).resolve().parents[1] / "outputs/literature_memory_20261010"
root.mkdir(parents=True, exist_ok=True)
name = "Cut_Your_Losses_2411.09009v2"
path = root / f"{name}.pdf"
url = "https://arxiv.org/pdf/2411.09009v2"
if not path.exists():
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("download is not a PDF")
    path.write_bytes(data)
data = path.read_bytes()
if not data.startswith(b"%PDF"):
    raise ValueError("archived file is not a PDF")
record = {"file": path.name, "url": url, "bytes": len(data),
          "sha256": hashlib.sha256(data).hexdigest(), "retrieved_local": "2026-10-10",
          "title": "Cut Your Losses in Large-Vocabulary Language Models",
          "scope": "Vocabulary loss working-set rationale; current implementation is token chunking, not fused CCE."}
(root / f"{name}.provenance.json").write_text(json.dumps(record, indent=2))
reader = PdfReader(path)
(root / f"{name}.txt").write_text("\n".join(f"PAGE {i+1}\n" + page.extract_text() for i, page in enumerate(reader.pages)).replace("\x00", ""), encoding="utf-8")
print(json.dumps(record, indent=2))
