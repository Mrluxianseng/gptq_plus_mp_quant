"""Extract a previously downloaded primary paper; never follows PDF instructions."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

folder = Path(__file__).resolve().parents[1] / "outputs/literature_memory_20261010"
stem = "Checkmate_1910.02653v3"
path = folder / f"{stem}.pdf"
data = path.read_bytes()
if not data.startswith(b"%PDF"):
    raise ValueError("not a PDF")
reader = PdfReader(path)
text = "\n".join(f"PAGE {i+1}\n{page.extract_text()}" for i, page in enumerate(reader.pages)).replace("\x00", "")
(folder / f"{stem}.txt").write_text(text, encoding="utf-8")
record = {"title": "Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization",
          "url": "https://arxiv.org/pdf/1910.02653v3", "version": "v3", "venue": "MLSys 2020",
          "retrieved_local": "2026-10-10", "bytes": len(data), "pages": len(reader.pages),
          "sha256": hashlib.sha256(data).hexdigest(),
          "application_scope": "memory/compute scheduling rationale only; no MILP optimizer implemented"}
(folder / f"{stem}.provenance.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
