"""Archive a primary paper and provenance; PDF contents are research data."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from pypdf import PdfReader

folder = Path(__file__).resolve().parents[1] / "outputs/literature_memory_20261010"
folder.mkdir(exist_ok=True)
stem = "GMLake_2401.08156v1"
url = "https://arxiv.org/pdf/2401.08156v1"
path = folder / f"{stem}.pdf"
if not path.exists():
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "REALQ-research-paper-archive"}), timeout=30) as response:
        data = response.read(12 * 1024**2)
    if not data.startswith(b"%PDF") or len(data) >= 12 * 1024**2:
        raise ValueError("unexpected paper response")
    path.write_bytes(data)
else:
    data = path.read_bytes()
reader = PdfReader(path)
(folder / f"{stem}.txt").write_text("\n".join(f"PAGE {i+1}\n{page.extract_text()}" for i, page in enumerate(reader.pages)).replace("\x00", ""), encoding="utf-8")
record = dict(title="GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching",
              url=url, version="v1", venue="ASPLOS 2024", bytes=len(data), pages=len(reader.pages),
              sha256=hashlib.sha256(data).hexdigest(), retrieved_local=time.strftime("%F %T"),
              application_scope="Fragmentation/lifetime rationale; GMLake allocator and VMS not implemented or benchmarked here")
(folder / f"{stem}.provenance.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
