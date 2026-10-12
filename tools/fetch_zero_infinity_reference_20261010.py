"""Archive the primary paper and text, without modifying research outcomes."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import urllib.request

root = Path(__file__).resolve().parents[1] / "outputs/literature_memory_20261010"
name = "ZeRO_Infinity_2104.07857v1"
path = root / f"{name}.pdf"
url = "https://arxiv.org/pdf/2104.07857v1"
if not path.exists():
    with urllib.request.urlopen(url, timeout=45) as response:
        value = response.read()
    if not value.startswith(b"%PDF"):
        raise ValueError("download is not a PDF")
    path.write_bytes(value)
value = path.read_bytes()
record = {"file": path.name, "url": url, "bytes": len(value),
          "sha256": hashlib.sha256(value).hexdigest(), "retrieved_local": "2026-10-10"}
(root / f"{name}.provenance.json").write_text(json.dumps(record, indent=2))
if shutil.which("pdftotext"):
    subprocess.run(["pdftotext", "-layout", str(path), str(root / f"{name}.txt")], check=True)
else:
    from pypdf import PdfReader
    reader = PdfReader(path)
    (root / f"{name}.txt").write_text("\n".join(f"PAGE {i+1}\n" + page.extract_text() for i, page in enumerate(reader.pages)), encoding="utf-8")
print(json.dumps(record, indent=2))
