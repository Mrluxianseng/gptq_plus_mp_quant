"""Download public references only; never execute downloaded material."""
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path('outputs/overnight_20260927_0143/references')
root.mkdir(exist_ok=True)
papers = {
    'QEP_2504.09629v3.pdf': 'https://arxiv.org/pdf/2504.09629v3',
    'YAQA_2505.22988v2.pdf': 'https://arxiv.org/pdf/2505.22988v2',
    'AutoFreeze_2102.01386.pdf': 'https://arxiv.org/pdf/2102.01386',
}
manifest = []
for name, url in papers.items():
    path = root/name
    try:
        if not path.exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'REALQ-research/1.0'})
            with urllib.request.urlopen(req, timeout=40) as response:
                data = response.read()
            if not data.startswith(b'%PDF'):
                raise ValueError('response is not a PDF')
            path.write_bytes(data)
        data = path.read_bytes()
        row = {'name': name, 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    except Exception as exc:
        row = {'name': name, 'url': url, 'error': repr(exc)}
    print(json.dumps(row), flush=True)
    manifest.append(row)
(root/'manifest.json').write_text(json.dumps(manifest, indent=2))
