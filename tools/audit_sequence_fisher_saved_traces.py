"""Per-sample view of the first four saved layer outputs from the pilot."""
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1] / 'outputs/sequence_fisher_pilot_20260929'
result = {}
for seed in (42, 44):
    result[str(seed)] = {}
    for mode in ('off', 'token', 'sequence'):
        result[str(seed)][mode] = {}
        folder = root / f'seed{seed}_{mode}'
        for layer in (1, 2, 3, 4):
            saved = torch.load(folder / f'layer_{layer}_outputs.pt', map_location='cpu', weights_only=True)
            per_token = (saved['output'].float() - saved['target'].float()).square().mean(-1)
            result[str(seed)][mode][str(layer)] = {
                'sample_first_token_mse': [round(float(v), 3) for v in per_token[:, 0]],
                'sample_remaining_token_mse': [round(float(v), 6) for v in per_token[:, 1:].mean(-1)],
            }
(root / 'per_sample_saved_trace_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
