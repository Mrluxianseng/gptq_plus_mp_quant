"""Compare the saved layer-2 FP trace with a separate plain HF model forward."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import model_utils

ROOT = Path(__file__).resolve().parents[1]
MODEL = '/mnt/d/llamaModels/Qwen3-0.6B'
OUT = ROOT / 'outputs/sequence_fisher_pilot_20260929'
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_grad_enabled(False)
analyzer = model_utils.ModelAnalyzer(MODEL, 256)
model = analyzer.model.eval().to('cuda')
records = {}
for seed in (42, 44):
    path = ROOT / f'cache/tokens/Qwen3-0.6B-wikitext2_s32_blk256_seed{seed}.pt'
    tokens = torch.load(path, map_location='cpu', weights_only=False)
    reference = torch.load(OUT / f'seed{seed}_off/layer_2_outputs.pt', map_location='cpu', weights_only=True)['target']
    rows = []
    for idx in range(4):
        output = model(input_ids=tokens[idx].unsqueeze(0).to('cuda'), output_hidden_states=True,
                       use_cache=False, return_dict=True)
        # hidden_states[0] is embedding output; [3] follows decoder block 2.
        independent = output.hidden_states[3][0].to(torch.bfloat16).cpu()
        saved = reference[idx]
        diff = independent.float() - saved.float()
        rows.append(dict(sample=idx, first_token=diff[:, 0].abs().max().item(),
                         overall_max=diff.abs().max().item(),
                         overall_rmse=diff.square().mean().sqrt().item(),
                         independent_first_rms=independent[0].float().square().mean().sqrt().item(),
                         saved_first_rms=saved[0].float().square().mean().sqrt().item()))
        del output
    records[str(seed)] = rows
    print(seed, json.dumps(rows), flush=True)
result = dict(model=MODEL, comparison='plain HF output_hidden_states[3] vs static FP layer-2 trace; BF16 comparison',
              seeds=records)
(OUT / 'fp_reference_audit.json').write_text(json.dumps(result, indent=2))
