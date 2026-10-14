"""Opt-in rank0 recovery receipt before KL/QA, with unchanged model bytes.

The checkpoint is a new quantized result artifact, not a copy of base models
or data. Saving/loading verification time is part of this recovery run's wall
time and must not be presented as a stable algorithm speed measurement.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from tools.qa_tensor_fingerprint_stream import update_tensor_bytes


def state_digest(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(name.encode())
        result.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        update_tensor_bytes(result, tensor)
    return dict(state_tensors=len(state), state_sha256=result.hexdigest())


def install(eval_module, destination):
    destination = Path(destination)
    receipt = destination.with_suffix(destination.suffix + '.receipt.json')
    if destination.exists() or receipt.exists():
        raise FileExistsError('Preserve existing quantized recovery checkpoint')
    original = eval_module.kl_ppl_eval
    saved = False

    def evaluate(args, analyzer, *values, **options):
        nonlocal saved
        if int(os.environ.get('RANK', '0')) == 0 and not saved:
            from utils import checkpoint_utils
            model = analyzer.model
            before = state_digest(model.state_dict())
            identities = {name: id(p) for name, p in model.named_parameters(remove_duplicate=False)}
            started = time.monotonic()
            checkpoint_utils.save_quantized_checkpoint(destination, model, args, analyzer.tokenizer)
            loaded = checkpoint_utils.load_quantized_checkpoint(destination, map_location='cpu')
            stored = state_digest(loaded['model'])
            del loaded
            after = state_digest(model.state_dict())
            parameter_identity = identities == {name: id(p) for name, p in model.named_parameters(remove_duplicate=False)}
            record = dict(before=before, saved=stored, after=after,
                parameter_identity=parameter_identity,
                completed=before == stored == after and parameter_identity,
                wall_seconds=time.monotonic()-started,
                checkpoint=str(destination), checkpoint_bytes=destination.stat().st_size,
                scope='Checkpoint byte/reload verification only; complete formal QA acceptance remains pending')
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
            if not record['completed']:
                raise RuntimeError('Recovery checkpoint full state/identity differs')
            saved = True
            print('QA_RECOVERY_CHECKPOINT ' + json.dumps(record), flush=True)
        return original(args, analyzer, *values, **options)

    eval_module.kl_ppl_eval = evaluate
    return original
