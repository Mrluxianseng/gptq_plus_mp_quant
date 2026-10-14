"""Prepared QA placement bridge; no formal launcher or validation admission.

Install before ptq calls distribute_model. Control delegates unchanged; candidate
uses the tested single-device layerwise placement instead of accelerate hooks.
The caller still owes complete QA numerical and memory telemetry validation.
"""
from pathlib import Path
import hashlib
import inspect
import json
import torch
from tools.qa_layerwise_weight_prototype import LayerwiseWeights
from tools.qa_score_lifetime_prototype import build_method as rows_method
from tools.qa_scoring_cache_offload_prototype import build_method as cache_method


def layout(model):
    return {name: [list(value.shape), str(value.dtype)] for name, value in model.state_dict().items()}


def validate_model(model):
    if model.config.model_type != 'qwen3' or len(model.model.layers) != 28:
        raise RuntimeError('Requires a complete 28-layer Qwen3 model')
    if model.training or model.config.use_cache is not True:
        raise RuntimeError('Requires eval mode and original enabled KV cache')
    if any(getattr(module, '_hf_hook', None) is not None for module in model.modules()):
        raise RuntimeError('Install before accelerate dispatch; do not remove active hooks')
    if any(value.is_meta for value in model.state_dict().values()):
        raise RuntimeError('Cannot place incomplete meta state')
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError('Only the validated original BF16 precision is supported')
    for module in model.modules():
        if hasattr(module, 'bits') and module.bits != 16:
            raise RuntimeError('Low-bit activation/cache runtime needs separate validation')


class PlacementBridge:
    def __init__(self, eval_module, dist_module, *, enabled, destination,
                 device='cuda:0', mark=lambda phase: None, capture=lambda index, output: None):
        if getattr(eval_module, '_qa_layerwise_bridge_installed', False):
            raise RuntimeError('Placement bridge already installed')
        if getattr(eval_module, '_qa_scoring_lifetime_installed', False):
            raise RuntimeError('Do not stack placement and existing scoring-lifetime installers')
        self.eval_module = eval_module
        self.dist_module = dist_module
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.destination = Path(destination)
        self.mark = mark
        self.capture = capture
        self.original_eval = eval_module.qa_eval
        self.original_distribute = dist_module.distribute_model
        self.pending_model_id = None
        self.placement = None
        self.path = self.destination/'qa_layerwise_bridge_rank0.json'
        if self.path.exists():
            raise RuntimeError('QA bridge evidence exists; use a fresh directory')
        eval_module.qa_eval = self.qa_eval
        dist_module.distribute_model = self.distribute_model
        eval_module._qa_layerwise_bridge_installed = True

    def distribute_model(self, model):
        if self.pending_model_id is not None or self.path.exists():
            raise RuntimeError('QA placement already pending or completed')
        if self.enabled:
            validate_model(model)
            before = layout(model)
            model.cpu()
            if layout(model) != before:
                raise RuntimeError('Placement changed complete state layout')
            self.placement = dict(mode='single-device-layerwise', before_layout=before)
            result = None
        else:
            # Original device-map policy remains entirely intact in control.
            result = self.original_distribute(model)
            self.placement = dict(mode='original-distribute-model', before_layout=layout(model))
        self.pending_model_id = id(model)
        return result

    def qa_eval(self, model, tokenizer, *args, **kwargs):
        if self.pending_model_id != id(model) or self.path.exists():
            raise RuntimeError('QA requires this model to complete the preceding placement')
        from lm_eval.models.huggingface import HFLM
        original_score, original_call = HFLM._loglikelihood_tokens, HFLM._model_call
        ticket = None
        primary_error = None
        record = dict(schema='qa-layerwise-placement-bridge-v1', enabled=self.enabled,
            device=str(self.device), placement=self.placement,
            cache_offload_requested=self.enabled,
            cache_offload_cuda_effective=self.enabled and self.device.type == 'cuda',
            scope='prepared placement bridge; not a complete numerical or memory acceptance report',
            likelihood_original_sha256=hashlib.sha256(inspect.getsource(original_score).encode()).hexdigest(),
            model_call_original_sha256=hashlib.sha256(inspect.getsource(original_call).encode()).hexdigest())
        try:
            if self.enabled:
                validate_model(model)
                ticket = LayerwiseWeights(model, enabled=True, device=self.device,
                    mark=self.mark, capture=self.capture)
                HFLM._loglikelihood_tokens = rows_method(original_score)
                # CPU fixtures construct the same cache class; CUDA offloading
                # remains a real-GPU-only validation obligation.
                HFLM._model_call = cache_method(original_call, offloading=self.device.type == 'cuda')
            record['config_use_cache_before'] = model.config.use_cache
            result = self.original_eval(model, tokenizer, *args, **kwargs)
            if not isinstance(result, dict) or not result:
                raise RuntimeError('Evaluator must return a nonempty primitive metrics dictionary')
            try:
                json.dumps(result)
            except (TypeError, ValueError) as exc:
                raise RuntimeError('Evaluator metrics must be JSON primitives') from exc
            record.update(completed=True, metrics=result)
            return result
        except BaseException as exc:
            primary_error = exc
            record.update(completed=False, error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            HFLM._loglikelihood_tokens, HFLM._model_call = original_score, original_call
            cleanup_error = None
            try:
                if ticket is not None:
                    try:
                        ticket.close()
                    except BaseException as exc:
                        cleanup_error = exc
                        record.update(completed=False, cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
                record.update(weight_records=[] if ticket is None else ticket.records,
                    parameter_identity_preserved=ticket is not None and cleanup_error is None if self.enabled else None,
                    after_layout=layout(model), config_use_cache_after=model.config.use_cache)
            finally:
                self.pending_model_id = None
                self.destination.mkdir(parents=True, exist_ok=True)
                with self.path.open('x') as stream:
                    json.dump(record, stream, indent=2)
            if cleanup_error is not None and primary_error is None:
                raise cleanup_error

    def restore(self):
        if self.pending_model_id is not None:
            raise RuntimeError('Pending QA placement must be consumed before restoration')
        self.eval_module.qa_eval = self.original_eval
        self.dist_module.distribute_model = self.original_distribute
        self.eval_module._qa_layerwise_bridge_installed = False
