import logging
import os
import copy
import hashlib
import json
import tempfile
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import memory_utils, dist_utils, model_utils, log_utils
from utils.cache_identity import artifact_identity
from utils.loss_utils import tokenwise_kl_from_logits


_REF_LOGITS_SCHEMA_VERSION = 2
PAPER_QA_TASKS = (
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
    "ceval-valid",
    "boolq",
    "openbookqa",
    "social_iqa",
)


def _tensor_identity(tensor: torch.Tensor) -> str:
    """Hash the exact evaluation tokens, including layout metadata."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _reference_cache_metadata(args, analyzer, dataset, dataloader) -> dict:
    """Describe every artifact that changes the FP reference semantics."""
    model = analyzer.model
    tokenizer = analyzer.tokenizer
    prepared_path = getattr(
        model, "_gptqplus_prepared_checkpoint_path", None
    )
    tokenizer_source = getattr(tokenizer, "name_or_path", None)
    input_ids = dataloader.input_ids
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if tokenizer_vocab_size is None:
        tokenizer_vocab_size = len(tokenizer)
    model_config = getattr(model, "config", None)
    model_revision = getattr(model_config, "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None)
    if tokenizer_revision is None:
        tokenizer_revision = getattr(
            tokenizer, "init_kwargs", {}
        ).get("_commit_hash")
    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = analyzer.model.lm_head.in_features
    expected_samples = (
        int(input_ids.numel()) // int(args.eval_seq_len)
    )
    return {
        "schema_version": _REF_LOGITS_SCHEMA_VERSION,
        "source_model_identity": model_utils.source_model_cache_identity(args),
        "resolved_model_revision": model_revision,
        "prepared_checkpoint_identity": artifact_identity(prepared_path),
        "rotation_identity": model_utils.rotation_cache_identity(args),
        "checkpoint_is_rotated": bool(
            getattr(model, "_gptqplus_checkpoint_is_rotated", False)
        ),
        "model_dtype": str(next(model.parameters()).dtype),
        "tokenizer_identity": artifact_identity(tokenizer_source),
        "resolved_tokenizer_revision": tokenizer_revision,
        "tokenizer_class": type(tokenizer).__qualname__,
        "tokenizer_vocab_size": int(tokenizer_vocab_size),
        "dataset": str(dataset),
        "eval_seq_len": int(args.eval_seq_len),
        "eval_tokens_identity": _tensor_identity(input_ids),
        "hidden_states_shape": [
            expected_samples,
            int(args.eval_seq_len),
            int(hidden_size),
        ],
        "hidden_states_dtype": str(next(model.parameters()).dtype),
    }


def _atomic_torch_save(payload: dict, path: str) -> None:
    """Publish a complete cache archive with a same-filesystem replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_reference_cache(path: str, expected_metadata: dict):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        logging.warning("Ignoring unreadable reference cache %s: %s", path, exc)
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("metadata") != expected_metadata
        or not isinstance(payload.get("hidden_states"), torch.Tensor)
    ):
        logging.warning(
            "Ignoring stale/invalid reference cache %s (schema or identity mismatch).",
            path,
        )
        return None
    hidden_states = payload["hidden_states"]
    if (
        list(hidden_states.shape)
        != expected_metadata["hidden_states_shape"]
        or str(hidden_states.dtype)
        != expected_metadata["hidden_states_dtype"]
    ):
        logging.warning(
            "Ignoring invalid reference cache %s (hidden-state shape/dtype mismatch).",
            path,
        )
        return None
    flat = hidden_states.reshape(-1)
    for start in range(0, flat.numel(), 1_048_576):
        if not torch.isfinite(flat[start : start + 1_048_576]).all():
            logging.warning(
                "Ignoring invalid reference cache %s (non-finite hidden state).",
                path,
            )
            return None
    return hidden_states


def _resolve_paper_qa_tasks(pattern_match, all_tasks) -> list[str]:
    resolved = {
        task: pattern_match([task], all_tasks)
        for task in PAPER_QA_TASKS
    }
    invalid = {
        task: matches
        for task, matches in resolved.items()
        if len(matches) != 1
    }
    if invalid:
        raise RuntimeError(
            "qa_eval: each entry in the paper's ten-task protocol must resolve "
            f"to exactly one task; invalid resolutions={invalid!r}. "
            f"Requested {list(PAPER_QA_TASKS)!r}. "
            "Install lm_eval with task configs (`pip install lm-eval`) or "
            "populate ./datasets/lm_eval_configs/tasks."
        )
    task_names = [resolved[task][0] for task in PAPER_QA_TASKS]
    if len(task_names) != 10 or len(set(task_names)) != 10:
        raise RuntimeError(
            "qa_eval: the paper protocol must resolve to ten unique canonical "
            f"tasks, got {task_names!r}."
        )
    return task_names


def _task_accuracy(task_name: str, result: dict) -> float:
    if "acc_norm,none" in result:
        raw_acc = result["acc_norm,none"]
    elif "acc,none" in result:
        raw_acc = result["acc,none"]
    else:
        raise RuntimeError(
            f"qa_eval: task {task_name!r} returned neither "
            "'acc_norm,none' nor 'acc,none'."
        )
    return round(float(raw_acc) * 100, 2)


@torch.no_grad()
def _get_logits(args, analyzer: model_utils.ModelAnalyzer, testenc, dev):
    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    input_ids = testenc.input_ids  
    nsamples = input_ids.numel() // args.eval_seq_len
    input_ids = input_ids[:, :nsamples * args.eval_seq_len].view(nsamples, args.eval_seq_len).to(dev)

    for module in analyzer.get_pre_block_modules():
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, args.eval_seq_len, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in tqdm(range(nsamples), ncols=80, desc="Caching layer-0 inputs", leave=False):
        try:
            model(input_ids[i: i+1])
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    for i in tqdm(range(len(layers)), ncols=80, desc="Forwarding Layers"):
        layer = layers[i].to(dev)
        for j in tqdm(range(nsamples), ncols=80, desc=f"  layer {i}", leave=False, position=1):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        layers[i] = layer.to(orig_device)
        del layer
        memory_utils.cleanup_memory()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    memory_utils.cleanup_memory()

    # Get model logits
    model.model.norm.to(dev)
    lm_logits = []
    for i in tqdm(range(nsamples), ncols=80, desc="Applying final norm", leave=False):
        hidden_states = inps[i: i + 1].to(dev)
        hidden_states = model.model.norm(hidden_states)
        lm_logits.append(hidden_states.cpu())
    lm_logits = torch.cat(lm_logits, dim=0)
    model.model.norm.to(orig_device)

    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    memory_utils.cleanup_memory(verbos=True)

    return lm_logits, input_ids


@torch.no_grad()
def get_ref_logits(args, analyzer, dataset, dataloader):
    cache_dir = os.path.join(args.cache_dir, "ref_logits")
    os.makedirs(cache_dir, exist_ok=True)
    metadata = _reference_cache_metadata(
        args, analyzer, dataset, dataloader
    )
    cache_tag = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()[:20]
    ref_logits_path = os.path.join(
        cache_dir,
        f"{args.model_name}_{dataset}_test_{args.eval_seq_len}_{cache_tag}.cache",
    )
    ref_logits = (
        _load_reference_cache(ref_logits_path, metadata)
        if os.path.exists(ref_logits_path)
        else None
    )
    if ref_logits is None:
        logging.info(f"Generating reference logits for {dataset} at {ref_logits_path}...")
        ref_logits, _ = _get_logits(args, analyzer, dataloader, torch.device("cuda"))
        if dist_utils.is_main():
            _atomic_torch_save(
                {"metadata": metadata, "hidden_states": ref_logits},
                ref_logits_path,
            )
    else:
        logging.info(f"Loading reference logits for {dataset} from {ref_logits_path}...")
    orig_lm_head = copy.deepcopy(analyzer.model.lm_head)
    # Dtype is part of the cache identity.  The conditional is defensive for
    # malformed/custom models; unlike the legacy basename cache, a fp16 cache
    # can never silently stand in for a bf16 reference.
    target_dtype = orig_lm_head.weight.dtype
    if ref_logits.dtype != target_dtype:
        ref_logits = ref_logits.to(target_dtype)
    memory_utils.cleanup_memory()
    return ref_logits, orig_lm_head


@torch.no_grad()
def _kl_ppl_eval(args, analyzer: model_utils.ModelAnalyzer, orig_lm_head, dataloader, ref_logits_list):
    model = analyzer.model
    orig_device = next(model.parameters()).device

    dev = torch.device("cuda")

    logits_list, input_ids_list = _get_logits(args, analyzer, dataloader, dev)
    model.lm_head.to(dev)
    orig_lm_head.to(dev)

    kl_loss = 0
    nlls = []
    for logits, ref_logits, input_ids in tqdm(zip(logits_list, ref_logits_list, input_ids_list), ncols=80,
                                              total=len(ref_logits_list), desc="Computing PPL & KL"):
        # The paper evaluates full-vocabulary distribution shift.  Cast the
        # LM-head outputs before CE/softmax/KL so bf16 rounding does not erase
        # small quantisation differences or create a negative reported KL.
        logits = model.lm_head(logits.to(dev)).float()
        ref_logits = orig_lm_head(ref_logits.to(dev)).float()

        # NLL loss
        shift_labels = input_ids[None, 1:].to(dev)
        shift_logits = logits[None, :-1, :].to(dev)
        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels,
                               reduction="none")
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)

        # Paper evaluation uses full-vocabulary KL. Keep an explicit positive
        # top-k override for diagnostic/legacy runs, but never hard-code one.
        k = int(getattr(args, "kl_topk", -1))
        if k > 0:
            ref_logits, indices = ref_logits.topk(k, dim=-1, sorted=False)
            logits = logits.gather(-1, indices)
        kl_loss += tokenwise_kl_from_logits(
            logits, ref_logits
        ).mean()
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    kl_loss /= len(ref_logits_list)
    if kl_loss.item() < -1e-7:
        raise RuntimeError(
            "Full-vocabulary fp32 KL became materially negative "
            f"({kl_loss.item():.3e}); reference/student logits are invalid."
        )
    if kl_loss.item() < 0:
        logging.warning(
            "Clamping aggregate KL %.3e to zero (fp32 round-off).",
            kl_loss.item(),
        )
        kl_loss = kl_loss.clamp_min(0.0)

    model.lm_head.to(orig_device)
    orig_lm_head.to(orig_device)
    memory_utils.cleanup_memory()

    return ppl.item(), kl_loss.item()


def kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict):
    metric_vals = OrderedDict()
    for eval_dataset in args.eval_datasets:
        logging.info(f"Evaluating KL&PPL on {eval_dataset}")
        ppl, kl_loss = _kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict[eval_dataset], ref_logits_dict[eval_dataset])
        metric_vals[f"KL-{eval_dataset}"] = f"{kl_loss:.2e}"
        metric_vals[f"PPL-{eval_dataset}"] = f"{ppl:.2f}"
        logging.info(f"KL&PPL on {eval_dataset}: {kl_loss:.2e}, {ppl:.2f}")
    pretty_print_results(metric_vals)


def qa_eval(model, tokenizer, lm_eval_batch_size=32):
    # Keep lm-eval optional for the normal PPL/KL path.  Importing it at module
    # load time makes every quantisation run depend on the full QA-evaluation
    # dependency set even when --lm_eval is disabled.
    try:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM, eval_logger
    except ImportError as exc:
        raise RuntimeError(
            "qa_eval requires the optional lm-eval dependencies; install "
            "`lm-eval==0.4.4` before enabling --lm_eval."
        ) from exc
    eval_logger.level = logging.ERROR
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=lm_eval_batch_size)

    # Pick up the project's custom task YAMLs only if that directory exists.
    # Without this guard, `include_defaults=False` + a missing include_path
    # leaves `all_tasks` empty → `pattern_match` returns [] → later division
    # by zero. Always enabling defaults lets us fall back to lm_eval's built-in
    # task library when the custom dir isn't shipped with the checkout.
    custom_task_dir = "./datasets/lm_eval_configs/tasks"
    if os.path.isdir(custom_task_dir):
        task_manager = lm_eval.tasks.TaskManager(include_path=custom_task_dir, include_defaults=True)
    else:
        task_manager = lm_eval.tasks.TaskManager(include_defaults=True)
    task_names = _resolve_paper_qa_tasks(
        lm_eval_utils.pattern_match, task_manager.all_tasks
    )
    results, results_str = {}, {}
    for task_name in task_names:
        logging.info(f"Evaluating {task_name}...")
        hflm.batch_size_per_gpu = lm_eval_batch_size
        with log_utils.disable_logging_context():
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], task_manager=task_manager)['results']
        result = result[task_name]
        acc = _task_accuracy(task_name, result)
        results[task_name] = acc
        logging.info(f"acc: {acc}%")
    results_str.update({task: f"{result:.2f}" for task, result in results.items()})
    results_str['acc_avg'] = f"{sum(results.values()) / len(task_names):.2f}"
    pretty_print_results(results_str)


def pretty_print_results(data):
    headers = list(data.keys())
    values = [str(v) for v in data.values()]

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_row = "| " + " | ".join(values) + " |"

    logging.info("\n" + header_row + "\n" + separator_row + "\n" + data_row)
