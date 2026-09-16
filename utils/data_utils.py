import os
import random
import logging
from tqdm import tqdm

import torch
from datasets import load_dataset, load_from_disk


# Where a Hub download is materialised so the next run does not repeat it.
# Sits under datasets/, which is gitignored, next to the checked-in loaders.
DATASET_SNAPSHOT_ROOT = os.environ.get(
    "DATASET_SNAPSHOT_ROOT", os.path.join("datasets", "_hf_snapshots")
)

# name -> (local loader dir, Hub id, Hub config, split mapper)
# The local dir is tried first and is authoritative when present, so a machine
# that already has the data behaves exactly as before.
_DATASET_SOURCES = {
    "wikitext2": ("./datasets/wikitext", "wikitext", "wikitext-2-raw-v1"),
    "neuralmagic": ("./datasets/LLM_compression_calibration",
                    "neuralmagic/LLM_compression_calibration", None),
    "ultrachat_2k": ("./datasets/ultrachat_2k", "HuggingFaceH4/ultrachat_200k", None),
    "numinamath": ("./datasets/NuminaMath-1.5", "AI-MO/NuminaMath-1.5", None),
}


def _offline_reason():
    """Non-empty when an env var would make a Hub download fail anyway."""
    for var in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "0") not in ("0", "", "false", "False"):
            return var
    return ""


def _snapshot_dir(name, config, split):
    parts = [p for p in (name, config, str(split).replace("/", "_")) if p]
    return os.path.join(DATASET_SNAPSHOT_ROOT, *parts)


def load_split(name, split, hub_split=None):
    """Load one split, preferring the local loader directory.

    Order: the checked-in local loader dir, then a previously downloaded
    snapshot, then the Hub (which is then snapshotted so the download happens
    once per machine).

    `hub_split` overrides the split expression sent to the loader, for callers
    that slice (e.g. "train[:256]").
    """
    if name not in _DATASET_SOURCES:
        raise ValueError(f"Unknown dataset {name}")
    local_dir, hub_id, config = _DATASET_SOURCES[name]
    split_expr = hub_split or split

    if os.path.isdir(local_dir):
        return load_dataset(local_dir, config, split=split_expr, trust_remote_code=True)

    snap = _snapshot_dir(name, config, split_expr)
    if os.path.isdir(snap):
        logging.info("Loading %s[%s] from snapshot %s", name, split_expr, snap)
        return load_from_disk(snap)

    blocked_by = _offline_reason()
    if blocked_by:
        raise RuntimeError(
            f"Dataset '{name}' is not available locally: '{local_dir}' is missing and "
            f"no snapshot exists at '{snap}'. A Hub download would be needed, but "
            f"{blocked_by} is set. Either unset it (e.g. {blocked_by}=0) to allow a "
            f"one-time download, or place the dataset at '{local_dir}'."
        )

    logging.info(
        "Dataset '%s' not found locally; downloading %s (%s)[%s] from the Hub once.",
        name, hub_id, config or "default", split_expr,
    )
    data = load_dataset(hub_id, config, split=split_expr)
    try:
        os.makedirs(os.path.dirname(snap) or ".", exist_ok=True)
        tmp = f"{snap}.tmp.{os.getpid()}"
        data.save_to_disk(tmp)
        os.replace(tmp, snap)
        logging.info("Snapshotted %s[%s] to %s", name, split_expr, snap)
    except Exception as exc:  # a failed cache write must not fail the run
        logging.warning("Could not snapshot %s to %s: %s", name, snap, exc)
    return data


def ensure_datasets_available(names, splits=("train", "test")):
    """Preflight: resolve every dataset this run needs before any real work.

    Loading the calibration set happens early but the eval sets are only touched
    after the model is prepared, so without this a missing eval dataset surfaces
    minutes in. Downloads triggered here are the same one-time downloads the
    loaders would do.
    """
    for name in dict.fromkeys(n for n in names if n):
        if name not in _DATASET_SOURCES:
            raise ValueError(f"Unknown dataset {name}")
        local_dir, _, _ = _DATASET_SOURCES[name]
        ok = False
        for split in splits:
            try:
                load_split(name, split)
                ok = True
            except RuntimeError:
                # Genuinely unavailable: missing locally and a download is
                # blocked. That is the case this preflight exists to surface.
                raise
            except Exception as exc:
                # Split-name mismatches are expected -- ultrachat uses
                # train_sft, numinamath slices train -- and the real loaders
                # pass the right expression. A preflight must never fail a run
                # that would otherwise work.
                logging.debug(
                    "Dataset check: %s[%s] not resolvable by plain split name (%s)",
                    name, split, exc,
                )
        if not ok:
            logging.info(
                "Dataset check: %s could not be verified by plain split name; "
                "deferring to its own loader.", name,
            )
            continue
        logging.info(
            "Dataset check: %s ready (%s)",
            name, local_dir if os.path.isdir(local_dir) else "snapshot/Hub",
        )


def format_messages(messages: list[dict]) -> str:
    chunks = []
    system_done = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()

        assert role in ["system", "user", "assistant"]

        if role == "system" and not system_done:
            chunks.append(f"### Instruction:\n{content}")
            system_done = True
        elif role == "user":
            chunks.append(f"### Instruction:\n{content}")
        elif role == "assistant":
            chunks.append(f"### Response:\n{content}")

    text = "\n\n".join(chunks).strip()
    return text


def _get_wikitext2(split):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for wikitext2"

    data = load_split('wikitext2', split)
    return data['text']


def _get_neuralmagic(tokenizer, split):
    assert split in ['train'], "NeuralMagic only has a train split"

    def preprocess_fn(example):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            text = example["text"]
        return {"text": text}

    data = load_split('neuralmagic', split)
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data['text']


def _get_numinamath(tokenizer, split):
    assert split in ['train', 'test'], f"Unknown split {split} for numinamath"

    def preprocess_fn(example):
        example["messages"] = [
            {
                "content": example["problem"],
                "role": "user",
            },
            {
                "content": example["solution"],
                "role": "assistant",
            }
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            text = format_messages(example["messages"])
        return {"text": text}

    data = load_split(
        'numinamath', split,
        hub_split="train[:256]" if split == "test" else "train[256:]",
    )
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data['text']


def _get_ultrachat_2k(tokenizer, split):
    assert split in ['train', 'test'], f"Unknown split {split} for ultrachat_2k"

    def preprocess_fn(example):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            text = format_messages(example["messages"])
        return {"text": text}

    data = load_split(
        'ultrachat_2k', split,
        hub_split="train_sft[:128]" if split == "test" else "train_sft[128:]",
    )
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data['text']


def _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # A local RNG makes calibration sampling a self-contained seed domain:
    # cache misses and calibration-seed sweeps cannot perturb later
    # optimization RNG state.
    rng = random.Random(seed)

    selected_indices = set()

    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = rng.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        tokens = tokens[:seq_len]

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples


def _sample_and_tokenize_from_middle(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    rng = random.Random(seed)

    selected_indices = set()
    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = rng.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        seq_start = rng.randint(0, len(tokens) - seq_len)

        tokens = tokens[seq_start:seq_start + seq_len]
        assert tokens.shape[-1] == seq_len, f"Token length {len(tokens)} != seq_len {seq_len}"

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()
    return samples


def _sample_concat_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
    f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    rng = random.Random(seed)

    selected_indices = set()

    logging.info(f"Tokenizing {len(texts)} texts")
    trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
    samples = []
    pbar = tqdm(total=num_samples, desc=f"Sampling {num_samples} samples of length {seq_len}")
    while len(samples) < num_samples:
        idx = rng.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        
        # if selected_indices:
        #     closest_idx = min(selected_indices, key=lambda x: abs(x - idx), default=idx)
        #     if idx <= closest_idx + seq_len and idx >= closest_idx - seq_len:
        #         continue

        j = idx + seq_len
        inp = trainenc.input_ids[:, idx:j]
        tokens = inp.clone()
        tokens = tokens.squeeze(0)

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples


def _get_dataset(tokenizer, dataset_name, split):
    if dataset_name == 'wikitext2':
        return _get_wikitext2(split)
    elif dataset_name == 'neuralmagic':
        return _get_neuralmagic(tokenizer, split)
    elif dataset_name == 'ultrachat_2k':
        return _get_ultrachat_2k(tokenizer, split)
    elif dataset_name == 'numinamath':
        return _get_numinamath(tokenizer, split)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_tokens(dataset_name, split, tokenizer, seq_len, num_samples, save_path=None, seed=0):

    if save_path is not None and os.path.isfile(save_path):
        logging.info(f"Loading tokens from {save_path}")
        return torch.load(save_path)

    logging.info(f"Fetching dataset: {dataset_name}")
    texts = _get_dataset(tokenizer, dataset_name, split)
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")

    tokens = _sample_concat_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)
    # tokens = _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)

    if save_path is not None:
        logging.info(f"Saving tokens to {save_path}")
        cache_parent = os.path.dirname(save_path) or "."
        os.makedirs(cache_parent, exist_ok=True)
        # All DP ranks may observe the same initial miss. They generate
        # byte-equivalent tensors from the local seeded RNG, but must never
        # expose a partially-written torch archive to another rank.
        tmp_path = f"{save_path}.tmp.{os.getpid()}"
        try:
            torch.save(tokens, tmp_path)
            os.replace(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return tokens


def get_loaders(dataset_name, split, tokenizer, seq_len, num_samples, seed=0):
    logging.info(f"Fetching dataset: {dataset_name}")
    texts = _get_dataset(tokenizer, dataset_name, split)
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")

    enc = tokenizer("\n\n".join(texts), return_tensors='pt')
    assert split in ["train", "test"]
    if split == "train":
        rng = random.Random(seed)
        trainloader = []
        for _ in range(num_samples):
            i = rng.randint(0, enc.input_ids.shape[1] - seq_len - 1)
            j = i + seq_len
            inp = enc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    elif split == "test":
        return enc
