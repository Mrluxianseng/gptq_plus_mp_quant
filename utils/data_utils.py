import os
import random
import logging
from tqdm import tqdm

import torch
import numpy as np
import transformers
from datasets import load_dataset


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

    data = load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
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

    data = load_dataset("./datasets/LLM_compression_calibration", split=split, trust_remote_code=True)
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

    data = load_dataset("./datasets/NuminaMath-1.5", split=f"train[:256]" if split == "test" else "train[256:]", trust_remote_code=True)
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

    data = load_dataset("./datasets/ultrachat_2k", split=f"train_sft[:128]" if split == "test" else "train_sft[128:]", trust_remote_code=True)
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data['text']


def _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
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

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()
    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        seq_start = random.randint(0, len(tokens) - seq_len)

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

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    logging.info(f"Tokenizing {len(texts)} texts")
    trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
    samples = []
    pbar = tqdm(total=num_samples, desc=f"Sampling {num_samples} samples of length {seq_len}")
    while len(samples) < num_samples:
        idx = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tokens, save_path)

    return tokens


def get_loaders(dataset_name, split, tokenizer, seq_len, num_samples, seed=0):
    logging.info(f"Fetching dataset: {dataset_name}")
    texts = _get_dataset(tokenizer, dataset_name, split)
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")

    enc = tokenizer("\n\n".join(texts), return_tensors='pt')
    assert split in ["train", "test"]
    if split == "train":
        np.random.seed(seed)
        random.seed(seed)
        trainloader = []
        for _ in range(num_samples):
            i = random.randint(0, enc.input_ids.shape[1] - seq_len - 1)
            j = i + seq_len
            inp = enc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    elif split == "test":
        return enc
