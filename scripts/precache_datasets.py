"""Download and save all datasets to ROOT_DIR/datasets/ for offline use.
Run this from a Colab cell (not from bash) so the main process can reach HuggingFace:
    %cd /content/drive/MyDrive/gptq_plus_upload
    !python3 scripts/precache_datasets.py
"""
import os
from datasets import load_dataset, load_from_disk

ROOT_DATASETS = './datasets'
os.makedirs(ROOT_DATASETS, exist_ok=True)


def save_if_needed(local_path, name, download_fn):
    if os.path.exists(local_path):
        print(f"[SKIP] {name}: already at {local_path}")
        return
    print(f"[DOWN] {name}: downloading and saving to {local_path}...")
    data = download_fn()
    data.save_to_disk(local_path)
    print(f"[DONE] {name}")


# wikitext2: save each split separately
for split in ['train', 'test']:
    save_if_needed(
        f'{ROOT_DATASETS}/wikitext/{split}',
        f'wikitext2/{split}',
        lambda s=split: load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=s, trust_remote_code=True)
    )

# ultrachat_2k: save combined (first 2128 examples), select in data_utils
save_if_needed(
    f'{ROOT_DATASETS}/ultrachat_2k',
    'ultrachat_2k',
    lambda: load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft[:2128]', trust_remote_code=True)
)

# NuminaMath-1.5: save full train split, select in data_utils
save_if_needed(
    f'{ROOT_DATASETS}/NuminaMath-1.5',
    'NuminaMath-1.5',
    lambda: load_dataset('AI-MO/NuminaMath-1.5', split='train', trust_remote_code=True)
)

print("\nAll datasets saved. Ready to run experiments.")
