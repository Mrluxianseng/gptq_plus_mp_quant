# REAL-Q local dataset mirror

Validated on 2026-07-24 in Canoe job `j-7x9o0je4pk` with
`datasets==3.6.0`, `pyarrow==17.0.0`, `HF_DATASETS_OFFLINE=1`, and
`HF_HUB_OFFLINE=1`.

The machine-readable source, payload, schema, installation, and validation
record is:

```text
/minimax-avatar-new/zhangqian/realq/dataset_mirror_staging/MANIFEST.json
```

## Layout and installation

The persistent mirror is:

```text
/minimax-avatar-new/zhangqian/realq/dataset_mirror_staging/
├── wikitext/
├── LLM_compression_calibration/
├── NuminaMath-1.5/
└── ultrachat_2k/
```

The current checkout uses relative symlinks at the exact locations expected by
`utils/data_utils.py`:

```text
datasets/wikitext -> ../../dataset_mirror_staging/wikitext
datasets/LLM_compression_calibration -> ../../dataset_mirror_staging/LLM_compression_calibration
datasets/NuminaMath-1.5 -> ../../dataset_mirror_staging/NuminaMath-1.5
datasets/ultrachat_2k -> ../../dataset_mirror_staging/ultrachat_2k
```

The repository's existing `.gitignore` rule `datasets*` ignores this local
installation. No dataset payload is copied into Git and no loader change is
required.

To install the same mirror in another checkout located next to
`dataset_mirror_staging`:

```bash
mkdir -p datasets
ln -s ../../dataset_mirror_staging/wikitext datasets/wikitext
ln -s ../../dataset_mirror_staging/LLM_compression_calibration datasets/LLM_compression_calibration
ln -s ../../dataset_mirror_staging/NuminaMath-1.5 datasets/NuminaMath-1.5
ln -s ../../dataset_mirror_staging/ultrachat_2k datasets/ultrachat_2k
```

## Pinned sources and files

All sources are public, ungated Hugging Face dataset repositories. The three
non-WikiText sources use commit-pinned `resolve/<source_revision>/...` URLs.
WikiText-2 was already downloaded for the formal ablation; the mirror reuses
those verified blobs. For WikiText, the source dataset `main` commit and the
generated parquet conversion commit are separate pins.

| REAL-Q name | Official repository | Source revision | Payload revision | Split rows |
|---|---|---|---|---:|
| `wikitext2` | `Salesforce/wikitext`, config `wikitext-2-raw-v1` | `b08601e04326c79dfdd32d625aee71d232d685c3` | `refs/convert/parquet` commit `3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e` | train 36,718; validation 3,760; test 4,358 |
| `neuralmagic` | `neuralmagic/LLM_compression_calibration` | `85e4a40773bf4cbc9dc17d6c63ee69ccd8390b6d` | same repository commit | train 10,000 |
| `ultrachat_2k` | `neuralmagic/ultrachat_2k` | `5313fae072417a33dbcb307a7643d48617ec6425` | same repository commit | train_sft 2,048 |
| `numinamath` | `AI-MO/NuminaMath-1.5` | `1b05109f9e5c1ad06c0663519502416c30b300f8` | same repository commit | train 896,215 |

| Relative file | Bytes | SHA256 |
|---|---:|---|
| `wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet` | 6,357,543 | `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` |
| `wikitext/wikitext-2-raw-v1/validation-00000-of-00001.parquet` | 657,209 | `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c` |
| `wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet` | 732,610 | `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91` |
| `LLM_compression_calibration/calibration.json.gz` | 4,575,521 | `30a5165952be966bfba94e53c0b847dd2770d46666b4564520576feaee2d1dbe` |
| `ultrachat_2k/data/train_sft-00000-of-00001.parquet` | 7,145,258 | `f2616cc2813cdcf6f5666b098a4e249d1dedd73fdacfa3edd4145b642725d582` |
| `NuminaMath-1.5/data/train-00000-of-00003.parquet` | 195,064,200 | `1e37cedcc5104c0d5a9afbf8b26328a59d7005cd37c2bd08352419f5f0e03214` |
| `NuminaMath-1.5/data/train-00001-of-00003.parquet` | 175,917,356 | `bc48fdddc1ab4e559727ad4f1f03b0263785c1c16b7964fb885fba07e18d6a39` |
| `NuminaMath-1.5/data/train-00002-of-00003.parquet` | 160,374,171 | `951aa899fe2a1b64f5b40c6636372d35b9db4f0b9c922e874d4bf357db64da66` |

The SHA256 values for the non-WikiText payloads equal the LFS object IDs
reported by the Hugging Face Hub tree API at their source revisions. The
WikiText conversion ref resolves to commit
`3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e`; its tree API reports the three
listed hashes as the parquet LFS object IDs. Those hashes also equal the formal
REAL-Q experiment pins. Thus neither the mutable conversion-ref name nor the
source-repository revision is being used as a substitute for payload identity.

## Local schema

- `wikitext2`: `text: string`.
- `neuralmagic`: `text: string`;
  `messages: list<struct<content: string, role: string>>`.
- `ultrachat_2k`: `prompt: string`, `prompt_id: string`;
  `messages: list<struct<content: string, role: string>>`.
- `numinamath`: `problem`, `solution`, `answer`, `problem_type`,
  `question_type`, `problem_is_valid`, `solution_is_valid`, and `source` are
  strings; `synthetic` is boolean.

These fields satisfy every field read by `utils/data_utils.py`: WikiText uses
`text`, the NeuralMagic calibration loader uses `messages` (or `text` as its
fallback), UltraChat uses `messages`, and NuminaMath uses `problem` plus
`solution`.

## Offline verification

Run from the repository root inside the experiment container:

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python - <<'PY'
from datasets import load_dataset

checks = [
    (
        "wikitext2",
        load_dataset(
            "./datasets/wikitext",
            "wikitext-2-raw-v1",
            split="test",
            trust_remote_code=True,
        ),
        4358,
        {"text"},
    ),
    (
        "neuralmagic",
        load_dataset(
            "./datasets/LLM_compression_calibration",
            split="train",
            trust_remote_code=True,
        ),
        10000,
        {"text", "messages"},
    ),
    (
        "numinamath-test",
        load_dataset(
            "./datasets/NuminaMath-1.5",
            split="train[:256]",
            trust_remote_code=True,
        ),
        256,
        {"problem", "solution"},
    ),
    (
        "ultrachat_2k-test",
        load_dataset(
            "./datasets/ultrachat_2k",
            split="train_sft[:128]",
            trust_remote_code=True,
        ),
        128,
        {"messages"},
    ),
]

for label, dataset, expected_rows, required_columns in checks:
    assert len(dataset) == expected_rows
    assert required_columns <= set(dataset.column_names)
    print("PASS", label, len(dataset), dataset.column_names, dataset.features)
PY
```

This exact check passed for all four datasets on 2026-07-24. The decisive run
used a newly created `/tmp` cache with zero files before the call, built 27
cache files (1,206,999,080 bytes), also passed all four production
`data_utils.py` wrappers, and removed the temporary cache afterward. Dataset
construction plus wrapper validation took 4.626 seconds in Python (4 seconds
whole-second wall-clock resolution). Its complete log is:

```text
/minimax-avatar-new/zhangqian/realq/dataset_mirror_staging/validation/offline_fresh_cache_20260724T123301Z.log
```

The reusable fresh-cache runner is:

```text
/minimax-avatar-new/zhangqian/realq/dataset_mirror_staging/run_fresh_offline_smoke.sh
```

To recheck file identity:

```bash
find /minimax-avatar-new/zhangqian/realq/dataset_mirror_staging \
  -type f \( -name '*.parquet' -o -name '*.json.gz' \) \
  -print0 | sort -z | xargs -0 sha256sum
```
