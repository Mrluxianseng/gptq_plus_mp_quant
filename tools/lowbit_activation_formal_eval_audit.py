#!/usr/bin/env python3
"""Fail-closed, network-free audit of the formal evaluation protocol.

This program intentionally does not import ``lm_eval`` or ``datasets`` and
does not instantiate a task.  It audits the installed harness, task YAML
closure, local Hugging Face cache metadata/artifacts, the project's evaluation
call sites, and the local WikiText-2 source using ordinary file and AST reads.

The JSON written to stdout (and, optionally, ``--write-manifest``) is
canonical: UTF-8, sorted keys, compact separators, and one trailing newline.
An invalid or incomplete audit exits non-zero.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "experiments" / "lowbit_activation" / "plan.json"
SCHEMA_VERSION = 2

EXPECTED_TASKS = (
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
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.9.1",
    "torch_runtime": "2.9.1+cu128",
    "transformers": "4.56.2",
    "lm-eval": "0.4.4",
    "datasets": "3.6.0",
    "accelerate": "1.12.0",
}
PLAN_RUNTIME_VERSION_KEYS = (
    "torch",
    "torch_runtime",
    "transformers",
    "lm-eval",
)
EXPECTED_SIMPLE_EVALUATE_DEFAULTS = {
    "num_fewshot": None,
    "limit": None,
    "system_instruction": None,
    "apply_chat_template": False,
    "fewshot_as_multiturn": False,
}

# These hashes pin the reviewed lm-eval 0.4.4 implementation and the exact
# transitive task YAML/include closure used by the paper's ten-task protocol.
EXPECTED_EVALUATOR_SHA256 = (
    "6edf1298d87080a15ba95905e792d4170d2d9faf375f87effe26d74248e12385"
)
EXPECTED_TASK_CLOSURE_SHA256 = (
    "db9f217991e10064fa6ff0d13698dab09943e2f5a469ce70e506c36e41bab6b7"
)
# This is a deliberately conservative execution closure: every Python source
# shipped inside the reviewed lm-eval package is pinned.  That contains the
# task-local modules referenced by ``!function`` plus TaskManager, YAML loader,
# task/group, metric, filter, evaluator, and Hugging Face model code.  Pinning
# the package-wide Python set avoids silently missing a transitive import.
EXPECTED_LM_EVAL_PYTHON_CLOSURE_SHA256 = (
    "0843a8e42eccd7edfa47173f8d511638d7f90f371ef017e1770c5daace95b47d"
)
REQUIRED_LM_EVAL_RUNTIME_FILES = (
    "evaluator.py",
    "evaluator_utils.py",
    "utils.py",
    "tasks/__init__.py",
    "api/filter.py",
    "api/group.py",
    "api/metrics.py",
    "api/model.py",
    "api/registry.py",
    "api/task.py",
    "filters/__init__.py",
    "filters/extraction.py",
    "filters/selection.py",
    "filters/transformation.py",
    "models/huggingface.py",
    "models/utils.py",
)

EXPECTED_PROJECT_SOURCE_SHA256 = {
    "data_utils": "ae8ac74254b2616e3eec7e0a74110e74e3114b9cb10375a4a7d18639b7ff8489",
    "eval_utils": "e486fcd2d7fe31b78eeb8a407e8667b6f4774c7f2395e728b4654f7c41e1cbff",
    "realq_pipeline": "d02ef5535faf1892089e3c8db1eec403a9b690d9b7b906f1b69041ae89315bd0",
    "legacy_ptq": "0356956989a97fa8fc350f1a84274c2e2c18d77b092217b4355d136a2686b495",
}

CEVAL_LEAVES = (
    "ceval-valid_computer_network",
    "ceval-valid_operating_system",
    "ceval-valid_computer_architecture",
    "ceval-valid_college_programming",
    "ceval-valid_college_physics",
    "ceval-valid_college_chemistry",
    "ceval-valid_advanced_mathematics",
    "ceval-valid_probability_and_statistics",
    "ceval-valid_discrete_mathematics",
    "ceval-valid_electrical_engineer",
    "ceval-valid_metrology_engineer",
    "ceval-valid_high_school_mathematics",
    "ceval-valid_high_school_physics",
    "ceval-valid_high_school_chemistry",
    "ceval-valid_high_school_biology",
    "ceval-valid_middle_school_mathematics",
    "ceval-valid_middle_school_biology",
    "ceval-valid_middle_school_physics",
    "ceval-valid_middle_school_chemistry",
    "ceval-valid_veterinary_medicine",
    "ceval-valid_college_economics",
    "ceval-valid_business_administration",
    "ceval-valid_marxism",
    "ceval-valid_mao_zedong_thought",
    "ceval-valid_education_science",
    "ceval-valid_teacher_qualification",
    "ceval-valid_high_school_politics",
    "ceval-valid_high_school_geography",
    "ceval-valid_middle_school_politics",
    "ceval-valid_middle_school_geography",
    "ceval-valid_modern_chinese_history",
    "ceval-valid_ideological_and_moral_cultivation",
    "ceval-valid_logic",
    "ceval-valid_law",
    "ceval-valid_chinese_language_and_literature",
    "ceval-valid_art_studies",
    "ceval-valid_professional_tour_guide",
    "ceval-valid_legal_professional",
    "ceval-valid_high_school_chinese",
    "ceval-valid_high_school_history",
    "ceval-valid_middle_school_history",
    "ceval-valid_civil_servant",
    "ceval-valid_sports_science",
    "ceval-valid_plant_protection",
    "ceval-valid_basic_medicine",
    "ceval-valid_clinical_medicine",
    "ceval-valid_urban_and_rural_planner",
    "ceval-valid_accountant",
    "ceval-valid_fire_engineer",
    "ceval-valid_environmental_impact_assessment_engineer",
    "ceval-valid_tax_accountant",
    "ceval-valid_physician",
)

CEVAL_SAMPLE_COUNTS = {
    "accountant": 49,
    "advanced_mathematics": 19,
    "art_studies": 33,
    "basic_medicine": 19,
    "business_administration": 33,
    "chinese_language_and_literature": 23,
    "civil_servant": 47,
    "clinical_medicine": 22,
    "college_chemistry": 24,
    "college_economics": 55,
    "college_physics": 19,
    "college_programming": 37,
    "computer_architecture": 21,
    "computer_network": 19,
    "discrete_mathematics": 16,
    "education_science": 29,
    "electrical_engineer": 37,
    "environmental_impact_assessment_engineer": 31,
    "fire_engineer": 31,
    "high_school_biology": 19,
    "high_school_chemistry": 19,
    "high_school_chinese": 19,
    "high_school_geography": 19,
    "high_school_history": 20,
    "high_school_mathematics": 18,
    "high_school_physics": 19,
    "high_school_politics": 19,
    "ideological_and_moral_cultivation": 19,
    "law": 24,
    "legal_professional": 23,
    "logic": 22,
    "mao_zedong_thought": 24,
    "marxism": 19,
    "metrology_engineer": 24,
    "middle_school_biology": 21,
    "middle_school_chemistry": 20,
    "middle_school_geography": 12,
    "middle_school_history": 22,
    "middle_school_mathematics": 19,
    "middle_school_physics": 19,
    "middle_school_politics": 21,
    "modern_chinese_history": 23,
    "operating_system": 19,
    "physician": 49,
    "plant_protection": 22,
    "probability_and_statistics": 18,
    "professional_tour_guide": 29,
    "sports_science": 19,
    "tax_accountant": 49,
    "teacher_qualification": 44,
    "urban_and_rural_planner": 46,
    "veterinary_medicine": 23,
}

# Each digest covers the exact dataset_info.json plus the one Arrow artifact
# for the evaluated split.  Builder fingerprints and row counts alone are not
# content identities: a truncated or modified Arrow file can retain both.
EXPECTED_DATASET_ARTIFACT_SHA256 = {
    "arc_challenge": "4479e0a5e59fee2b13f5567849dd59161f87ff9b3f7a71ce615e4ef976879cea",
    "arc_easy": "d5ce5c871fa6e8699546e2cd9d353bd2f41c30122ffff9513f535bb64ffb86e7",
    "boolq": "72b97660c1fd80bf570d0f94714f849bdf70eebe6dc96489835d4153a918bd2c",
    "ceval-valid_accountant": "9cfdbe656f987c7eca843a1c089563038782942533e0ef3478b91586147f846a",
    "ceval-valid_advanced_mathematics": "cbbac004689640a3a888c9b3e979a077d417828fd8d9b45af18fb04f07dfcfc4",
    "ceval-valid_art_studies": "834e023940718359d8b66074170f1a8ea60cba10313357eca31e6f1e9797f8ef",
    "ceval-valid_basic_medicine": "95bf741e608fcef882f67b4df719e3939d62bc6b1733f6f1cd16705bd761ec2f",
    "ceval-valid_business_administration": "4f29888c2708b7fef3983d65f9e162152bd0de9fd86b1cfeb89c9d7898f3e3d0",
    "ceval-valid_chinese_language_and_literature": "87cf0a9d74e885cb0338f93139c7728b8b6104238b30e0a8abe27d68fb469a84",
    "ceval-valid_civil_servant": "812eb6e56d0b94985fa195f6cf1a6dcec8fcd1c1afee8cb533102a587f993803",
    "ceval-valid_clinical_medicine": "1202c1bed4dd1b5f0048f31d15345986555be41c1c382ff0585643284d62da25",
    "ceval-valid_college_chemistry": "c64cbe819aad9d998c007c89ba83b830e1b92ea33615c0b9a9cf1321e0801de1",
    "ceval-valid_college_economics": "b3fa62dc7b6be00699d2fd76b027c3995f4555909b3aaceed641ce3deda6e559",
    "ceval-valid_college_physics": "52f6d0c08331f1a6351f206a14c95133633d6ad2f230a15b97dd749f64828292",
    "ceval-valid_college_programming": "ce932e3eec73aa4df7b52fb853a0d57a743c6d22a903a316745b65284130a349",
    "ceval-valid_computer_architecture": "1ac2dab8214b78901a1fa469b8b1c81f0639ae8321a80222a0d6b1eccef2465c",
    "ceval-valid_computer_network": "eed741d542d00ad68681a21cf33a5958bbfe1277c1853dc81ff45f9b4ff597ab",
    "ceval-valid_discrete_mathematics": "bad86252c5f2ac56347ce284e81fdc803f7baf7b67020cbc82f0623697e5ffdc",
    "ceval-valid_education_science": "b0899d5689936f16bffdf6d2384fcd8a171ad5464929baad8c41fda2a1b61352",
    "ceval-valid_electrical_engineer": "d709d470774dd060af4a3a409246432b7f32c47d854de2fd16ad7e93765ce641",
    "ceval-valid_environmental_impact_assessment_engineer": "d1b6bcf713e3dc2f181de353188331fa41d05627c60631165a6e4d884dc5dd8f",
    "ceval-valid_fire_engineer": "bab76c956497201549151b1852d66643dae9d9eb3a2b29462aa4bfe35a35918c",
    "ceval-valid_high_school_biology": "5b2d44dfc90ace30982ae15cd5a63dfaaedd401169e830d078b0ca368368d220",
    "ceval-valid_high_school_chemistry": "ebe6d6eee8b30696876b8f417f6925033cad2692e423ee1b1638f34653e1ce7c",
    "ceval-valid_high_school_chinese": "d737727efda28ec309ac18a1fbb8bdd27cc1343fa779044612bd3ded5cece346",
    "ceval-valid_high_school_geography": "26a97439dce19723c47cdfd71795890d4e46df5d6734fc62dd325bbe29dcabb8",
    "ceval-valid_high_school_history": "71732b6552ab3523d1d32fc5ccfc0781aa3c319139626ff52ed38a29a95f2aa3",
    "ceval-valid_high_school_mathematics": "6ce7b915b8d859b6deed0e382a2065bf910e7977bc8fc93be75561e08e3e520e",
    "ceval-valid_high_school_physics": "1b7cdb1124df49e23b54e730ff71bea7b6fc7afb6bf9556ff017b378f119ccd1",
    "ceval-valid_high_school_politics": "6ea17d28be7591ce38253969aa329394bbe0da76d6e15a6b71bfd33e07d7f2df",
    "ceval-valid_ideological_and_moral_cultivation": "4e8e790b48b497c688c357abe8cdb35e7cb5333d27dbe66eca2681151f2bcd6d",
    "ceval-valid_law": "ac08765e9230d48b6618a0ef20200901b73c6d2fd65152add4f656725e926078",
    "ceval-valid_legal_professional": "18332a29d725394f8aafb0ac9e1ff0430cb19f6fdea6a8691cf81d92b5d84e94",
    "ceval-valid_logic": "625bd2c7d1e519c52727b3672fa6f04b85fbc6d0ed58fd7f0aab31f91cffcc26",
    "ceval-valid_mao_zedong_thought": "acc22f8fb4b4ee1f4e8c9b1345fc83eab5014eeca87f4b7ebb5744c9c7b8726e",
    "ceval-valid_marxism": "d4d46ef5dfe555e1af301a17d46277c974f14b8208a3ab081e89f8023d842548",
    "ceval-valid_metrology_engineer": "27e716802cd6c7eba0e22e3c01f0851baff62143071e9df9bfcc8bb922a6751f",
    "ceval-valid_middle_school_biology": "6f981e02095b8eb672a6801d5ba208dd184575a8131ff1832eb5280b05543584",
    "ceval-valid_middle_school_chemistry": "e80906f73f76beed982d5a7cef832d2cd08b10a911942a358f866671d1851fba",
    "ceval-valid_middle_school_geography": "3411310a1c677614c6a9dbdd20e18293e8aa1b01d49ddce8a93bce01ad0afb59",
    "ceval-valid_middle_school_history": "29ea9db1b9521a45055032b8beb2820b7ed11c680c3b0b2c4e29af1e29e69a6a",
    "ceval-valid_middle_school_mathematics": "9434bafbc0e65d1c553a263306cec5cc5571dcf0c637d5dc048dc3ce963cb8eb",
    "ceval-valid_middle_school_physics": "af542b24ed046bd08bc247974efcf46291eb33702a33d0ecb0f10fef2c788066",
    "ceval-valid_middle_school_politics": "c07646e589ddaf3890cf1af5c60c441502394f187f5dbecc1f6aaa2f661804fd",
    "ceval-valid_modern_chinese_history": "34d259ce20e3d0c2c77365cce2ce860f89bd164e437d7c7bf22facb8ccbc5794",
    "ceval-valid_operating_system": "f66d968876dbe87cdb80729dc7d34fda9ef322164f9c48d9195d87ff84795f1d",
    "ceval-valid_physician": "5c70667a1f0da6d7c2df49c6450cfe8787b3c08c3098342e74905ea81169a8a5",
    "ceval-valid_plant_protection": "0c488eec081f9203fcfc1194bab6abb49a932d56636206a1daa8f5251213cdcb",
    "ceval-valid_probability_and_statistics": "cc8d592fbe70ea8c5b2245793124e59ee985145c5bc7d26095f19b2094d4ed7c",
    "ceval-valid_professional_tour_guide": "bea9b5cb62191ceb307ae2cfd01276db6d9c372843bd6c7365783a9f3934738e",
    "ceval-valid_sports_science": "1164140418b52d5d3ade8b9d1f2dbee8b803199fa3310caf9ef104f20658b1bd",
    "ceval-valid_tax_accountant": "f541c741cc2b7beedcf58ea99a0a428e8fd104b6e6fd20e549cb34b148ae27f0",
    "ceval-valid_teacher_qualification": "e7dd1255cca3546a26481e4d60bc20a0fcfa485bceda04857e564ab3bce81b32",
    "ceval-valid_urban_and_rural_planner": "c8dba06b1792fc80fe2178ae747749a61666e870c938facd8afbd1cd27658ed3",
    "ceval-valid_veterinary_medicine": "981108b9dca3e515d1f021a4a30450291fdddf80c58ba78f75ae36978064fabc",
    "hellaswag": "92b4da22ebc0f4bdb4d26a44b955f98c8e128d137781a81707a38c65035874e8",
    "lambada_openai": "c5155b601c1a02af21c0c2990c0d794b29117822c0a513940c371dde10c04a10",
    "openbookqa": "00c397614451a84d332de78d05f2eb5443d14091fc797063e981e7665345aaa6",
    "piqa": "ad2bda3e02dd5a71a8ec9460794581acd73ca35b8548f8238cea3cffdb454554",
    "social_iqa": "08ff2425b6d5895d425cbf92f8afe6d1e4db57d810666e4f7a179d9444483c35",
    "wikitext2": "b42a9ba6f6d93d2d82a8e5ea0859e3f366cd64817bc06f68467e52a32eedb31f",
    "winogrande": "174bc8134966f7c063189f30fac2fede7a19b9601d2719c92ef7a68bcac51053",
}

_COMMON_DATASET_FINGERPRINTS = {
    "arc": "210d026faf9955653af8916fad021475a3f00453",
    "boolq": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
    "ceval": "617524a00b307ff6f9933702f724131fe12ca7ce",
    "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "lambada": "900124bf3b8235c6daf21033af9948b3f07346c4",
    "openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "piqa": "6c611c1a9bf220943c4174e117d3b660859665baf1d43156230116185312d011",
    "social_iqa": "674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8",
    "wikitext2": "2bbebf333b82adc0",
    "winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
}

TASK_PROTOCOLS: dict[str, dict[str, Any]] = {
    "piqa": {
        "dataset_path": "piqa",
        "task_dataset_name": None,
        "cache_dataset_name": "piqa",
        "cache_config_name": "plain_text",
        "dataset_version": "1.1.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["piqa"],
        "split": "validation",
        "sample_count": 1838,
        "task_version": "1.0",
        "metric": "acc_norm",
        "metric_key": "acc_norm,none",
    },
    "hellaswag": {
        "dataset_path": "hellaswag",
        "task_dataset_name": None,
        "cache_dataset_name": "hellaswag",
        "cache_config_name": "default",
        "dataset_version": "0.0.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["hellaswag"],
        "split": "validation",
        "sample_count": 10042,
        "task_version": "1.0",
        "metric": "acc_norm",
        "metric_key": "acc_norm,none",
    },
    "arc_easy": {
        "dataset_path": "allenai/ai2_arc",
        "task_dataset_name": "ARC-Easy",
        "cache_dataset_name": "ai2_arc",
        "cache_config_name": "ARC-Easy",
        "dataset_version": "0.0.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["arc"],
        "split": "test",
        "sample_count": 2376,
        "task_version": "1.0",
        "metric": "acc_norm",
        "metric_key": "acc_norm,none",
    },
    "arc_challenge": {
        "dataset_path": "allenai/ai2_arc",
        "task_dataset_name": "ARC-Challenge",
        "cache_dataset_name": "ai2_arc",
        "cache_config_name": "ARC-Challenge",
        "dataset_version": "0.0.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["arc"],
        "split": "test",
        "sample_count": 1172,
        "task_version": "1.0",
        "metric": "acc_norm",
        "metric_key": "acc_norm,none",
    },
    "winogrande": {
        "dataset_path": "winogrande",
        "task_dataset_name": "winogrande_xl",
        "cache_dataset_name": "winogrande",
        "cache_config_name": "winogrande_xl",
        "dataset_version": "0.0.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["winogrande"],
        "split": "validation",
        "sample_count": 1267,
        "task_version": "1.0",
        "metric": "acc",
        "metric_key": "acc,none",
    },
    "lambada_openai": {
        "dataset_path": "EleutherAI/lambada_openai",
        "task_dataset_name": "default",
        "cache_dataset_name": "lambada_openai",
        "cache_config_name": "default",
        "dataset_version": "0.0.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["lambada"],
        "split": "test",
        "sample_count": 5153,
        "task_version": "1.0",
        "metric": "acc",
        "metric_key": "acc,none",
    },
    "boolq": {
        "dataset_path": "super_glue",
        "task_dataset_name": "boolq",
        "cache_dataset_name": "super_glue",
        "cache_config_name": "boolq",
        "dataset_version": "0.0.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["boolq"],
        "split": "validation",
        "sample_count": 3270,
        "task_version": "2.0",
        "metric": "acc",
        "metric_key": "acc,none",
    },
    "openbookqa": {
        "dataset_path": "openbookqa",
        "task_dataset_name": "main",
        "cache_dataset_name": "openbookqa",
        "cache_config_name": "main",
        "dataset_version": "0.0.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["openbookqa"],
        "split": "test",
        "sample_count": 500,
        "task_version": "1.0",
        "metric": "acc_norm",
        "metric_key": "acc_norm,none",
    },
    "social_iqa": {
        "dataset_path": "social_i_qa",
        "task_dataset_name": None,
        "cache_dataset_name": "social_i_qa",
        "cache_config_name": "default",
        "dataset_version": "0.1.0",
        "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["social_iqa"],
        "split": "validation",
        "sample_count": 1954,
        "task_version": "0.0",
        "metric": "acc",
        "metric_key": "acc,none",
    },
}

WIKITEXT_PROTOCOL = {
    "dataset_path": "./datasets/wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "cache_dataset_name": "wikitext",
    "cache_config_name": "wikitext-2-raw-v1",
    "dataset_version": "0.0.0",
    "builder_fingerprint": _COMMON_DATASET_FINGERPRINTS["wikitext2"],
    "split": "test",
    "sample_count": 4358,
    "eval_seq_len": 2048,
}
EXPECTED_WIKITEXT_FILES = {
    "test-00000-of-00001.parquet": {
        "sha256": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
        "size_bytes": 732610,
    },
    "train-00000-of-00001.parquet": {
        "sha256": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
        "size_bytes": 6357543,
    },
    "validation-00000-of-00001.parquet": {
        "sha256": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
        "size_bytes": 657209,
    },
}
EXPECTED_WIKITEXT_COMBINED_SHA256 = (
    "bcf4550f15820804c2fb686595fa25087131a3d2e7df7ab30152e70c742774bd"
)


class AuditError(RuntimeError):
    """An audit input is absent, ambiguous, or semantically invalid."""


class _TaggedSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves lm-eval's ``!function`` scalars."""


class _TaggedScalar(str):
    """A string-compatible YAML scalar that retains its custom tag."""

    def __new__(cls, value: str, tag_suffix: str) -> "_TaggedScalar":
        instance = super().__new__(cls, value)
        instance.tag_suffix = tag_suffix
        return instance


def _unknown_yaml_tag(
    loader: _TaggedSafeLoader,
    tag_suffix: str,
    node: yaml.Node,
) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return _TaggedScalar(loader.construct_scalar(node), tag_suffix)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise AuditError(f"unsupported YAML node {type(node).__name__}")


_TaggedSafeLoader.add_multi_constructor("!", _unknown_yaml_tag)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return the exact canonical JSON representation emitted by the CLI."""

    return _canonical_bytes(value).decode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(value: Any) -> str:
    # The newline is deliberately excluded from composite identities.
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            root.resolve(strict=True)
        ).as_posix()
    except (FileNotFoundError, ValueError):
        return str(path.resolve())


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise AuditError(f"not a regular file: {resolved}")
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise AuditError(f"file escapes audited root: {resolved}") from exc
    return {
        "relative_path": relative,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _hash_file_set(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    """Hash a set by root-relative names, independent of roots/input order."""

    unique = {path.resolve(strict=True) for path in paths}
    entries = sorted(
        (_file_record(path, root) for path in unique),
        key=lambda item: item["relative_path"],
    )
    return {
        "file_count": len(entries),
        "files": entries,
        "combined_sha256": _stable_digest(entries),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_TaggedSafeLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AuditError(f"cannot parse YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"task YAML is not a mapping: {path}")
    return value


def _yaml_paths(root: Path) -> list[Path]:
    return sorted(
        {
            *root.rglob("*.yaml"),
            *root.rglob("*.yml"),
        },
        key=lambda path: path.as_posix(),
    )


def _index_task_tree(
    task_root: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    index: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    if not task_root.is_dir():
        return index, [f"task directory is missing: {task_root}"]
    for path in _yaml_paths(task_root):
        try:
            config = _load_yaml(path)
        except AuditError as exc:
            errors.append(str(exc))
            continue
        definitions: list[tuple[str, str]] = []
        task = config.get("task")
        group = config.get("group")
        if isinstance(task, str):
            definitions.append(("task", task))
        elif isinstance(task, list) and isinstance(group, str):
            definitions.append(("group", group))
        tags = config.get("tag")
        if isinstance(tags, str):
            definitions.append(("tag", tags))
        elif isinstance(tags, list):
            definitions.extend(
                ("tag", tag) for tag in tags if isinstance(tag, str)
            )
        for kind, name in definitions:
            index.setdefault(name, []).append(
                {
                    "kind": kind,
                    "path": path.resolve(strict=True).relative_to(
                        task_root.resolve(strict=True)
                    ).as_posix(),
                }
            )
    for entries in index.values():
        entries.sort(key=lambda item: (item["kind"], item["path"]))
    return index, errors


def _resolve_include(path: Path, reference: str, task_root: Path) -> Path:
    candidate = Path(reference)
    if not candidate.is_absolute():
        # lm-eval first tries the process cwd.  A formal task is invalid if
        # that could shadow the reviewed same-directory include.
        cwd_candidate = candidate.resolve()
        local_candidate = (path.parent / candidate).resolve()
        if cwd_candidate.is_file() and cwd_candidate != local_candidate:
            raise AuditError(
                f"cwd shadows task include {reference!r} from {path}"
            )
        candidate = local_candidate
    else:
        candidate = candidate.resolve()
    if not candidate.is_file():
        raise AuditError(f"missing include {reference!r} from {path}")
    try:
        candidate.relative_to(task_root.resolve(strict=True))
    except ValueError as exc:
        raise AuditError(f"task include escapes package root: {candidate}") from exc
    return candidate


def _effective_yaml(
    path: Path,
    task_root: Path,
    *,
    stack: tuple[Path, ...] = (),
) -> tuple[dict[str, Any], set[Path]]:
    resolved = path.resolve(strict=True)
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise AuditError(f"cyclic task YAML include: {cycle}")
    config = dict(_load_yaml(resolved))
    include_value = config.pop("include", None)
    if include_value is None:
        return config, {resolved}
    references = (
        [include_value] if isinstance(include_value, str) else include_value
    )
    if not isinstance(references, list) or not all(
        isinstance(item, str) for item in references
    ):
        raise AuditError(f"invalid include value in {resolved}: {include_value!r}")
    merged: dict[str, Any] = {}
    closure = {resolved}
    # Match lm_eval.utils.load_yaml_config: the first listed include wins.
    for reference in reversed(references):
        include_path = _resolve_include(resolved, reference, task_root)
        included, nested = _effective_yaml(
            include_path,
            task_root,
            stack=(*stack, resolved),
        )
        merged.update(included)
        closure.update(nested)
    merged.update(config)
    return merged, closure


def _index_effective_task_tree(
    task_root: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    """Index effective configs so include-inherited custom names are visible."""

    index: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    if not task_root.is_dir():
        return index, [f"task directory is missing: {task_root}"]
    resolved_root = task_root.resolve(strict=True)
    for path in _yaml_paths(task_root):
        try:
            config, _ = _effective_yaml(path, task_root)
            relative = path.resolve(strict=True).relative_to(
                resolved_root
            ).as_posix()
        except (AuditError, OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        definitions: list[tuple[str, str]] = []
        task = config.get("task")
        group = config.get("group")
        if isinstance(task, str):
            definitions.append(("task", task))
        elif isinstance(task, list) and isinstance(group, str):
            definitions.append(("group", group))
        tags = config.get("tag")
        if isinstance(tags, str):
            definitions.append(("tag", tags))
        elif isinstance(tags, list):
            definitions.extend(
                ("tag", tag) for tag in tags if isinstance(tag, str)
            )
        for kind, name in definitions:
            index.setdefault(name, []).append(
                {"kind": kind, "path": relative}
            )
    for entries in index.values():
        entries.sort(key=lambda item: (item["kind"], item["path"]))
    return index, errors


def _iter_tagged_scalars(value: Any) -> Iterable[_TaggedScalar]:
    if isinstance(value, _TaggedScalar):
        yield value
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _iter_tagged_scalars(key)
            yield from _iter_tagged_scalars(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tagged_scalars(nested)


def _module_defines_name(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ) and node.name == name:
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


def _resolve_function_closure(
    yaml_paths: Iterable[Path],
    *,
    package_root: Path,
    task_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve every task-closure ``!function`` without importing it."""

    errors: list[str] = []
    references: list[dict[str, Any]] = []
    module_paths: set[Path] = set()
    package_root = package_root.resolve(strict=True)
    task_root = task_root.resolve(strict=True)
    for yaml_path in sorted(
        {path.resolve(strict=True) for path in yaml_paths},
        key=lambda path: path.as_posix(),
    ):
        try:
            config = _load_yaml(yaml_path)
        except AuditError as exc:
            errors.append(str(exc))
            continue
        for tagged in _iter_tagged_scalars(config):
            yaml_relative = yaml_path.relative_to(task_root).as_posix()
            if tagged.tag_suffix != "function":
                errors.append(
                    f"unsupported custom YAML tag !{tagged.tag_suffix} in "
                    f"{yaml_relative}"
                )
                continue
            parts = str(tagged).split(".")
            if len(parts) < 2 or not all(parts):
                errors.append(
                    f"invalid !function reference {str(tagged)!r} in "
                    f"{yaml_relative}"
                )
                continue
            module_name = ".".join(parts[:-1])
            function_name = parts[-1]
            module_path = (
                yaml_path.parent / f"{module_name}.py"
            ).resolve()
            try:
                module_path.relative_to(package_root)
            except ValueError:
                errors.append(
                    f"!function module escapes lm_eval package: {module_path}"
                )
                continue
            if not module_path.is_file():
                errors.append(
                    f"missing !function module {module_path} for "
                    f"{str(tagged)!r}"
                )
                continue
            function_present = False
            try:
                tree = ast.parse(
                    module_path.read_text(encoding="utf-8"),
                    filename=str(module_path),
                )
                function_present = _module_defines_name(tree, function_name)
            except (OSError, UnicodeError, SyntaxError) as exc:
                errors.append(
                    f"cannot parse !function module {module_path}: {exc}"
                )
            if not function_present:
                errors.append(
                    f"!function target {function_name!r} is not defined at "
                    f"module top level in {module_path}"
                )
            module_paths.add(module_path)
            references.append(
                {
                    "yaml_path": yaml_relative,
                    "reference": str(tagged),
                    "module_path": module_path.relative_to(
                        package_root
                    ).as_posix(),
                    "function": function_name,
                    "function_present": function_present,
                }
            )
    references.sort(
        key=lambda item: (
            item["yaml_path"],
            item["reference"],
        )
    )
    try:
        modules = _hash_file_set(package_root, module_paths)
    except (AuditError, OSError) as exc:
        errors.append(f"cannot hash !function module closure: {exc}")
        modules = {"file_count": 0, "files": [], "combined_sha256": None}
    return {
        "reference_count": len(references),
        "references": references,
        "modules": modules,
    }, errors


def _lm_eval_python_closure(
    package_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Pin a conservative superset of all lm-eval Python execution code."""

    errors: list[str] = []
    package_root = package_root.resolve(strict=True)
    required: dict[str, bool] = {}
    for relative in REQUIRED_LM_EVAL_RUNTIME_FILES:
        present = (package_root / relative).is_file()
        required[relative] = present
        if not present:
            errors.append(f"required lm_eval runtime file is missing: {relative}")
    try:
        report = _hash_file_set(
            package_root,
            package_root.rglob("*.py"),
        )
    except (AuditError, OSError) as exc:
        errors.append(f"cannot hash lm_eval Python closure: {exc}")
        report = {"file_count": 0, "files": [], "combined_sha256": None}
    observed = report.get("combined_sha256")
    matches = observed == EXPECTED_LM_EVAL_PYTHON_CLOSURE_SHA256
    report.update(
        {
            "expected_combined_sha256": (
                EXPECTED_LM_EVAL_PYTHON_CLOSURE_SHA256
            ),
            "matches_expected": matches,
            "required_files": required,
        }
    )
    if not matches:
        errors.append(
            f"lm_eval Python closure SHA256 is {observed}, expected "
            f"{EXPECTED_LM_EVAL_PYTHON_CLOSURE_SHA256}"
        )
    return report, errors


def _single_definition(
    index: Mapping[str, list[dict[str, str]]],
    name: str,
    kind: str,
) -> dict[str, str]:
    definitions = list(index.get(name, ()))
    if len(definitions) != 1:
        raise AuditError(
            f"task name {name!r} resolves to {definitions!r}; expected exactly one"
        )
    if definitions[0]["kind"] != kind:
        raise AuditError(
            f"task name {name!r} resolved as {definitions[0]['kind']!r}, "
            f"expected {kind!r}"
        )
    return definitions[0]


def _version_string(value: Any) -> str | None:
    if isinstance(value, Mapping):
        nested = value.get("version_str")
        return None if nested is None else str(nested)
    return None if value is None else str(value)


def _metric_names(config: Mapping[str, Any], key: str = "metric_list") -> list[str]:
    metrics = config.get(key)
    if not isinstance(metrics, list):
        return []
    return [
        str(item["metric"])
        for item in metrics
        if isinstance(item, Mapping) and isinstance(item.get("metric"), str)
    ]


def _validate_leaf_config(
    task_name: str,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    metadata = config.get("metadata")
    task_version = (
        _version_string(metadata.get("version"))
        if isinstance(metadata, Mapping)
        else None
    )
    split = (
        config.get("test_split")
        if config.get("test_split") is not None
        else config.get("validation_split")
    )
    metrics = _metric_names(config)
    checks = {
        "dataset_path": config.get("dataset_path"),
        "dataset_name": config.get("dataset_name"),
        "evaluation_split": split,
        "filter_list": config.get("filter_list"),
        "metric_names": metrics,
        "metric_key": protocol["metric_key"],
        "num_fewshot": config.get("num_fewshot"),
        "task_version": task_version,
    }
    expected_pairs = (
        ("dataset_path", config.get("dataset_path"), protocol["dataset_path"]),
        (
            "dataset_name",
            config.get("dataset_name"),
            protocol["task_dataset_name"],
        ),
        ("evaluation_split", split, protocol["split"]),
        ("task_version", task_version, protocol["task_version"]),
    )
    for field, observed, expected in expected_pairs:
        if observed != expected:
            errors.append(
                f"{task_name}: {field}={observed!r}, expected {expected!r}"
            )
    if protocol["metric"] not in metrics:
        errors.append(
            f"{task_name}: metric {protocol['metric']!r} absent from {metrics!r}"
        )
    if config.get("filter_list") is not None:
        errors.append(
            f"{task_name}: filter_list must be absent/None so the metric "
            "filter suffix is exactly ',none'"
        )
    if config.get("num_fewshot") not in (None, 0):
        errors.append(
            f"{task_name}: num_fewshot must be None/0, got "
            f"{config.get('num_fewshot')!r}"
        )
    return checks, errors


def _scan_dataset_info(
    hf_cache_root: Path,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    dataset_root = hf_cache_root / "datasets"
    if not dataset_root.is_dir():
        return records, [f"Hugging Face dataset cache is missing: {dataset_root}"]
    for path in sorted(dataset_root.rglob("dataset_info.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"cannot parse dataset metadata {path}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"dataset metadata is not a mapping: {path}")
            continue
        records.append((path.resolve(strict=True), value))
    return records, errors


def _dataset_artifact(
    records: Sequence[tuple[Path, dict[str, Any]]],
    hf_cache_root: Path,
    *,
    label: str,
    dataset_name: str,
    config_name: str,
    dataset_version: str,
    builder_fingerprint: str,
    split: str,
    sample_count: int,
    expected_artifact_sha256: str | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    candidates = [
        (path, info)
        for path, info in records
        if info.get("dataset_name") == dataset_name
        and info.get("config_name") == config_name
    ]
    if len(candidates) != 1:
        errors.append(
            f"{label}: local dataset metadata resolution is ambiguous/missing "
            f"for {dataset_name}/{config_name}: "
            f"{[_relative_path(path, hf_cache_root) for path, _ in candidates]!r}"
        )
        return None, errors
    info_path, info = candidates[0]
    observed_version = _version_string(info.get("version"))
    observed_fingerprint = info_path.parent.name
    splits = info.get("splits")
    split_info = splits.get(split) if isinstance(splits, Mapping) else None
    observed_count = (
        split_info.get("num_examples")
        if isinstance(split_info, Mapping)
        else None
    )
    for field, observed, expected in (
        ("dataset_version", observed_version, dataset_version),
        ("builder_fingerprint", observed_fingerprint, builder_fingerprint),
        ("sample_count", observed_count, sample_count),
    ):
        if observed != expected:
            errors.append(
                f"{label}: {field}={observed!r}, expected {expected!r}"
            )
    arrow_files = sorted(info_path.parent.glob(f"*-{split}.arrow"))
    if len(arrow_files) != 1:
        errors.append(
            f"{label}: expected one local *-{split}.arrow artifact, got "
            f"{[path.name for path in arrow_files]!r}"
        )
    files = [info_path, *arrow_files]
    try:
        hashed = _hash_file_set(hf_cache_root, files)
    except (AuditError, OSError) as exc:
        errors.append(f"{label}: cannot hash local dataset artifacts: {exc}")
        hashed = {"file_count": 0, "files": [], "combined_sha256": None}
    observed_artifact_sha256 = hashed.get("combined_sha256")
    artifact_matches = (
        expected_artifact_sha256 is not None
        and observed_artifact_sha256 == expected_artifact_sha256
    )
    if expected_artifact_sha256 is None:
        errors.append(f"{label}: trusted artifact SHA256 is missing")
    elif not artifact_matches:
        errors.append(
            f"{label}: artifact SHA256={observed_artifact_sha256}, expected "
            f"{expected_artifact_sha256}"
        )
    return {
        "dataset_name": info.get("dataset_name"),
        "config_name": info.get("config_name"),
        "dataset_version": observed_version,
        "builder_fingerprint": observed_fingerprint,
        "split": split,
        "sample_count": observed_count,
        "artifacts": {
            **hashed,
            "expected_combined_sha256": expected_artifact_sha256,
            "matches_expected": artifact_matches,
        },
    }, errors


def _literal_default(node: ast.expr | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise AuditError(
            f"non-literal default/override at line {getattr(node, 'lineno', '?')}"
        ) from exc


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise AuditError(
            f"expected exactly one synchronous function {name!r}, got {len(matches)}"
        )
    return matches[0]


def _signature_defaults(function: ast.FunctionDef) -> dict[str, Any]:
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults: dict[str, Any] = {}
    start = len(positional) - len(function.args.defaults)
    for argument, default in zip(positional[start:], function.args.defaults):
        defaults[argument.arg] = _literal_default(default)
    for argument, default in zip(
        function.args.kwonlyargs,
        function.args.kw_defaults,
    ):
        if default is not None:
            defaults[argument.arg] = _literal_default(default)
    return defaults


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _module_literal_assignment(tree: ast.Module, name: str) -> Any:
    matches: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                matches.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            matches.append(node.value)
    if len(matches) != 1:
        raise AuditError(
            f"expected one module assignment for {name!r}, got {len(matches)}"
        )
    return _literal_default(matches[0])


def _keyword_node(call: ast.Call, name: str) -> ast.expr:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1:
        raise AuditError(
            f"call at line {call.lineno} must pass exactly one {name!r}"
        )
    return matches[0]


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _inspect_simple_evaluate(
    evaluator_path: Path,
    caller_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        evaluator_tree = ast.parse(
            evaluator_path.read_text(encoding="utf-8"),
            filename=str(evaluator_path),
        )
        simple = _function_node(evaluator_tree, "simple_evaluate")
        defaults = _signature_defaults(simple)
    except (OSError, UnicodeError, SyntaxError, AuditError) as exc:
        return {}, [f"cannot audit simple_evaluate: {exc}"]
    selected_defaults = {
        key: defaults.get(key) for key in EXPECTED_SIMPLE_EVALUATE_DEFAULTS
    }
    if selected_defaults != EXPECTED_SIMPLE_EVALUATE_DEFAULTS:
        errors.append(
            "lm_eval.simple_evaluate defaults changed: "
            f"{selected_defaults!r} != {EXPECTED_SIMPLE_EVALUATE_DEFAULTS!r}"
        )

    caller_effective: dict[str, Any] = {}
    project_semantics: dict[str, Any] = {}
    call_line = None
    try:
        caller_tree = ast.parse(
            caller_path.read_text(encoding="utf-8"),
            filename=str(caller_path),
        )
        qa_eval = _function_node(caller_tree, "qa_eval")
        paper_tasks = _module_literal_assignment(
            caller_tree,
            "PAPER_QA_TASKS",
        )
        project_semantics["paper_qa_tasks"] = list(paper_tasks)
        if tuple(paper_tasks) != EXPECTED_TASKS:
            errors.append(
                f"utils.eval_utils.PAPER_QA_TASKS={paper_tasks!r}, expected "
                f"{EXPECTED_TASKS!r}"
            )

        custom_task_dir = None
        for node in qa_eval.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "custom_task_dir"
                for target in node.targets
            ):
                custom_task_dir = _literal_default(node.value)
        project_semantics["custom_task_dir"] = custom_task_dir
        if custom_task_dir != "./datasets/lm_eval_configs/tasks":
            errors.append(
                f"qa_eval custom_task_dir={custom_task_dir!r}, expected "
                "'./datasets/lm_eval_configs/tasks'"
            )

        manager_calls = _calls_named(
            qa_eval,
            "lm_eval.tasks.TaskManager",
        )
        manager_modes: list[dict[str, Any]] = []
        if len(manager_calls) != 2:
            errors.append(
                f"qa_eval must have exactly two guarded TaskManager "
                f"constructions, got {len(manager_calls)}"
            )
        for manager_call in manager_calls:
            if any(keyword.arg is None for keyword in manager_call.keywords):
                errors.append("qa_eval TaskManager call uses **kwargs")
                continue
            keywords = {
                keyword.arg: keyword.value
                for keyword in manager_call.keywords
            }
            include_defaults = (
                _literal_default(keywords["include_defaults"])
                if "include_defaults" in keywords
                else None
            )
            include_path = (
                _dotted_name(keywords["include_path"])
                if "include_path" in keywords
                else None
            )
            manager_modes.append(
                {
                    "line": manager_call.lineno,
                    "include_defaults": include_defaults,
                    "include_path": include_path,
                }
            )
            if include_defaults is not True:
                errors.append(
                    "qa_eval TaskManager include_defaults must be True"
                )
            if include_path not in (None, "custom_task_dir"):
                errors.append(
                    f"qa_eval TaskManager include_path={include_path!r}, "
                    "expected custom_task_dir/absent"
                )
        if sorted(
            item["include_path"] or "" for item in manager_modes
        ) != ["", "custom_task_dir"]:
            errors.append(
                "qa_eval must construct one default-only and one "
                "custom-plus-default TaskManager"
            )
        project_semantics["task_manager_modes"] = manager_modes

        resolver_calls = _calls_named(qa_eval, "_resolve_paper_qa_tasks")
        if len(resolver_calls) != 1:
            errors.append(
                f"qa_eval must call _resolve_paper_qa_tasks exactly once, "
                f"got {len(resolver_calls)}"
            )
        else:
            resolver = resolver_calls[0]
            resolver_ok = (
                len(resolver.args) == 2
                and _dotted_name(resolver.args[0])
                == "lm_eval_utils.pattern_match"
                and _dotted_name(resolver.args[1])
                == "task_manager.all_tasks"
            )
            project_semantics["task_resolver_call_valid"] = resolver_ok
            if not resolver_ok:
                errors.append(
                    "qa_eval task resolver must consume pattern_match and "
                    "task_manager.all_tasks"
                )

        calls = [
            node
            for node in ast.walk(qa_eval)
            if isinstance(node, ast.Call)
            and _dotted_name(node.func) == "lm_eval.simple_evaluate"
        ]
        if len(calls) != 1:
            raise AuditError(
                f"qa_eval must contain exactly one lm_eval.simple_evaluate "
                f"call, got {len(calls)}"
            )
        call = calls[0]
        call_line = call.lineno
        if any(keyword.arg is None for keyword in call.keywords):
            raise AuditError("qa_eval simple_evaluate call uses **kwargs")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        if "tasks" not in keywords or "task_manager" not in keywords:
            raise AuditError(
                "qa_eval must explicitly pass tasks and task_manager"
            )
        task_node = _keyword_node(call, "tasks")
        task_manager_node = _keyword_node(call, "task_manager")
        task_binding_valid = (
            isinstance(task_node, ast.List)
            and len(task_node.elts) == 1
            and _is_name(task_node.elts[0], "task_name")
            and _is_name(task_manager_node, "task_manager")
        )
        project_semantics["simple_evaluate_task_binding_valid"] = (
            task_binding_valid
        )
        if not task_binding_valid:
            errors.append(
                "qa_eval simple_evaluate must use tasks=[task_name] and "
                "task_manager=task_manager"
            )
        matching_loops = [
            node
            for node in ast.walk(qa_eval)
            if isinstance(node, ast.For)
            and _is_name(node.target, "task_name")
            and _is_name(node.iter, "task_names")
            and call in list(ast.walk(node))
        ]
        if len(matching_loops) != 1:
            errors.append(
                "qa_eval simple_evaluate call must be inside the unique "
                "'for task_name in task_names' loop"
            )

        accuracy_function = _function_node(caller_tree, "_task_accuracy")
        accuracy_literals = {
            node.value
            for node in ast.walk(accuracy_function)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        metric_keys_present = {
            key: key in accuracy_literals
            for key in ("acc_norm,none", "acc,none")
        }
        project_semantics["result_metric_keys"] = metric_keys_present
        if not all(metric_keys_present.values()):
            errors.append(
                "utils.eval_utils._task_accuracy must recognize both "
                "'acc_norm,none' and 'acc,none'"
            )
        for name, expected in EXPECTED_SIMPLE_EVALUATE_DEFAULTS.items():
            observed = (
                _literal_default(keywords[name])
                if name in keywords
                else defaults.get(name)
            )
            caller_effective[name] = observed
            if observed != expected:
                errors.append(
                    f"qa_eval effective {name}={observed!r}, expected {expected!r}"
                )
    except (OSError, UnicodeError, SyntaxError, AuditError) as exc:
        errors.append(f"cannot audit qa_eval call: {exc}")
    return {
        "simple_evaluate_defaults": selected_defaults,
        "qa_eval_effective_arguments": caller_effective,
        "qa_eval_call_line": call_line,
        "effective_zero_shot": (
            selected_defaults.get("num_fewshot") is None
            and caller_effective.get("num_fewshot") is None
        ),
        "apply_chat_template": caller_effective.get("apply_chat_template"),
        "limit": caller_effective.get("limit"),
        "system_instruction": caller_effective.get("system_instruction"),
        "project_semantics": project_semantics,
    }, errors


def _calls_named(function: ast.AST, dotted_name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == dotted_name
    ]


def _keyword_literal(call: ast.Call, name: str) -> Any:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1:
        raise AuditError(
            f"call at line {call.lineno} must pass exactly one {name!r}"
        )
    return _literal_default(matches[0])


def _inspect_wikitext_callers(
    data_utils_path: Path,
    pipeline_path: Path,
    legacy_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    result: dict[str, Any] = {}
    try:
        data_tree = ast.parse(
            data_utils_path.read_text(encoding="utf-8"),
            filename=str(data_utils_path),
        )
        getter = _function_node(data_tree, "_get_wikitext2")
        load_calls = _calls_named(getter, "load_dataset")
        if len(load_calls) != 1:
            raise AuditError(
                f"_get_wikitext2 must contain one load_dataset call, "
                f"got {len(load_calls)}"
            )
        load_call = load_calls[0]
        if len(load_call.args) < 2:
            raise AuditError("_get_wikitext2 load_dataset lacks path/config")
        path_value = _literal_default(load_call.args[0])
        config_value = _literal_default(load_call.args[1])
        split_keywords = [
            keyword.value
            for keyword in load_call.keywords
            if keyword.arg == "split"
        ]
        split_forwarded = (
            len(split_keywords) == 1
            and isinstance(split_keywords[0], ast.Name)
            and split_keywords[0].id == "split"
        )
        trust_remote_code = _keyword_literal(load_call, "trust_remote_code")
        result["loader"] = {
            "dataset_path": path_value,
            "dataset_config": config_value,
            "split_forwarded": split_forwarded,
            "trust_remote_code": trust_remote_code,
            "line": load_call.lineno,
        }
        if path_value != WIKITEXT_PROTOCOL["dataset_path"]:
            errors.append(
                f"_get_wikitext2 dataset path is {path_value!r}, expected "
                f"{WIKITEXT_PROTOCOL['dataset_path']!r}"
            )
        if config_value != WIKITEXT_PROTOCOL["dataset_config"]:
            errors.append(
                f"_get_wikitext2 config is {config_value!r}, expected "
                f"{WIKITEXT_PROTOCOL['dataset_config']!r}"
            )
        if not split_forwarded:
            errors.append("_get_wikitext2 does not forward its split argument")
        if trust_remote_code is not True:
            errors.append("_get_wikitext2 trust_remote_code must be True")
    except (OSError, UnicodeError, SyntaxError, AuditError) as exc:
        errors.append(f"cannot audit WikiText-2 loader: {exc}")

    for label, path, function_name in (
        ("realq", pipeline_path, "_setup_eval"),
        ("legacy", legacy_path, "main"),
    ):
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            if label == "realq":
                scope: ast.AST = _function_node(tree, function_name)
            else:
                # ptq.py's evaluation path is inside its module-level main
                # function in the reviewed implementation.
                try:
                    scope = _function_node(tree, function_name)
                except AuditError:
                    scope = tree
            calls = _calls_named(scope, "data_utils.get_loaders")
            test_calls = []
            for call in calls:
                try:
                    if _keyword_literal(call, "split") == "test":
                        test_calls.append(call)
                except AuditError:
                    continue
            if len(test_calls) != 1:
                raise AuditError(
                    f"expected exactly one test-split get_loaders call, "
                    f"got {len(test_calls)}"
                )
            result[label] = {
                "split": "test",
                "line": test_calls[0].lineno,
            }
        except (OSError, UnicodeError, SyntaxError, AuditError) as exc:
            errors.append(f"cannot audit {label} WikiText-2 caller: {exc}")
    return result, errors


def _discover_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for key, distribution in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("lm-eval", "lm-eval"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
    ):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    try:
        versions["torch_runtime"] = str(
            importlib.import_module("torch").__version__
        )
    except Exception:  # noqa: BLE001 - absence is reported fail-closed below.
        versions["torch_runtime"] = None
    return versions


def _discover_lm_eval_root() -> Path:
    spec = importlib.util.find_spec("lm_eval")
    if spec is None:
        raise AuditError("lm_eval is not importable in the active interpreter")
    if spec.submodule_search_locations:
        locations = list(spec.submodule_search_locations)
        if len(locations) != 1:
            raise AuditError(
                f"lm_eval resolves to multiple package roots: {locations!r}"
            )
        return Path(locations[0]).resolve(strict=True)
    if spec.origin:
        return Path(spec.origin).resolve(strict=True).parent
    raise AuditError("cannot determine lm_eval package root")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _check_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    tasks = plan.get("paper_zero_shot_tasks")
    fixed = plan.get("fixed_numerics")
    final = plan.get("final")
    runtime_environment = plan.get("runtime_environment")
    if not isinstance(fixed, Mapping):
        fixed = {}
        errors.append("plan.fixed_numerics must be a mapping")
    if not isinstance(final, Mapping):
        final = {}
        errors.append("plan.final must be a mapping")
    if not isinstance(runtime_environment, Mapping):
        runtime_environment = {}
        errors.append("plan.runtime_environment must be a mapping")
    checks = {
        "paper_zero_shot_tasks": tasks,
        "dataset": fixed.get("dataset"),
        "eval_datasets": fixed.get("eval_datasets"),
        "eval_seq_len": fixed.get("eval_seq_len"),
        "final_lm_eval": final.get("lm_eval"),
        "final_lm_eval_batch_size": final.get("lm_eval_batch_size"),
        "offline_environment": {
            name: runtime_environment.get(name)
            for name in (
                "HF_DATASETS_OFFLINE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
    }
    if tasks != list(EXPECTED_TASKS):
        errors.append(
            f"paper_zero_shot_tasks={tasks!r}, expected {list(EXPECTED_TASKS)!r}"
        )
    if fixed.get("dataset") != "wikitext2":
        errors.append("fixed_numerics.dataset must be 'wikitext2'")
    if fixed.get("eval_datasets") != ["wikitext2"]:
        errors.append(
            "fixed_numerics.eval_datasets must be exactly ['wikitext2']"
        )
    if fixed.get("eval_seq_len") != WIKITEXT_PROTOCOL["eval_seq_len"]:
        errors.append(
            f"fixed_numerics.eval_seq_len must be "
            f"{WIKITEXT_PROTOCOL['eval_seq_len']}"
        )
    if final.get("lm_eval") is not True:
        errors.append("formal plan final.lm_eval must be true")
    batch_size = final.get("lm_eval_batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        errors.append("formal plan final.lm_eval_batch_size must be positive")
    for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if runtime_environment.get(name) != "1":
            errors.append(f"plan runtime_environment.{name} must be '1'")
    return checks, errors


def run_audit(
    plan_path: Path,
    *,
    repo_root: Path = ROOT,
    lm_eval_root: Path | None = None,
    installed_versions: Mapping[str, str | None] | None = None,
    python_prefix: Path | None = None,
    hf_cache_root: Path | None = None,
    wikitext_root: Path | None = None,
    process_cwd: Path | None = None,
    process_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the complete audit without loading a dataset or touching a network."""

    errors: list[str] = []
    repo_root = repo_root.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    expected_plan_path = (
        repo_root / "experiments" / "lowbit_activation" / "plan.json"
    ).resolve(strict=True)
    if plan_path != expected_plan_path:
        errors.append(
            f"audited plan path {plan_path} is not canonical "
            f"{expected_plan_path}"
        )
    active_cwd = Path(
        process_cwd if process_cwd is not None else Path.cwd()
    ).resolve(strict=True)
    if active_cwd != repo_root:
        errors.append(
            f"active cwd {active_cwd} must equal repository root {repo_root}; "
            "project dataset/task paths are cwd-relative"
        )
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read plan {plan_path}: {exc}") from exc
    if not isinstance(plan, dict):
        raise AuditError("experiment plan must be a JSON object")

    plan_checks, plan_errors = _check_plan(plan)
    errors.extend(plan_errors)
    plan_sha256 = _sha256_file(plan_path)

    plan_versions = plan.get("runtime_versions")
    if not isinstance(plan_versions, Mapping):
        plan_versions = {}
        errors.append("plan.runtime_versions must be a mapping")
    actual_versions = dict(
        installed_versions
        if installed_versions is not None
        else _discover_versions()
    )
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items():
        if (
            name in PLAN_RUNTIME_VERSION_KEYS or name in plan_versions
        ) and plan_versions.get(name) != expected:
            errors.append(
                f"plan runtime version {name}={plan_versions.get(name)!r}, "
                f"expected {expected!r}"
            )
        if actual_versions.get(name) != expected:
            errors.append(
                f"installed runtime version {name}={actual_versions.get(name)!r}, "
                f"expected {expected!r}"
            )

    python_environment = plan.get("python_environment")
    if not isinstance(python_environment, Mapping):
        python_environment = {}
        errors.append("plan.python_environment must be a mapping")
    planned_venv_value = python_environment.get("venv")
    planned_venv = (
        Path(str(planned_venv_value)).resolve()
        if planned_venv_value is not None
        else None
    )
    active_prefix = Path(
        python_prefix if python_prefix is not None else sys.prefix
    ).resolve()
    if python_environment.get("activation_required") is not True:
        errors.append("plan must require venv activation")
    if planned_venv is None or active_prefix != planned_venv:
        errors.append(
            f"active Python prefix {str(active_prefix)!r} does not equal "
            f"planned venv {str(planned_venv)!r}"
        )

    try:
        package_root = (
            lm_eval_root.resolve(strict=True)
            if lm_eval_root is not None
            else _discover_lm_eval_root()
        )
    except (AuditError, OSError) as exc:
        package_root = None
        errors.append(str(exc))
    if package_root is not None and planned_venv is not None:
        try:
            package_root.relative_to(planned_venv.resolve(strict=True))
        except (FileNotFoundError, ValueError):
            errors.append(
                f"lm_eval package {package_root} is outside planned venv "
                f"{planned_venv}"
            )

    runtime_environment = plan.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        runtime_environment = {}
    active_environment = dict(
        process_environment
        if process_environment is not None
        else os.environ
    )
    actual_offline_environment = {
        name: active_environment.get(name)
        for name in (
            "HF_DATASETS_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        )
    }
    for name, value in actual_offline_environment.items():
        if value != "1":
            errors.append(
                f"active environment {name}={value!r}, expected '1'"
            )
    hf_home = runtime_environment.get("HF_HOME")
    if not isinstance(hf_home, str) or not hf_home:
        errors.append("plan.runtime_environment.HF_HOME is missing")
        planned_hf_cache_root = repo_root / "__missing_hf_home__"
    else:
        candidate = Path(hf_home)
        planned_hf_cache_root = (
            candidate if candidate.is_absolute() else repo_root / candidate
        )
    planned_hf_cache_root = planned_hf_cache_root.resolve()
    if hf_cache_root is None:
        hf_cache_root = planned_hf_cache_root
    hf_cache_root = hf_cache_root.resolve()
    if hf_cache_root != planned_hf_cache_root:
        errors.append(
            f"audited cache root {hf_cache_root} does not equal planned "
            f"HF_HOME {planned_hf_cache_root}"
        )
    active_hf_home = active_environment.get("HF_HOME")
    if active_hf_home is None:
        errors.append("active environment HF_HOME is missing")
    elif Path(active_hf_home).resolve() != hf_cache_root:
        errors.append(
            f"active HF_HOME {Path(active_hf_home).resolve()} does not equal "
            f"audited cache root {hf_cache_root}"
        )
    if wikitext_root is None:
        wikitext_root = (
            repo_root / "datasets" / "wikitext" / "wikitext-2-raw-v1"
        )
    wikitext_root = wikitext_root.resolve()
    expected_wikitext_root = (
        repo_root / "datasets" / "wikitext" / "wikitext-2-raw-v1"
    ).resolve()
    if wikitext_root != expected_wikitext_root:
        errors.append(
            f"audited WikiText-2 root {wikitext_root} is not canonical "
            f"{expected_wikitext_root}"
        )

    evaluator_report: dict[str, Any] = {}
    task_report: dict[str, Any] = {
        "requested": list(EXPECTED_TASKS),
        "resolved": {},
        "tasks": {},
        "ceval": {},
        "custom_task_overrides": {},
        "task_config_closure": {},
        "python_execution_closure": {},
        "function_execution_closure": {},
    }
    semantics_report: dict[str, Any] = {}
    task_closure_paths: set[Path] = set()
    if package_root is not None:
        evaluator_path = package_root / "evaluator.py"
        task_root = package_root / "tasks"
        python_closure, python_closure_errors = _lm_eval_python_closure(
            package_root
        )
        task_report["python_execution_closure"] = python_closure
        errors.extend(python_closure_errors)
        try:
            evaluator_record = _file_record(evaluator_path, package_root)
            evaluator_report = {
                **evaluator_record,
                "expected_sha256": EXPECTED_EVALUATOR_SHA256,
                "matches_expected": (
                    evaluator_record["sha256"] == EXPECTED_EVALUATOR_SHA256
                ),
            }
            if not evaluator_report["matches_expected"]:
                errors.append(
                    f"lm_eval evaluator.py SHA256 is "
                    f"{evaluator_record['sha256']}, expected "
                    f"{EXPECTED_EVALUATOR_SHA256}"
                )
        except (AuditError, OSError) as exc:
            errors.append(f"cannot hash lm_eval evaluator.py: {exc}")

        index, index_errors = _index_task_tree(task_root)
        errors.extend(f"lm_eval task index: {item}" for item in index_errors)
        resolved_paths: dict[str, Path] = {}
        for name in EXPECTED_TASKS:
            kind = "group" if name == "ceval-valid" else "task"
            try:
                definition = _single_definition(index, name, kind)
                resolved_path = task_root / definition["path"]
                resolved_paths[name] = resolved_path
                task_report["resolved"][name] = definition
            except AuditError as exc:
                errors.append(str(exc))

        ceval_leaf_paths: dict[str, Path] = {}
        for leaf in CEVAL_LEAVES:
            try:
                definition = _single_definition(index, leaf, "task")
                ceval_leaf_paths[leaf] = task_root / definition["path"]
            except AuditError as exc:
                errors.append(str(exc))

        caller_path = repo_root / "utils" / "eval_utils.py"
        if evaluator_path.is_file() and caller_path.is_file():
            semantics_report, semantic_errors = _inspect_simple_evaluate(
                evaluator_path,
                caller_path,
            )
            errors.extend(semantic_errors)

        for name, protocol in TASK_PROTOCOLS.items():
            path = resolved_paths.get(name)
            if path is None:
                continue
            try:
                effective, closure = _effective_yaml(path, task_root)
                task_closure_paths.update(closure)
                checks, config_errors = _validate_leaf_config(
                    name,
                    effective,
                    protocol,
                )
                errors.extend(config_errors)
                task_report["tasks"][name] = {
                    "config_path": path.resolve().relative_to(
                        task_root.resolve()
                    ).as_posix(),
                    **checks,
                    "expected_split": protocol["split"],
                    "expected_sample_count": protocol["sample_count"],
                    "expected_task_version": protocol["task_version"],
                    "expected_metric_key": protocol["metric_key"],
                }
            except (AuditError, OSError) as exc:
                errors.append(f"{name}: cannot resolve task config: {exc}")

        group_path = resolved_paths.get("ceval-valid")
        ceval_group: dict[str, Any] = {}
        if group_path is not None:
            try:
                group_config, closure = _effective_yaml(group_path, task_root)
                task_closure_paths.update(closure)
                group_tasks = group_config.get("task")
                group_metadata = group_config.get("metadata")
                group_version = (
                    _version_string(group_metadata.get("version"))
                    if isinstance(group_metadata, Mapping)
                    else None
                )
                aggregate_metrics = _metric_names(
                    group_config,
                    key="aggregate_metric_list",
                )
                ceval_group = {
                    "config_path": group_path.resolve().relative_to(
                        task_root.resolve()
                    ).as_posix(),
                    "leaves": group_tasks,
                    "leaf_count": len(group_tasks)
                    if isinstance(group_tasks, list)
                    else None,
                    "sample_count": sum(CEVAL_SAMPLE_COUNTS.values()),
                    "task_version": group_version,
                    "aggregate_metric_names": aggregate_metrics,
                    "metric_key": "acc_norm,none",
                }
                if group_tasks != list(CEVAL_LEAVES):
                    errors.append(
                        "ceval-valid must resolve to the exact reviewed 52-leaf "
                        f"list; got {group_tasks!r}"
                    )
                if group_version != "2.0":
                    errors.append(
                        f"ceval-valid version={group_version!r}, expected '2.0'"
                    )
                if "acc_norm" not in aggregate_metrics:
                    errors.append(
                        "ceval-valid aggregate metric must include 'acc_norm'"
                    )
            except (AuditError, OSError) as exc:
                errors.append(f"cannot resolve ceval-valid group: {exc}")

        ceval_leaf_report: dict[str, Any] = {}
        ceval_protocol_base = {
            "dataset_path": "ceval/ceval-exam",
            "task_version": "2.0",
            "split": "val",
            "metric": "acc_norm",
            "metric_key": "acc_norm,none",
        }
        for leaf in CEVAL_LEAVES:
            path = ceval_leaf_paths.get(leaf)
            if path is None:
                continue
            suffix = leaf.removeprefix("ceval-valid_")
            protocol = {
                **ceval_protocol_base,
                "task_dataset_name": suffix,
            }
            try:
                effective, closure = _effective_yaml(path, task_root)
                task_closure_paths.update(closure)
                checks, config_errors = _validate_leaf_config(
                    leaf,
                    effective,
                    protocol,
                )
                errors.extend(config_errors)
                ceval_leaf_report[leaf] = {
                    "config_path": path.resolve().relative_to(
                        task_root.resolve()
                    ).as_posix(),
                    **checks,
                    "expected_sample_count": CEVAL_SAMPLE_COUNTS[suffix],
                }
            except (AuditError, OSError) as exc:
                errors.append(f"{leaf}: cannot resolve task config: {exc}")
        task_report["ceval"] = {
            **ceval_group,
            "leaf_protocols": ceval_leaf_report,
        }

        function_closure, function_closure_errors = (
            _resolve_function_closure(
                task_closure_paths,
                package_root=package_root,
                task_root=task_root,
            )
        )
        task_report["function_execution_closure"] = function_closure
        errors.extend(function_closure_errors)

        if task_closure_paths:
            try:
                closure_report = _hash_file_set(
                    task_root,
                    task_closure_paths,
                )
                closure_report.update(
                    {
                        "expected_combined_sha256": (
                            EXPECTED_TASK_CLOSURE_SHA256
                        ),
                        "matches_expected": (
                            closure_report["combined_sha256"]
                            == EXPECTED_TASK_CLOSURE_SHA256
                        ),
                    }
                )
                task_report["task_config_closure"] = closure_report
                if not closure_report["matches_expected"]:
                    errors.append(
                        f"lm_eval task closure SHA256 is "
                        f"{closure_report['combined_sha256']}, expected "
                        f"{EXPECTED_TASK_CLOSURE_SHA256}"
                    )
            except (AuditError, OSError) as exc:
                errors.append(f"cannot hash task config closure: {exc}")

        custom_root = (
            repo_root / "datasets" / "lm_eval_configs" / "tasks"
        )
        custom_index: dict[str, list[dict[str, str]]] = {}
        custom_errors: list[str] = []
        if custom_root.exists():
            custom_index, custom_errors = _index_effective_task_tree(
                custom_root
            )
            errors.extend(f"custom task index: {item}" for item in custom_errors)
        protected_names = {*EXPECTED_TASKS, *CEVAL_LEAVES}
        collisions = {
            name: custom_index[name]
            for name in sorted(protected_names & custom_index.keys())
        }
        if collisions:
            errors.append(
                f"custom task definitions override formal protocol names: "
                f"{collisions!r}"
            )
        task_report["custom_task_overrides"] = {
            "directory": _relative_path(custom_root, repo_root),
            "directory_exists": custom_root.is_dir(),
            "definition_count": sum(len(items) for items in custom_index.values()),
            "collisions": collisions,
            "absent": not collisions,
        }

    expected_dataset_labels = {
        *TASK_PROTOCOLS,
        *CEVAL_LEAVES,
        "wikitext2",
    }
    if set(EXPECTED_DATASET_ARTIFACT_SHA256) != expected_dataset_labels:
        errors.append(
            "trusted dataset artifact digest keys do not equal the exact "
            "formal split set"
        )
    dataset_records, dataset_scan_errors = _scan_dataset_info(hf_cache_root)
    errors.extend(dataset_scan_errors)
    dataset_hash_inputs: list[dict[str, Any]] = []
    for name, protocol in TASK_PROTOCOLS.items():
        artifact, artifact_errors = _dataset_artifact(
            dataset_records,
            hf_cache_root,
            label=name,
            dataset_name=protocol["cache_dataset_name"],
            config_name=protocol["cache_config_name"],
            dataset_version=protocol["dataset_version"],
            builder_fingerprint=protocol["builder_fingerprint"],
            split=protocol["split"],
            sample_count=protocol["sample_count"],
            expected_artifact_sha256=(
                EXPECTED_DATASET_ARTIFACT_SHA256.get(name)
            ),
        )
        errors.extend(artifact_errors)
        if artifact is not None:
            task_report["tasks"].setdefault(name, {})["dataset"] = artifact
            dataset_hash_inputs.append({"task": name, **artifact["artifacts"]})

    ceval_dataset_report: dict[str, Any] = {}
    for leaf in CEVAL_LEAVES:
        suffix = leaf.removeprefix("ceval-valid_")
        artifact, artifact_errors = _dataset_artifact(
            dataset_records,
            hf_cache_root,
            label=leaf,
            dataset_name="ceval-exam",
            config_name=suffix,
            dataset_version="0.0.0",
            builder_fingerprint=_COMMON_DATASET_FINGERPRINTS["ceval"],
            split="val",
            sample_count=CEVAL_SAMPLE_COUNTS[suffix],
            expected_artifact_sha256=(
                EXPECTED_DATASET_ARTIFACT_SHA256.get(leaf)
            ),
        )
        errors.extend(artifact_errors)
        if artifact is not None:
            ceval_dataset_report[leaf] = artifact
            dataset_hash_inputs.append({"task": leaf, **artifact["artifacts"]})
    task_report["ceval"]["datasets"] = ceval_dataset_report

    wikitext_cache, wikitext_cache_errors = _dataset_artifact(
        dataset_records,
        hf_cache_root,
        label="wikitext2",
        dataset_name=WIKITEXT_PROTOCOL["cache_dataset_name"],
        config_name=WIKITEXT_PROTOCOL["cache_config_name"],
        dataset_version=WIKITEXT_PROTOCOL["dataset_version"],
        builder_fingerprint=WIKITEXT_PROTOCOL["builder_fingerprint"],
        split=WIKITEXT_PROTOCOL["split"],
        sample_count=WIKITEXT_PROTOCOL["sample_count"],
        expected_artifact_sha256=(
            EXPECTED_DATASET_ARTIFACT_SHA256.get("wikitext2")
        ),
    )
    errors.extend(wikitext_cache_errors)
    if wikitext_cache is not None:
        dataset_hash_inputs.append(
            {"task": "wikitext2", **wikitext_cache["artifacts"]}
        )
    dataset_cache_fingerprint = _stable_digest(
        sorted(dataset_hash_inputs, key=lambda item: item["task"])
    )

    wikitext_source: dict[str, Any] = {
        "root": str(wikitext_root),
        "expected_files": EXPECTED_WIKITEXT_FILES,
    }
    try:
        actual_parquet = sorted(wikitext_root.glob("*.parquet"))
        actual_names = [path.name for path in actual_parquet]
        if actual_names != sorted(EXPECTED_WIKITEXT_FILES):
            errors.append(
                f"WikiText-2 parquet set={actual_names!r}, expected "
                f"{sorted(EXPECTED_WIKITEXT_FILES)!r}"
            )
        source_hash = _hash_file_set(wikitext_root, actual_parquet)
        source_hash.update(
            {
                "expected_combined_sha256": (
                    EXPECTED_WIKITEXT_COMBINED_SHA256
                ),
                "matches_expected": (
                    source_hash["combined_sha256"]
                    == EXPECTED_WIKITEXT_COMBINED_SHA256
                ),
            }
        )
        wikitext_source["artifacts"] = source_hash
        for item in source_hash["files"]:
            expected = EXPECTED_WIKITEXT_FILES.get(item["relative_path"])
            if expected is None:
                continue
            if {
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            } != expected:
                errors.append(
                    f"WikiText-2 file {item['relative_path']} identity changed"
                )
        if not source_hash["matches_expected"]:
            errors.append(
                f"WikiText-2 source SHA256 is "
                f"{source_hash['combined_sha256']}, expected "
                f"{EXPECTED_WIKITEXT_COMBINED_SHA256}"
            )
    except (AuditError, OSError) as exc:
        errors.append(f"cannot hash local WikiText-2 source: {exc}")

    project_paths = {
        "data_utils": repo_root / "utils" / "data_utils.py",
        "eval_utils": repo_root / "utils" / "eval_utils.py",
        "realq_pipeline": repo_root / "realq" / "pipeline.py",
        "legacy_ptq": repo_root / "ptq.py",
    }
    project_sources: dict[str, Any] = {}
    if set(project_paths) != set(EXPECTED_PROJECT_SOURCE_SHA256):
        errors.append(
            "trusted project evaluation source digest keys are incomplete"
        )
    for name, path in project_paths.items():
        try:
            record = _file_record(path, repo_root)
            expected = EXPECTED_PROJECT_SOURCE_SHA256.get(name)
            matches = (
                expected is not None and record["sha256"] == expected
            )
            project_sources[name] = {
                **record,
                "expected_sha256": expected,
                "matches_expected": matches,
            }
            if not matches:
                errors.append(
                    f"project evaluation source {name} SHA256 is "
                    f"{record['sha256']}, expected {expected}"
                )
        except (AuditError, OSError) as exc:
            errors.append(f"cannot hash project evaluation source {name}: {exc}")
    if all(path.is_file() for path in (
        project_paths["data_utils"],
        project_paths["realq_pipeline"],
        project_paths["legacy_ptq"],
    )):
        caller_report, caller_errors = _inspect_wikitext_callers(
            project_paths["data_utils"],
            project_paths["realq_pipeline"],
            project_paths["legacy_ptq"],
        )
        errors.extend(caller_errors)
    else:
        caller_report = {}

    wikitext_report = {
        "protocol": WIKITEXT_PROTOCOL,
        "plan": {
            "dataset": plan_checks.get("dataset"),
            "eval_datasets": plan_checks.get("eval_datasets"),
            "eval_seq_len": plan_checks.get("eval_seq_len"),
        },
        "call_sites": caller_report,
        "local_source": wikitext_source,
        "local_cache": wikitext_cache,
    }

    task_closure_sha256 = task_report.get(
        "task_config_closure", {}
    ).get("combined_sha256")
    evaluator_sha256 = evaluator_report.get("sha256")
    python_closure_sha256 = task_report.get(
        "python_execution_closure", {}
    ).get("combined_sha256")
    function_closure_sha256 = task_report.get(
        "function_execution_closure", {}
    ).get("modules", {}).get("combined_sha256")
    wikitext_sha256 = wikitext_source.get("artifacts", {}).get(
        "combined_sha256"
    )
    protocol_fingerprint_input = {
        "plan_sha256": plan_sha256,
        "runtime_versions": dict(sorted(actual_versions.items())),
        "evaluator_sha256": evaluator_sha256,
        "lm_eval_python_closure_sha256": python_closure_sha256,
        "lm_eval_function_closure_sha256": function_closure_sha256,
        "task_config_closure_sha256": task_closure_sha256,
        "dataset_cache_sha256": dataset_cache_fingerprint,
        "wikitext_source_sha256": wikitext_sha256,
        "project_source_sha256": {
            name: item["sha256"]
            for name, item in sorted(project_sources.items())
        },
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "network_access": False,
        "dataset_loading": False,
        "plan": {
            "path": _relative_path(plan_path, repo_root),
            "sha256": plan_sha256,
            "checks": plan_checks,
        },
        "runtime": {
            "expected_versions": EXPECTED_RUNTIME_VERSIONS,
            "plan_versions": dict(plan_versions),
            "installed_versions": actual_versions,
            "planned_venv": str(planned_venv),
            "active_python_prefix": str(active_prefix),
            "active_cwd": str(active_cwd),
            "active_offline_environment": actual_offline_environment,
            "active_hf_home": active_hf_home,
            "lm_eval_package_root": (
                str(package_root) if package_root is not None else None
            ),
            "hf_cache_root": str(hf_cache_root),
        },
        "evaluation_semantics": semantics_report,
        "lm_eval": {
            "evaluator": evaluator_report,
            **task_report,
        },
        "dataset_cache_combined_sha256": dataset_cache_fingerprint,
        "wikitext2": wikitext_report,
        "project_evaluation_sources": project_sources,
        "protocol_fingerprint_input": protocol_fingerprint_input,
        "protocol_fingerprint_sha256": _stable_digest(
            protocol_fingerprint_input
        ),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="atomically write the same canonical JSON emitted to stdout",
    )
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.plan)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed as JSON.
        report = {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "network_access": False,
            "dataset_loading": False,
        }
    if args.write_manifest is not None:
        try:
            _atomic_write(args.write_manifest, report)
        except Exception as exc:  # noqa: BLE001
            report = {
                **report,
                "valid": False,
                "errors": sorted(
                    {
                        *report.get("errors", []),
                        f"manifest write failed: {type(exc).__name__}: {exc}",
                    }
                ),
            }
    sys.stdout.buffer.write(_canonical_bytes(report))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
