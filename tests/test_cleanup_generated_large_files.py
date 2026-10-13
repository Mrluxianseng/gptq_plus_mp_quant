import copy
import os
from pathlib import Path

import pytest

from tools import cleanup_generated_large_files as cleanup


def fixture(tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    (root/'ptq.py').write_text('fixture')
    (root/'process_args.py').write_text('fixture')
    def file(relative, data=b'x'*2048):
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    return root, file


def test_cleanup_preserves_original_models_data_logs_and_invalidates_prepared(tmp_path):
    root, file = fixture(tmp_path)
    fisher = file('cache/old/Qwen3_mainFisher_world4_rank0.pt')
    logits = file('cache/ref_logits/Qwen3-0.6B_wikitext2_test_2048.cache')
    weights = file('outputs/old/static_cache/_prepared_checkpoints/Qwen3_rot/model.safetensors', b'weights')
    marker = file(str(weights.parent.relative_to(root))+'/_SUCCESS', b'ok')
    protected = [file('llmModels/Qwen3/model.safetensors'), file('outputs/final-model/model.safetensors'),
                 file('outputs/Qwen3-quantized.pt'), file('cache/tokens/data.pt'),
                 file('cache/datasets/training.npy'), file('datasets/input.npy'),
                 file('env/lib/library.bin'), file('outputs/old/memory_profile.jsonl'), file('outputs/old/stage0.log')]
    plan = cleanup.make_plan(root, 1024)
    assert {r['path'] for r in plan['files']} == {str(fisher), str(logits), str(weights), str(marker)}
    audit = cleanup.apply_plan(plan, process_check=lambda: [])
    assert audit.is_file()
    assert not fisher.exists() and not logits.exists() and not weights.exists() and not marker.exists()
    assert all(p.is_file() and p.stat().st_size > 0 for p in protected)


def test_hardlink_release_estimate_counts_unique_data_and_preserves_external_link(tmp_path):
    root, file = fixture(tmp_path)
    a = file('cache/ref_logits/a.pt')
    b = root/'outputs/b.pt'
    b.parent.mkdir()
    os.link(a, b)
    plan = cleanup.make_plan(root, 1024)
    assert len(plan['files']) == 2 and plan['estimated_reclaimed_bytes'] == a.stat().st_blocks*512
    protected = root/'llmModels/protected.pt'
    protected.parent.mkdir()
    os.link(a, protected)
    plan = cleanup.make_plan(root, 1024)
    assert plan['estimated_reclaimed_bytes'] == 0
    cleanup.apply_plan(plan, process_check=lambda: [])
    assert protected.read_bytes() == b'x'*2048


def test_symlinks_never_delete_external_targets_or_traverse_directories(tmp_path):
    root, file = fixture(tmp_path)
    protected = file('datasets/training.pt')
    (root/'outputs').mkdir()
    directory = root/'outputs/data-link'
    directory.symlink_to(protected.parent, target_is_directory=True)
    link = root/'outputs/generated_capture.pt'
    link.symlink_to(protected)
    plan = cleanup.make_plan(root, 1024)
    assert [r['path'] for r in plan['files']] == [str(link)]
    cleanup.apply_plan(plan, process_check=lambda: [])
    assert protected.read_bytes() == b'x'*2048 and directory.is_symlink()


def test_changed_file_and_tampered_scope_reject_before_any_deletion(tmp_path):
    root, file = fixture(tmp_path)
    a, b = file('outputs/a.pt'), file('outputs/b.pt')
    plan = cleanup.make_plan(root, 1024)
    b.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='nothing deleted'):
        cleanup.apply_plan(plan, process_check=lambda: [])
    assert a.exists()
    plan = cleanup.make_plan(root, 1024)
    bad = copy.deepcopy(plan)
    bad['files'][0]['path'] = str(file('llmModels/protected.pt'))
    with pytest.raises(ValueError, match='outside allowed'):
        cleanup.apply_plan(bad, process_check=lambda: [])
    assert a.exists()


def test_running_experiments_block_cleanup_and_new_files_not_in_plan_are_preserved(tmp_path):
    root, file = fixture(tmp_path)
    a = file('outputs/old.pt')
    plan = cleanup.make_plan(root, 1024)
    with pytest.raises(RuntimeError, match='still running'):
        cleanup.apply_plan(plan, process_check=lambda: ['123 python tools/run_storage_method_formal.py --run'])
    assert a.exists()
    b = file('outputs/new.pt')
    cleanup.apply_plan(plan, process_check=lambda: [])
    assert not a.exists() and b.exists()


def test_process_guard_matches_workers_not_cleanup_tool():
    assert cleanup.ACTIVE.search('123 /output/env/bin/python -u tools/run_storage_method_formal.py --run')
    assert cleanup.ACTIVE.search('456 python -m torch.distributed.run --nproc_per_node=4 entry.py')
    assert cleanup.ACTIVE.search('457 python outputs/entry.py baseline-profile --model x')
    assert not cleanup.ACTIVE.search('500 python tools/cleanup_generated_large_files.py --apply plan.json')
