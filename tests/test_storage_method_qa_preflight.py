from types import SimpleNamespace

import pytest

from tools.storage_method_qa_preflight import check_datasets, dataset_cache_directory
from utils.eval_utils import PAPER_QA_TASKS


def task(empty=False):
    docs = [] if empty else [{'text': 'fixture'}]
    return SimpleNamespace(dataset={'validation': docs}, has_test_docs=lambda: False,
        has_validation_docs=lambda: True, validation_docs=lambda: docs)


def test_exact_ten_tasks_and_nested_ceval_leaves_load_without_a_model():
    calls = []
    def load(names, **kwargs):
        assert kwargs['task_manager'] == 'manager'
        calls.extend(names)
        if names == ['ceval-valid']:
            return {'group': {'leaf1': ('group-label', task()), 'leaf2': task()}}
        return {names[0]: task()}
    report = check_datasets('manager', PAPER_QA_TASKS, load)
    assert report['valid'] and calls == list(PAPER_QA_TASKS)
    assert len(report['tasks'][6]['leaves']) == 2
    assert all(row['leaves'][0]['evaluation_documents'] == 1 for row in report['tasks'])


@pytest.mark.parametrize('fault', ['offline', 'empty', 'no_leaf'])
def test_missing_dataset_rejects_preflight_and_keeps_all_failures(fault):
    calls = []
    def load(names, **kwargs):
        calls.extend(names)
        if names == ['piqa']:
            if fault == 'offline': raise RuntimeError('OfflineModeIsEnabled: physicaliqa-train-dev.zip')
            if fault == 'empty': return {'piqa': task(empty=True)}
            return {'group': {}}
        return {names[0]: task()}
    report = check_datasets('manager', PAPER_QA_TASKS, load)
    assert not report['valid'] and len(report['errors']) == 1
    assert report['errors'][0]['task'] == 'piqa'
    assert len(report['tasks']) == 9 and calls == list(PAPER_QA_TASKS)


def test_existing_server_data_cache_selected_without_copy_and_explicit_setting_respected(tmp_path):
    repo = tmp_path/'gptq_plus'
    repo.mkdir()
    cache = tmp_path/'hf_cache/datasets'
    (cache/'piqa').mkdir(parents=True)
    assert dataset_cache_directory(repo, {}) == str(cache.resolve())
    explicit = tmp_path/'user-selected-cache'
    assert dataset_cache_directory(repo, {'HF_DATASETS_CACHE': str(explicit)}) == str(explicit.resolve())
    assert not explicit.exists()


def test_no_known_dataset_cache_keeps_library_default(tmp_path):
    assert dataset_cache_directory(tmp_path/'repo', {}) is None
