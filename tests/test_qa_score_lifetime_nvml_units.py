import pytest
from tools.audit_qa_score_lifetime_component import memory_mib


@pytest.mark.parametrize('text,expected',[(' 16692 MiB',16692),('10802',10802),(' 2 MiB ',2)])
def test_nvidia_csv_units(text,expected):
    assert memory_mib(text)==expected


@pytest.mark.parametrize('text',['N/A','19.1 GB','-2',''])
def test_missing_or_wrong_units_refused(text):
    with pytest.raises(ValueError): memory_mib(text)
