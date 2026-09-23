import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_inference import atomic_json, read_complete

def test_complete_run_is_reused_and_partial_run_rejected(tmp_path):
    path = tmp_path/'run.json'
    sig = {'repeats': 3, 'threads': 4}
    assert read_complete(path, sig) is None
    record = dict(status='complete', signature=sig, seconds=[.1,.2,.3])
    atomic_json(path, record)
    assert read_complete(path, sig) == record
    assert not path.with_suffix('.json.tmp').exists()
    atomic_json(path, dict(record, seconds=[.1,.2]))
    assert read_complete(path, sig) is None
    atomic_json(path, dict(record, status='incomplete'))
    assert read_complete(path, sig) is None

def test_changed_workload_cannot_mix_with_completed_measurements(tmp_path):
    path = tmp_path/'run.json'
    atomic_json(path, dict(status='complete', signature={'repeats':3, 'threads':4}, seconds=[.1,.2,.3]))
    with pytest.raises(ValueError, match='differ'):
        read_complete(path, {'repeats':3, 'threads':2})

def test_uncommitted_temp_file_is_not_a_completed_run(tmp_path):
    path = tmp_path/'run.json'
    path.with_suffix('.json.tmp').write_text('{')
    assert read_complete(path, {'repeats':3}) is None
