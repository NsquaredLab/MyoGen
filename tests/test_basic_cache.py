"""Synthetic contract-only cache identity, integrity, and atomicity tests."""
import json
import numpy as np
import pytest

from myogen.simulator.core.emg.runtime.cache import DiskCache, fingerprint


def test_fingerprint_tracks_contents_dtype_shape_and_all_dependencies():
    value = np.arange(8, dtype=np.float64)
    key = fingerprint(value, {"rate": 1000, "electrode": [1, 2, 3]})
    assert key != fingerprint(value.astype(np.float32), {"rate": 1000, "electrode": [1, 2, 3]})
    assert key != fingerprint(value.reshape(2, 4), {"rate": 1000, "electrode": [1, 2, 3]})
    assert key != fingerprint(value, {"rate": 1001, "electrode": [1, 2, 3]})
    assert key != fingerprint(value, {"rate": 1000, "electrode": [1, 2, 4]})
    value[0] = 9
    assert key != fingerprint(value, {"rate": 1000, "electrode": [1, 2, 3]})
    with pytest.raises(TypeError):
        fingerprint(object())


def test_cache_checksum_and_no_pickle(tmp_path):
    cache = DiskCache(tmp_path)
    value = np.arange(12.0).reshape(3, 4)
    key = fingerprint("realized inputs")
    cache.put(key, value)
    np.testing.assert_array_equal(cache.get(key), value)
    payload = next(tmp_path.glob("*.npy"))
    with payload.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    assert cache.get(key) is None
    with pytest.raises(ValueError):
        cache.put(key, np.array([object()], dtype=object))


def test_interrupted_publish_is_a_miss_and_does_not_touch_other_files(tmp_path, monkeypatch):
    import myogen.simulator.core.emg.runtime.cache as module
    cache = DiskCache(tmp_path)
    marker = tmp_path / "user-data.txt"
    marker.write_text("preserve")
    def fail(*_):
        raise OSError("injected interruption")
    monkeypatch.setattr(module.os, "replace", fail)
    key = fingerprint("incomplete")
    with pytest.raises(OSError):
        cache.put(key, np.arange(10.0))
    assert cache.get(key) is None
    assert marker.read_text() == "preserve"


def test_eviction_only_touches_owned_valid_entries(tmp_path):
    cache = DiskCache(tmp_path, limit_bytes=1000)
    foreign = tmp_path / "unrecognized.npy"
    foreign.write_bytes(b"keep me")
    for index in range(4):
        cache.put(fingerprint(index), np.arange(64.0))
    assert foreign.read_bytes() == b"keep me"
    assert sum(path.stat().st_size for path in cache.directory.glob("*.npy")
               if path.name != foreign.name) <= 1000


def test_metadata_path_cannot_escape_cache_directory(tmp_path):
    cache = DiskCache(tmp_path)
    key = fingerprint("unsafe")
    cache.put(key, np.arange(8.0))
    manifest = tmp_path / (key + ".json")
    metadata = json.loads(manifest.read_text())
    metadata["payload"] = "../foreign.npy"
    manifest.write_text(json.dumps(metadata))
    assert cache.get(key) is None


@pytest.mark.parametrize("invalid", [[], None, 7, "invalid"])
def test_json_with_invalid_manifest_schema_is_a_cache_miss(tmp_path, invalid):
    cache = DiskCache(tmp_path)
    key = fingerprint("bad schema")
    (tmp_path / (key + ".json")).write_text(json.dumps(invalid))
    assert cache.get(key) is None


def test_long_cache_path_round_trip(tmp_path):
    # Deployment contract: a worktree and descriptive run folder exceed MAX_PATH.
    folder = tmp_path / ("nested-" + "x" * 150)
    cache = DiskCache(folder)
    value = np.arange(12.0)
    key = fingerprint("long path")
    cache.put(key, value)
    np.testing.assert_array_equal(cache.get(key), value)


def test_failed_manifest_publish_removes_only_its_new_payload(tmp_path, monkeypatch):
    import myogen.simulator.core.emg.runtime.cache as module
    cache = DiskCache(tmp_path)
    key = fingerprint("atomic overwrite")
    original = np.arange(12.0)
    cache.put(key, original)
    before = {path.name for path in tmp_path.iterdir()}
    replace = module.os.replace
    def fail_manifest(source, destination):
        if str(destination).endswith(".json"):
            raise OSError("injected manifest interruption")
        return replace(source, destination)
    monkeypatch.setattr(module.os, "replace", fail_manifest)
    with pytest.raises(OSError):
        cache.put(key, original + 1)
    np.testing.assert_array_equal(cache.get(key), original)
    assert {path.name for path in tmp_path.iterdir()} == before

def test_empty_array_fingerprint_preserves_dtype_shape_and_identity():
    assert fingerprint(np.empty((3, 0))) == fingerprint(np.empty((3, 0)))
    assert fingerprint(np.empty((3, 0))) != fingerprint(np.empty((0, 3)))
    assert fingerprint(np.empty((3, 0))) != fingerprint(np.empty((3, 0), dtype=np.float32))
