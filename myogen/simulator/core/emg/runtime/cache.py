"""Content-addressed numerical caches; no pickle, object identity, or seed keys."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import threading

import numpy as np

VERSION = "public-emg-cpu-v1"
OWNER = "myogen-basic-mu-cache"


def fingerprint(*values):
    digest = hashlib.sha256(VERSION.encode())
    def add(value):
        if hasattr(value, "units") and hasattr(value, "magnitude"):
            add(("quantity", str(value.units), value.magnitude))
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object arrays have no numerical cache identity")
            array = np.ascontiguousarray(value)
            add(("array", array.dtype.str, array.shape))
            if array.size:
                digest.update(memoryview(array).cast("B"))
        elif isinstance(value, np.generic):
            add(value.item())
        elif isinstance(value, dict):
            add(tuple(sorted(value.items())))
        elif isinstance(value, (tuple, list)):
            digest.update(b"[")
            for item in value:
                add(item)
            digest.update(b"]")
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(json.dumps(value, allow_nan=False).encode())
            digest.update(b";")
        else:
            raise TypeError(f"No complete dependency representation for {type(value).__name__}")
    add(values)
    return digest.hexdigest()


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiskCache:
    def __init__(self, directory, *, limit_bytes=4 * 1024**3):
        self.directory = Path(directory)
        if os.name == "nt":
            # Content-addressed names plus a nested worktree routinely exceed
            # MAX_PATH. Extended absolute paths work without a machine setting.
            absolute = str(self.directory.resolve())
            if not absolute.startswith("\\\\?\\"):
                absolute = ("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                            else "\\\\?\\" + absolute)
            self.directory = Path(absolute)
        self.limit = limit_bytes
        self.directory.mkdir(parents=True, exist_ok=True)

    def _metadata(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid cache key")
        path = self.directory / (key + ".json")
        if path.is_symlink() or not path.exists():
            return None
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                return None
            if (metadata.get("owner") != OWNER or metadata.get("version") != VERSION
                    or metadata.get("key") != key):
                return None
            name = metadata["payload"]
            if not (re.fullmatch(r"[0-9a-f]{32}\.npy", name) or
                    re.fullmatch(key + r"-(?:[0-9a-f]{64}|[0-9a-f]{32})\.npy", name)):
                return None
            payload = self.directory / name
            if payload.is_symlink() or not payload.is_file():
                return None
            return metadata
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def get(self, key, session=None):
        metadata = self._metadata(key)
        if metadata is None:
            return None
        try:
            path = self.directory / metadata["payload"]
            if checksum(path) != metadata["sha256"]:
                return None
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            if (array.dtype.hasobject or list(array.shape) != metadata["shape"] or
                    array.dtype.str != metadata["dtype"] or array.nbytes != metadata["nbytes"]):
                return None
            if session is not None:
                result = session.zeros(array.shape, array.dtype)
                np.copyto(result, array)
                session.memory.events["cache_hits"] += 1
            else:
                result = np.array(array)
            return result
        except (OSError, ValueError, KeyError, EOFError):
            return None

    def put(self, key, array):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid cache key")
        array = np.asarray(array)
        if array.dtype.hasobject or not np.all(np.isfinite(array)):
            raise ValueError("Only finite numerical completed-MU results may be cached")
        if array.nbytes > self.limit:
            return
        token = uuid.uuid4().hex
        temporary = self.directory / (f".{token}.npy.tmp")
        manifest_tmp = self.directory / (f".{token}.json.tmp")
        previous = self._metadata(key)
        payload = None
        published = False
        try:
            with temporary.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            digest = checksum(temporary)
            # Unique publication ownership also permits safe failure cleanup
            # when another process is writing this same numerical key.
            name = f"{token}.npy"
            payload = self.directory / name
            os.replace(temporary, payload)
            metadata = dict(owner=OWNER, version=VERSION, key=key, payload=name,
                            sha256=digest, shape=list(array.shape), dtype=array.dtype.str,
                            nbytes=array.nbytes)
            with manifest_tmp.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            # The manifest is the commit point. Readers never observe partial MUs.
            os.replace(manifest_tmp, self.directory / (key + ".json"))
            published = True
            if previous is not None:
                try:
                    (self.directory / previous["payload"]).unlink(missing_ok=True)
                except OSError:
                    pass  # An existing Windows reader can retain its mapping.
            self.evict()
        finally:
            temporary.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
            if not published and payload is not None:
                payload.unlink(missing_ok=True)

    def evict(self):
        entries = []
        for path in self.directory.glob("*.json"):
            if not re.fullmatch(r"[0-9a-f]{64}", path.stem):
                continue
            metadata = self._metadata(path.stem)
            if metadata is not None:
                payload = self.directory / metadata["payload"]
                try:
                    entries.append((path.stat().st_mtime_ns, path, payload, payload.stat().st_size))
                except OSError:
                    pass
        total = sum(entry[3] for entry in entries)
        for _, manifest, payload, size in sorted(entries):
            if total <= self.limit:
                break
            try:
                # Owned, validated local names only. Open readers can finish;
                # a race with eviction is a cache miss, never corrupt output.
                payload.unlink()
                manifest.unlink(missing_ok=True)
                total -= size
            except OSError:
                continue


class ArrayCache:
    """Small useful LRU cache; entries are read-only and entirely evictable."""
    def __init__(self, limit_bytes=64 * 1024**2):
        self.limit = limit_bytes
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            value = self.entries.get(key)
            if value is not None:
                self.entries.move_to_end(key)
            return value

    def put(self, key, value):
        with self.lock:
            if value.nbytes > self.limit:
                return
            value.flags.writeable = False
            self.entries[key] = value
            self.entries.move_to_end(key)
            while sum(array.nbytes for array in self.entries.values()) > self.limit:
                self.entries.popitem(last=False)

    def clear(self):
        with self.lock:
            count = len(self.entries)
            self.entries.clear()
            return count
