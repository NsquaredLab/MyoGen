"""Corrected GitLab baseline benchmark utilities."""
from collections.abc import Mapping
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gl"
BASE = "fc077cb4fa89b8cb23c7f0ec4fb50dc8fef05831"

def dump_json(path, value):
    def convert(item):
        if isinstance(item, Mapping):
            return dict(item)
        if isinstance(item, Path):
            return str(item)
        return item.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def setup():
    import sys
    import psutil
    process = psutil.Process()
    allowed = process.cpu_affinity()
    cores = sorted(topology(), key=lambda c: (-c["efficiency_class"], c["logical"][0]))
    chosen = [next(i for i in c["logical"] if i in allowed)
              for c in cores if any(i in allowed for i in c["logical"])][:4]
    process.cpu_affinity(chosen)
    sys.modules["cupy"] = None
    import myogen
    assert Path(myogen.__file__).resolve().is_relative_to(ROOT)
    return dict(python=sys.executable, imported=myogen.__file__, affinity=chosen,
                cpu_only=True, actual_total_memory=psutil.virtual_memory().total,
                available_memory=psutil.virtual_memory().available)

def restore(inputs):
    import myogen
    myogen.set_random_seed(180319)
    myogen.get_random_generator().bit_generator.state = json.loads(
        (inputs/"realization-rng.json").read_text())

def owner(muscle, kind, selected=None):
    from myogen import simulator
    import quantities as pq
    if kind == "i":
        return simulator.IntramuscularEMG(muscle, simulator.IntramuscularElectrodeArray(
            num_electrodes=4, inter_electrode_distance__mm=2*pq.mm,
            position__mm=(0*pq.mm, 0*pq.mm, 15*pq.mm),
            differentiation_mode="consecutive"), MUs_to_simulate=selected)
    return simulator.SurfaceEMG(muscle, [simulator.SurfaceElectrodeArray(
        num_rows=5, num_cols=5, inter_electrode_distance__mm=5*pq.mm,
        electrode_radius__mm=5*pq.mm, differentiation_mode="monopolar",
        bending_radius__mm=muscle.radius__mm+muscle.skin_thickness__mm+muscle.fat_thickness__mm)],
        MUs_to_simulate=selected)

def topology():
    """Read physical-core masks; use one logical thread per selected core.

    PROCESSOR_RELATIONSHIP ABI: flags/class at +8/+9, GroupCount at +30,
    first GROUP_AFFINITY at +32. Windows x64, single processor group only.
    https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-processor_relationship
    """
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("This estimate harness requires Windows x64")
    fn = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    fn.restype = ctypes.c_int
    length = ctypes.c_uint32()
    fn(0, None, ctypes.byref(length))  # RelationProcessorCore
    if not length.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(length.value)
    if not fn(0, buffer, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    raw, offset, cores = buffer.raw[:length.value], 0, []
    while offset < len(raw):
        relation, size = struct.unpack_from("<II", raw, offset)
        if relation != 0 or size < 48 or offset + size > len(raw):
            raise RuntimeError("Unexpected Windows processor-core record")
        groups = struct.unpack_from("<H", raw, offset + 30)[0]
        mask, group = struct.unpack_from("<QH", raw, offset + 32)
        if groups != 1 or group != 0:
            raise RuntimeError("Multiple processor groups are outside this harness")
        cores.append(dict(efficiency_class=raw[offset + 9],
                          logical=[i for i in range(64) if mask & (1 << i)]))
        offset += size
    return cores
