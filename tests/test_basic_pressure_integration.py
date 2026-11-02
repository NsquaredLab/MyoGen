"""Contract-only 16-GiB probe driven by a real, bounded external consumer.

This exercises process observation and safe-boundary recovery. It is not a
16-GiB hardware benchmark or scientific waveform evidence.
"""
import subprocess
import sys
import threading
import time

import psutil

from myogen.simulator.core.emg.runtime import ExecutionPolicy
from myogen.simulator.core.emg.runtime.memory import MemoryController, MemorySample, MIB


def test_external_consumer_stops_admission_and_recovers_at_boundary():
    code = """
import mmap, sys
print('ready', flush=True)
sys.stdin.readline()
# Explicitly unmap the consumer's pages on release. Python's allocator may
# retain a freed bytearray, so deleting it does not guarantee an RSS drop.
buffer = mmap.mmap(-1, 64 * 1024**2)
for index in range(0, len(buffer), mmap.PAGESIZE): buffer[index] = 1
print('allocated', flush=True)
sys.stdin.readline()
buffer.close()
print('released', flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        assert child.stdout.readline().strip() == "ready"
        process = psutil.Process(child.pid)
        def rss():
            # Windows venv python.exe may be a launcher with the interpreter
            # in its child; measure the complete owned process tree.
            return sum(item.memory_info().rss for item in [process, *process.children(recursive=True)])
        baseline = rss()
        def probe():
            growth = max(0, rss() - baseline)
            return MemorySample(16 * 1024**3, int(4.5 * 1024**3) + 32 * MIB - growth, growth)
        controller = MemoryController(ExecutionPolicy(), probe=probe)
        with controller.wait_reserve(MIB):
            pass
        child.stdin.write("allocate\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "allocated"
        assert controller.try_reserve(MIB) is None
        def release():
            time.sleep(.2)
            child.stdin.write("release\n")
            child.stdin.flush()
        releaser = threading.Thread(target=release)
        releaser.start()
        started = time.monotonic()
        with controller.wait_reserve(MIB):
            elapsed = time.monotonic() - started
        releaser.join()
        assert child.stdout.readline().strip() == "released"
        assert controller.events["pressure_pauses"] == 1
        assert .2 <= elapsed < 3
        assert controller.pending == 0
    finally:
        if child.poll() is None:
            child.stdin.close()
            child.wait(timeout=5)
