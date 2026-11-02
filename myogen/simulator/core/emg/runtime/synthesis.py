"""Bounded NumPy-compatible convolution, with overlap-add FFT blocks."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import choose_conv_method
from .cache import fingerprint


def array_key(array, *dependencies):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((value.shape, value.dtype.str, dependencies)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def convolve_add(output, spikes, kernel, session, *, method="auto", block_size=8192):
    """Add a complete contribution; failed FFT blocks never reach output.

    NumPy's 'same' length is max(T,K), including K>T. Crop the full
    convolution with (min(T,K)-1)//2, independently of SciPy's convention.
    """
    spikes = np.asarray(spikes)
    kernel = np.asarray(kernel)
    if spikes.ndim != 1 or kernel.ndim != 1 or not spikes.size or not kernel.size:
        raise ValueError("Convolution inputs must be nonempty vectors")
    if output.shape != (max(spikes.size, kernel.size),):
        raise ValueError("Output does not have NumPy same-convolution shape")
    if not np.any(spikes):
        return
    automatic = method == "auto"
    if method == "auto":
        # SciPy's estimate for Boolean occupancy and float64 templates depends
        # on shape/dtype, not values. Integer inputs retain the value-dependent
        # precision check in choose_conv_method on every call.
        if spikes.dtype == np.bool_ and kernel.dtype == np.float64:
            key = (spikes.size, kernel.size)
            method = session.convolution_methods.get(key)
            if method is None:
                method = choose_conv_method(spikes, kernel, mode="full")
                session.convolution_methods[key] = method
        else:
            method = choose_conv_method(spikes, kernel, mode="full")
    if method not in ("direct", "fft"):
        raise ValueError("method must be auto, direct, or fft")
    if method == "direct":
        # One MU temporary, released immediately, instead of a recording per MU.
        # Both the promoted boolean input and convolution result coexist.
        amount = (spikes.size + kernel.size + max(spikes.size, kernel.size)) * 8
        # Admission itself supplies the fresh physical-memory measurement;
        # avoid a separate check immediately followed by the same reservation.
        lease = (session.memory.try_reserve(amount) if automatic
                 else session.memory.wait_reserve(amount))
        if lease is not None:
            with lease:
                try:
                    result = np.convolve(spikes, kernel, mode="same")
                except MemoryError as error:
                    session.memory.events["allocation_retries"] += 1
                    if not automatic:
                        raise session.memory.failure(amount, "direct convolution allocation failed") from error
                else:
                    session.check_cancelled()
                    output += result
                    return
    crop = (min(spikes.size, kernel.size) - 1) // 2
    batch = min(block_size, spikes.size)
    offset = 0
    transform_size = transform_key = None
    # FFT arrays are bounded by K+block, not the recording length.
    while offset < spikes.size:
        session.check_cancelled()
        nfft = next_fast_len(batch + kernel.size - 1, real=True)
        estimate = nfft * 8 * 8
        while estimate > session.memory.allowance() and batch > 1:
            batch = max(1, batch // 2)
            nfft = next_fast_len(batch + kernel.size - 1, real=True)
            estimate = nfft * 8 * 8
        lease = session.memory.wait_reserve(estimate)
        stop = min(spikes.size, offset + batch)
        transformed = result = None
        try:
            # Transform cache is added by the cache layer; no recording-sized
            # Fourier arrays are required even when the recording is long.
            # A contribution has one stable kernel. Reuse its content key
            # across equal-sized blocks; rehash after a size-changing retry.
            # The next call hashes again, so caller mutations invalidate it.
            if nfft != transform_size:
                transform_key = fingerprint("rfft", kernel, nfft, "float64")
                transform_size = nfft
            transformed = session.cache.get(transform_key)
            if transformed is None:
                transformed = rfft(kernel, nfft, workers=1)
                child = lease.allocate_child(transformed.nbytes)
                session.leases.append(child)
                session._register(transformed, child)
                session.cache.put(transform_key, transformed)
            else:
                session.memory.events["cache_hits"] += 1
            result = irfft(rfft(spikes[offset:stop], nfft, workers=1) * transformed, nfft, workers=1)
            result = result[:stop - offset + kernel.size - 1]
        except MemoryError:
            # Do not keep a local reference to a failed block's transform:
            # cache eviction must be able to free it before retry admission.
            transformed = result = None
            session.memory.events["allocation_retries"] += 1
            if batch == 1:
                raise session.memory.failure(estimate, "FFT allocation failed at smallest block")
            batch = max(1, batch // 2)
            continue
        finally:
            lease.close()
        # Intersect this block's full-convolution interval with the global crop.
        left = max(offset, crop)
        right = min(offset + result.size, crop + output.size)
        session.check_cancelled()
        if left < right:
            output[left - crop:right - crop] += result[left - offset:right - offset]
        del result, transformed
        offset = stop
