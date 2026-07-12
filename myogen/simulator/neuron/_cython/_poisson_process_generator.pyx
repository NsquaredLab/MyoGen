# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport log, log1p
from libc.stdint cimport uint64_t

cdef class _PoissonProcessGenerator__Cython:
    """
    High-performance Cython implementation of a time-varying Poisson process
    generator for neural spike train simulation.

    This class generates spike events from an inhomogeneous Poisson process,
    commonly used to model neural firing in response to a continuous input
    intensity (rate). It uses the integrated-intensity (time-rescaling) method:
    the running integral of the input rate is accumulated over time, and a
    spike is emitted whenever that integral crosses an exponentially
    distributed threshold. After each spike the accumulator and the threshold
    are reset.

    Because the inter-spike threshold is a single ``Exp(1)`` draw, the emitted
    process is a discrete-time Poisson process (exact in the ``dt -> 0`` limit):
    at a constant input rate the inter-spike intervals are exponentially
    distributed with coefficient of variation ``CV = 1``. (At finite ``dt`` the
    ISIs are quantised to multiples of ``dt`` and at most one spike is emitted
    per step, so the fit is approximate for ``rate * dt`` not << 1.) For
    deliberately regular, low-CV firing (e.g. muscle
    afferents or cortical drive) use the Gamma generator instead — see
    ``_GammaProcessGenerator__Cython`` / ``DD_Gamma``.

    The implementation uses a custom xorshift64* random number generator for
    high performance and reproducible results across different platforms.

    Parameters
    ----------
    seed : uint64_t
        Random number generator seed for reproducible results. If 0, uses default seed.
    dt : double
        Time step in milliseconds for numerical integration of input intensity.
    Ninit : int, optional
        Number of random numbers to pre-consume from the generator before the
        first threshold is drawn, by default 0. Useful for decorrelating
        parallel generators seeded from nearby values. This only advances the
        RNG state and does not change the Poisson statistics.

    Attributes
    ----------
    dt : double
        Time step in milliseconds for numerical integration.
    yi : double
        Accumulated input intensity since the last spike event.
    thres : double
        Current ``Exp(1)`` threshold for the next spike.
    spk : int
        Binary spike output (1 for spike, 0 for no spike).
    state : uint64_t
        Internal state of the xorshift64* random number generator.

    """
    cdef double dt
    cdef double yi
    cdef double thres
    cdef int spk
    cdef uint64_t state  # C RNG state

    def __init__(self, uint64_t seed, double dt, int Ninit=0):
        self.dt = dt
        self.yi = 0.0
        self.spk = 0
        self.state = seed if seed != 0 else <uint64_t>0xDEADBEEFCAFEBABE

        # pre-consume Ninit uniforms to decorrelate parallel generators
        for _ in range(Ninit):
            self._rand_uniform()

        # first inter-spike threshold ~ Exp(1)
        self.thres = self._next_threshold()

    cdef double _rand_uniform(self):
        """
        Generate uniform random number using xorshift64* algorithm.

        High-performance pseudo-random number generator that produces uniformly
        distributed values in the range [0, 1). The algorithm uses bitwise XOR
        and shift operations for excellent performance and statistical properties.

        Returns
        -------
        double
            Uniformly distributed random number in range [0, 1).
        """
        cdef uint64_t x = self.state

        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27

        self.state = x

        return (<double>(x * 2685821657736338717 & 0xFFFFFFFFFFFFFFFF)) / 18446744073709551616.0

    cdef double _next_threshold(self):
        """
        Draw the next inter-spike integrated-intensity threshold.

        For a Poisson process the integrated intensity accumulated between two
        consecutive spikes is ``Exp(1)``-distributed, sampled here by inverse
        transform. ``_rand_uniform`` returns values in ``[0, 1)``, so ``-log1p(-U)``
        (i.e. ``-log(1 - U)``, evaluated more accurately for small ``U``) is a
        finite ``Exp(1)`` draw; ``U = 0`` yields a zero threshold (an immediate
        spike), which is harmless.

        Returns
        -------
        double
            A single exponentially distributed threshold with unit mean.
        """
        return -log1p(-self._rand_uniform())

    cpdef int compute(self, double y):
        """
        Compute spike output for given input intensity at current time step.

        Integrates the input intensity over the time step and compares the accumulated
        intensity against the current exponential threshold to determine if a spike
        should be generated. If a spike occurs, resets the accumulator and generates
        a new exponential threshold for the next inter-spike interval.

        Parameters
        ----------
        y : double
            Input intensity (rate) in Hz at the current time step. This represents
            the instantaneous firing probability density function.

        Returns
        -------
        int
            Binary spike output: 1 if spike occurs, 0 otherwise.

        Notes
        -----
        The input intensity is integrated over the time step using Euler's method:
        yi += y * dt * 1e-3, where dt is in milliseconds and y is in Hz.

        When yi exceeds the current threshold, a spike is generated and both
        yi and the threshold are reset. The new threshold is a single ``Exp(1)``
        draw, which makes the inter-spike intervals exponentially distributed
        (CV = 1) — a discrete-time Poisson process (exact as ``dt -> 0``).

        This method can be called repeatedly with time-varying input intensities
        to generate realistic Poisson spike trains that capture the temporal
        dynamics of neural firing patterns.
        """
        self.spk = 0
        self.yi += y * self.dt * 1e-3

        if self.yi >= self.thres:
            self.spk = 1
            self.yi = 0.0
            self.thres = self._next_threshold()

        return self.spk
