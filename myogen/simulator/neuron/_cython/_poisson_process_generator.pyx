# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport log
from libc.stdint cimport uint64_t

cdef class _PoissonProcessGenerator__Cython:
    """
    High-performance Cython implementation of a Poisson process generator for neural spike train simulation.
    
    This class implements an efficient algorithm for generating spike events from a time-varying
    Poisson process, commonly used to model neural firing in response to continuous input currents.
    The generator uses exponential inter-arrival times with adaptive thresholding to determine
    spike timing based on accumulated input intensity.
    
    The implementation uses a custom xorshift64* random number generator for high performance
    and reproducible results across different platforms. The algorithm accumulates input intensity
    over time and compares against exponentially distributed thresholds to determine spike events.
    
    Parameters
    ----------
    seed : uint64_t
        Random number generator seed for reproducible results. If 0, uses default seed.
    N : int
        Batch size for threshold generation. Higher values increase computational cost
        but may improve statistical properties of the generated process.
    dt : double
        Time step in milliseconds for numerical integration of input intensity.
    Ninit : int, optional
        Number of random numbers to pre-consume from generator, useful for
        decorrelating parallel generators, by default 0.
    
    Attributes
    ----------
    dt : double
        Time step in milliseconds for numerical integration.
    N : int
        Batch size for threshold generation, affecting statistical properties.
    yi : double
        Accumulated input intensity since last spike event.
    aux : double
        Auxiliary variable for exponential threshold computation.
    thres : double
        Current exponential threshold for spike generation.
    spk : int
        Binary spike output (1 for spike, 0 for no spike).
    state : uint64_t
        Internal state of the xorshift64* random number generator.
    
    """
    cdef double dt
    cdef int N
    cdef double yi
    cdef double aux
    cdef double thres
    cdef int spk
    cdef uint64_t state  # C RNG state

    def __init__(self, uint64_t seed, int N, double dt, int Ninit=0):
        self.dt = dt
        self.N = N
        self.yi = 0.0
        self.aux = 1.0
        self.thres = 0.0
        self.spk = 0
        self.state = seed if seed != 0 else <uint64_t>0xDEADBEEFCAFEBABE

        # pre-consume Ninit uniforms
        for _ in range(Ninit):
            self._rand_uniform()

        # generate first exponential threshold
        for _ in range(self.N):
            self.aux *= self._rand_uniform()

        self.thres = -(1.0 / self.N) * log(self.aux)

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
        yi and the threshold are reset. The new threshold is drawn from an
        exponential distribution using the batch method for improved efficiency.
        
        This method can be called repeatedly with time-varying input intensities
        to generate realistic Poisson spike trains that capture the temporal
        dynamics of neural firing patterns.
        """
        self.spk = 0
        self.yi += y * self.dt * 1e-3

        if self.yi >= self.thres:
            self.spk = 1
            self.yi = 0.0
            self.aux = 1.0

            for _ in range(self.N):
                self.aux *= self._rand_uniform()

            self.thres = -(1.0 / self.N) * log(self.aux)

        return self.spk
