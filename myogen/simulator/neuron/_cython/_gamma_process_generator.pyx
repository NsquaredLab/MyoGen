# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport log, pow, exp, sqrt, cos
from libc.stdint cimport uint64_t

cdef class _GammaProcessGenerator__Cython:
    """
    High-performance Cython implementation of a Gamma process generator for neural spike train simulation.

    This class implements an efficient algorithm for generating spike events from a time-varying
    Gamma process, which produces more regular spike patterns than Poisson processes.
    The shape parameter controls the regularity: shape=1 gives Poisson, shape>1 gives more regular firing.

    The implementation uses a custom xorshift64* random number generator for high performance
    and reproducible results across different platforms. The algorithm uses the shape-rate
    parameterization where the mean ISI = shape/rate and CV = 1/sqrt(shape).

    Parameters
    ----------
    seed : uint64_t
        Random number generator seed for reproducible results. If 0, uses default seed.
    shape : double
        Shape parameter (k) controlling spike regularity. Higher values give more regular firing.
        shape=1 reduces to Poisson process, shape>1 gives sub-Poisson regularity.
    dt : double
        Time step in milliseconds for numerical integration of input intensity.

    Attributes
    ----------
    dt : double
        Time step in milliseconds for numerical integration.
    shape : double
        Shape parameter controlling spike regularity (k parameter).
    yi : double
        Accumulated input intensity since last spike event.
    thres : double
        Current Gamma-distributed threshold for spike generation.
    spk : int
        Binary spike output (1 for spike, 0 for no spike).
    state : uint64_t
        Internal state of the xorshift64* random number generator.

    Notes
    -----
    Uses Marsaglia and Tsang's method for Gamma variate generation when shape >= 1.
    For more regular firing patterns typical of cortical neurons, use shape between 2-5.
    """
    cdef double dt
    cdef double shape
    cdef double yi
    cdef double thres
    cdef int spk
    cdef uint64_t state  # C RNG state

    def __init__(self, uint64_t seed, double shape, double dt):
        self.dt = dt
        self.shape = shape
        self.yi = 0.0
        self.thres = 0.0
        self.spk = 0
        self.state = seed if seed != 0 else <uint64_t>0xDEADBEEFCAFEBABE

        # Generate first Gamma threshold
        self.thres = self._rand_gamma()

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

    cdef double _rand_normal(self):
        """
        Generate standard normal random variable using Box-Muller transform.

        Returns
        -------
        double
            Normally distributed random number with mean=0, std=1.
        """
        cdef double u1 = self._rand_uniform()
        cdef double u2 = self._rand_uniform()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2)

    cdef double _rand_gamma(self):
        """
        Generate Gamma-distributed random variable using Marsaglia-Tsang method.

        Generates random variates from Gamma(shape, 1) distribution using an efficient
        acceptance-rejection algorithm. For shape >= 1, uses the Marsaglia and Tsang
        method which has high acceptance rate.

        Returns
        -------
        double
            Gamma-distributed random number with shape parameter self.shape and scale=1.
        """
        cdef double d, c, x, v, u

        if self.shape >= 1.0:
            # Marsaglia and Tsang method
            d = self.shape - 1.0/3.0
            c = 1.0 / sqrt(9.0 * d)

            while True:
                x = self._rand_normal()
                v = pow(1.0 + c * x, 3.0)

                if v > 0:
                    u = self._rand_uniform()
                    if u < 1.0 - 0.0331 * pow(x, 4.0):
                        return d * v
                    if log(u) < 0.5 * pow(x, 2.0) + d * (1.0 - v + log(v)):
                        return d * v
        else:
            # For shape < 1, use rejection method with Gamma(shape+1) and scaling
            return self._rand_gamma_small_shape()

    cdef double _rand_gamma_small_shape(self):
        """Handle Gamma generation for shape < 1."""
        cdef double u = self._rand_uniform()
        cdef double gamma_plus_1 = self._rand_gamma_marsaglia(self.shape + 1.0)
        return gamma_plus_1 * pow(u, 1.0 / self.shape)

    cdef double _rand_gamma_marsaglia(self, double shape):
        """Marsaglia-Tsang for arbitrary shape >= 1."""
        cdef double d, c, x, v, u

        d = shape - 1.0/3.0
        c = 1.0 / sqrt(9.0 * d)

        while True:
            x = self._rand_normal()
            v = pow(1.0 + c * x, 3.0)

            if v > 0:
                u = self._rand_uniform()
                if u < 1.0 - 0.0331 * pow(x, 4.0):
                    return d * v
                if log(u) < 0.5 * pow(x, 2.0) + d * (1.0 - v + log(v)):
                    return d * v

    cpdef int compute(self, double y):
        """
        Compute spike output for given input intensity at current time step.

        Integrates the input intensity over the time step and compares the accumulated
        intensity against the current Gamma-distributed threshold to determine if a spike
        should be generated. If a spike occurs, resets the accumulator and generates
        a new Gamma threshold for the next inter-spike interval.

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
        yi and the threshold are reset. The new threshold is drawn from a
        Gamma distribution, producing more regular spike trains than Poisson.

        This method can be called repeatedly with time-varying input intensities
        to generate realistic Gamma-process spike trains with controlled regularity.
        """
        self.spk = 0
        self.yi += y * self.dt * 1e-3

        if self.yi >= self.thres:
            self.spk = 1
            self.yi = 0.0
            self.thres = self._rand_gamma()

        return self.spk
