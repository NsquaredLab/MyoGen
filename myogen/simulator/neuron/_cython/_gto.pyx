# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libc.math cimport log

cdef class _GolgiTendonOrgan__Cython():
    """
    Golgi Tendon Organ (GTO) mechanoreceptor model for force sensing.
    
    This class implements a biophysical model of Golgi Tendon Organs that 
    generate group Ib afferent firing patterns in response to muscle force 
    changes. GTOs are force sensors located at the junction between muscle 
    fibers and tendons, providing feedback about muscle tension.
    
    The model is based on multiple sources:
    - Lae PhD thesis, pg 83
    - Aniss et al., 1990b for GTO human data  
    - Lin & Crago, 2002 model

    Parameters
    ----------
    gtoD : dict
        Dictionary containing GTO model parameters (G1, G2)
    tstop__ms : double
        Simulation duration in milliseconds
    dt__ms : double
        Integration time step in milliseconds
        
    References
    ----------
    Aniss, A. M., et al. (1990). Reflex transmission in the pathway of 
    heteronymous excitation from soleus Ib afferents to quadriceps 
    motoneurons. Experimental Brain Research, 80(3), 691-700.
    
    Lin, C. C. K., & Crago, P. E. (2002). Neural and mechanical 
    contributions to the stretch reflex: a model synthesis. 
    Annals of biomedical engineering, 30(1), 54-67.
    """
    cdef double dt
    cdef int tInt
    cdef double b0
    cdef double b1
    cdef double b2
    cdef double a1
    cdef double a2
    cdef double G1
    cdef double G2
    cdef double[::1] _gto  # gtoR() output [Hz]
    cdef double[::1] gtoG  # gtoF() output [Hz]
    cdef readonly double[::1] Ib  # [Hz]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def __init__(self, dict gtoD, double tstop__ms, double dt__ms):
        assert gtoD is not None
        assert tstop__ms > 0
        assert tstop__ms > dt__ms
        assert dt__ms > 0
        cdef Py_ssize_t tlen = len(np.arange(0, tstop__ms + dt__ms, dt__ms))
        self.dt = dt__ms * 1e-3  # [s]
        cdef double den = 0.4 * self.dt ** 2 + 4.4 * self.dt + 4
        self.b0 = (0.4 * self.dt ** 2 + 5.16 * self.dt + 6.8) / den
        self.b1 = (0.8 * self.dt ** 2 - 13.6) / den
        self.b2 = (0.4 * self.dt ** 2 - 5.16 * self.dt + 6.8) / den
        self.a1 = (0.8 * self.dt ** 2 - 8) / den
        self.a2 = (0.4 * self.dt ** 2 - 4.4 * self.dt + 4) / den
        self.G1 = gtoD['G1']
        self.G2 = gtoD['G2']
        self._gto = np.zeros(tlen, dtype=np.float64)
        self.gtoG = np.zeros(tlen, dtype=np.float64)
        self.Ib = np.zeros(tlen, dtype=np.float64)
        self.tInt = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void gtoR(self, double f) noexcept nogil:
        """
        Force-to-firing rate transduction using logarithmic relationship.
        
        Parameters
        ----------
        f : double
            Applied force in Newtons
        """
        self._gto[self.tInt] = self.G1 * log(f / self.G2 + 1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void gtoF(
        self,
        double R,
        double Rd1,
        double Rd2,
        double Fd1,
        double Fd2
    ) noexcept nogil:
        """
        Second-order digital filter for dynamic response.
        
        Implements the filter equation:
        F(n) = b0*R(n) + b1*R(n-1) + b2*R(n-2) - a1*F(n-1) - a2*F(n-2)
        
        Parameters
        ----------
        R : double
            Current firing rate input
        Rd1 : double
            Previous firing rate (R delayed by 1 timestep)
        Rd2 : double
            Previous firing rate (R delayed by 2 timesteps)
        Fd1 : double
            Previous filter output (F delayed by 1 timestep)
        Fd2 : double
            Previous filter output (F delayed by 2 timesteps)
        """
        cdef double feedforward_term, feedback_term
        feedforward_term = self.b0 * R + self.b1 * Rd1 + self.b2 * Rd2
        feedback_term = self.a1 * Fd1 + self.a2 * Fd2
        self.gtoG[self.tInt] = feedforward_term - feedback_term

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def integrate(self, double f):
        """
        Integrate one timestep of GTO dynamics.
        
        Processes force input through logarithmic transduction and digital filtering
        to generate Group Ib afferent firing rate output.
        
        Parameters
        ----------
        f : double
            Applied force in Newtons
            
        Returns
        -------
        double
            Group Ib afferent firing rate in Hz
        """
        self.gtoR(f)
        if self._gto[self.tInt] < 0:
            self._gto[self.tInt] = 0
        if self.tInt >= 2:
            self.gtoF(
                self._gto[self.tInt],
                self._gto[self.tInt - 1],
                self._gto[self.tInt - 2],
                self.gtoG[self.tInt - 1],
                self.gtoG[self.tInt - 2]
            )
        elif self.tInt == 0:
            self.gtoF(self._gto[self.tInt], 0, 0, 0, 0)
        elif self.tInt == 1:
            self.gtoF(
                self._gto[self.tInt],
                self._gto[self.tInt - 1],
                0,
                self.gtoG[self.tInt - 1],
                0
            )
        self.Ib[self.tInt] = self.gtoG[self.tInt]
        if self.Ib[self.tInt] < 0:
            self.Ib[self.tInt] = 0
        self.tInt = self.tInt + 1
        return self.Ib[self.tInt - 1]
