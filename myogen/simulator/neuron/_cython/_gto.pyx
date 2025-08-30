# cython: language_level=3
import numpy as np
cimport cython
from cython.parallel import prange

cdef class _GolgiTendonOrgan__Cython():
	# Lae phD thesis, pg 83
	# Aniss et al., 1990b for GTO human data
	# Lin & Crago, 2002 model
	# O número de aferentes Ib (300 para o SOL, 120 para o GM e 120 para o GL) 
	# foi baseado no fato de que em gatos o número de aferentes Ib é 
	# aproximadamente a metade do número de aferentes Ia (Cisi, 2007).
	# Faixa de velocidade de condução axonal das aferentes Ib 52-62 [m/s]
	# Limiares de recrutamento das aferentes Ib 0-50 Hz
	cdef double dt
	cdef int 	tInt
	cdef double b0
	cdef double b1
	cdef double b2
	cdef double a1
	cdef double a2
	cdef double G1
	cdef double G2
	cdef double[::1] _gto 			# gtoR() output [Hz]
	cdef double[::1] gtoG 			# gtoF() output [Hz]
	cdef readonly double[::1] Ib 	# [Hz]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	def __init__(self,dict gtoD,double tstop, double dt):
		assert gtoD != None
		assert tstop>0
		assert tstop>dt
		assert dt>0
		print('Creating GTO instance.')
		cdef Py_ssize_t tlen = len(np.arange(0,tstop+dt,dt))
		self.dt 		= dt*1e-3 # [s]
		cdef double den = 0.4*self.dt**2 + 4.4*self.dt + 4
		self.b0 		= (0.4*self.dt**2 + 5.16*self.dt + 6.8)/den
		self.b1 		= (0.8*self.dt**2 - 13.6)/den
		self.b2 		= (0.4*self.dt**2 - 5.16*self.dt +6.8)/den
		self.a1 		= (0.8*self.dt**2 - 8)/den
		self.a2 		= (0.4*self.dt**2 - 4.4*self.dt +4)/den
		self.G1 		= gtoD['G1']
		self.G2 		= gtoD['G2']
		self._gto 		= np.zeros(tlen,dtype= np.double)
		self.gtoG 		= np.zeros(tlen,dtype= np.double)
		self.Ib 		= np.zeros(tlen,dtype= np.double)
		self.tInt 		= 0
		print('GTO instance created.')

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void gtoR(self,double f): # F [N]
		self._gto[self.tInt] = self.G1*np.log(f/self.G2+1)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void gtoF(self,double R, double Rd1,double Rd2,double Fd1,
														double Fd2):
		cdef double t1, t2
		t1 = self.b0*R + self.b1*Rd1 + self.b2*Rd2
		t2 = self.a1*Fd1 + self.a2*Fd2
		self.gtoG[self.tInt] = t1 - t2

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	def integrate(self,double f):
		self.gtoR(f)
		if self._gto[self.tInt] < 0:
			self._gto[self.tInt] = 0
		if self.tInt >= 2:
			self.gtoF(
				self._gto[self.tInt],
				self._gto[self.tInt-1],
				self._gto[self.tInt-2],
				self.gtoG[self.tInt-1],
				self.gtoG[self.tInt-2])
		elif self.tInt == 0:
			self.gtoF(self._gto[self.tInt],0,0,0,0)
		elif self.tInt == 1:
			self.gtoF(
				self._gto[self.tInt],
				self._gto[self.tInt-1],
				0,
				self.gtoG[self.tInt-1],
				0)
		self.Ib[self.tInt] = self.gtoG[self.tInt]
		if self.Ib[self.tInt] < 0:
			self.Ib[self.tInt] = 0
		self.tInt = self.tInt + 1
		return self.Ib[self.tInt-1]
