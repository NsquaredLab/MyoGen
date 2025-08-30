# cython: language_level=3str
import numpy as np
cimport cython
from cython.parallel import prange

cdef class _Spindle__Cython():
	# Source: Mileusnic et al., 2006, pg 1774 Table 1

	#Activation parameters
	cdef int fBag1 		# [pps] Fusimotor freq. to actvation cte
	cdef int fBag2  	# [pps] Fusimotor freq. to actvation cte
	cdef int fChain  	# [pps] Fusimotor freq. to actvation cte
	cdef int P       	# Fusimotor freq. to actvation Power cte
	cdef double G1      # [FU] dynamic fusimotor input Force gen. coef
	cdef double G2      # [FU] static fusimotor input Force gen. coef
	cdef double G2Chain # [FU] static fusimotor input Force gen. coef
	#Add low pass filter for the two slow bag intrafusal fibers

	#Sensory region (SR) parameters
	cdef double K_SR    # SR spring constant [FU/L0]
	cdef double L0_SR   # SR rest length [L0]
	cdef double LN_SR   # SR threshold length [L0]

	#Polar Region (PR) parameters
	cdef double K_PR    # PR spring constant [FU/L0]
	cdef double L0_PR   # PR rest length [L0]
	cdef double LN_PR   # PR threshold length [L0]

	cdef double M       # Intrafusal fiber mass [FU/(L0/s^2)]
	cdef double b0Bag1  # Passive damping coefficient [FU/(L0/s)]
	cdef double b0Bag2  # Passive damping coefficient [FU/(L0/s)]
	cdef double b0Chain # Passive damping coefficient [FU/(L0/s)]
	cdef double b1Bag1  # Dynamic fusimotor damping coef [FU/(L0/s)]
	cdef double b2Bag2  # Static fusimotor damping coef [FU/(L0/s)]
	cdef double b2Chain # Static fusimotor damping coef [FU/(L0/s)]

	cdef double a       # Nonlinear velocity dependence power constant
	cdef double C_L     # Lengthening coef. of asymmetry in F-V curve
	cdef double C_S     # Shortening coef. of asymmetry in F-V curve
	cdef double R       # Fascile length where force prod is zero [L0]

	#Afferent properties
	cdef double X       # Secondary afferent [%] on sensory region
	                    # This value can vary among spindles
	cdef double gBag1   # SR's stretch to afferent firing [pps/L0]
	cdef double gBag2A1 # Bag2 Aff Ia stretch to aff. firing [pps/L0]
	cdef double gChainA1# Chain Aff Ia stretch to aff. firing [pps/L0]
	cdef double gBag2A2 # Bag2 Aff II stretch to aff. firing [pps/L0]
	cdef double gChainA2# Chain Aff II stretch to aff. firing [pps/L0]
	cdef double Lsec    # Secondary afferent rest length [L0]
	cdef double S       # occlusion that occurs in aff Ia [pg1776]
	cdef double tau1    # Low-pass filter time constant
	cdef double tau2    # Low-pass filter time constant
	cdef int tInt 		# Integration step
	cdef double tstop      # sim. time [ms]
	cdef double dt      # time step [ms]
	cdef readonly double[::1] time # time array [ms]
	cdef readonly double[:,::1] T 	# Tension
	cdef double[:,::1] dT
	cdef double[3] k1y
	cdef double[3] k2y
	cdef double[3] k3y
	cdef double[3] k4y
	cdef double[3] k1z
	cdef double[3] k2z
	cdef double[3] k3z
	cdef double[3] k4z
	cdef readonly double[::1] aBag1 # Activation Bag1
	cdef readonly double[::1] aBag2 # Activation Bag2
	cdef readonly double[::1] aChain# Activation Chain
	cdef double[:,::1] IaMid
	cdef readonly double[::1] Ia    # Primary afferent
	cdef readonly double[::1] II  	 # Secondary Afferent

	def __init__(self,double tstop, double dt, dict spinD):
		assert tstop>0
		assert tstop>dt
		assert dt>0
		print('Creating Spindle instance')
		cdef Py_ssize_t tlen
		self.time    = np.arange(0,tstop+dt,dt,dtype= np.double)
		tlen 	 = self.time.shape[0]
		self.fBag1   = spinD["fBag1"]
		self.fBag2   = spinD["fBag2"]
		self.fChain  = spinD["fChain"]
		self.P       = spinD["P"] 
		self.G1      = spinD["G1"]
		self.G2      = spinD["G2"]
		self.G2Chain = spinD["G2Chain"]
		self.K_SR    = spinD["K_SR"]
		self.L0_SR   = spinD["L0_SR"]
		self.LN_SR   = spinD["LN_SR"]
		self.K_PR    = spinD["K_PR"]
		self.L0_PR   = spinD["L0_PR"]
		self.LN_PR   = spinD["LN_PR"]
		self.M       = spinD["M"]
		self.b0Bag1  = spinD["b0Bag1"]
		self.b0Bag2  = spinD["b0Bag2"]
		self.b0Chain = spinD["b0Chain"]
		self.b1Bag1  = spinD["b1Bag1"]
		self.b2Bag2  = spinD["b2Bag2"]
		self.b2Chain = spinD["b2Chain"]
		self.a       = spinD["a"]
		self.C_L     = spinD["C_L"] 
		self.C_S     = spinD["C_S"]
		self.R       = spinD["R"]
		self.X       = spinD["X"]
		self.gBag1   = spinD["gBag1"]
		self.gBag2A1 = spinD["gBag2A1"]
		self.gChainA1= spinD["gChainA1"]
		self.gBag2A2 = spinD["gBag2A2"]
		self.gChainA2= spinD["gChainA2"]
		self.Lsec    = spinD["Lsec"]
		self.S       = spinD["S"]
		self.tau1    = spinD["tau1"]
		self.tau2    = spinD["tau2"]
		self.tInt 	 = 0      
		self.dt      = dt*1e-3 # [s]
		self.T       = np.zeros((3,tlen),dtype= np.double)
		self.dT      = np.zeros((3,tlen),dtype= np.double)
		self.k1y     = [0,0,0]
		self.k2y     = [0,0,0]
		self.k3y     = [0,0,0]
		self.k4y     = [0,0,0]
		self.k1z     = [0,0,0]
		self.k2z     = [0,0,0]
		self.k3z     = [0,0,0]
		self.k4z     = [0,0,0]
		self.aBag1   = np.zeros((tlen),dtype= np.double)
		self.aBag2   = np.zeros((tlen),dtype= np.double)
		self.aChain  = np.zeros((tlen),dtype= np.double)
		self.IaMid   = np.zeros((3,tlen),dtype= np.double)
		self.Ia      = np.zeros((tlen),dtype= np.double)
		self.II      = np.zeros((tlen),dtype= np.double)
		print('Spindle configured')

	cdef double dfdtBag1(	self,double t, double aBag1,
							double gDyn) except? -1:
		cdef double t1
		t1 = gDyn**self.P/(gDyn**self.P+self.fBag1**self.P)
		return (t1 - aBag1)/self.tau1

	cdef double dfdtBag2(	self,double t,double aBag2,
							double gStat) except? -1:
		cdef double t1
		t1 = gStat**self.P/(gStat**self.P+self.fBag2**self.P)
		return (t1 - aBag2)/self.tau2

	cdef double dfChain (self, double gStat) except? -1:
		return gStat**self.P/(gStat**self.P + self.fChain**self.P)

	cdef double lengthPR(self,double T,double L) except? -1:
		return -T/self.K_SR + L - self.L0_SR

	cdef int sign(self,double x) nogil:
		return 1 if x >= 0 else -1

	cdef double FVCoef(self,double V,double dT) nogil:
		if (V - dT/self.K_SR ) >= 0:
			return self.C_L
		else:
			return self.C_S

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef (double,double) PRForceDampingCoef(self, Py_ssize_t ifType) nogil:
		cdef double tB, tGm
		if ifType == 0:  
			tB = self.b0Bag1+self.b1Bag1*self.aBag1[self.tInt]
			tGm= self.G1*self.aBag1[self.tInt]
		if ifType == 1:
			tB = self.b0Bag2+self.b2Bag2*self.aBag2[self.tInt]
			tGm= self.G2*self.aBag2[self.tInt]
		if ifType == 2:
			tB = self.b0Chain+self.b2Chain*self.aChain[self.tInt]
			tGm= self.G2Chain*self.aChain[self.tInt]
		return tB,tGm

	cdef double dTen(self,double t,double y,double z) nogil:
		return z

	cdef double fabs(self,double value) nogil:
		if value < 0:
			return -value
		else:
			return value

	cdef double ddTen(self,double t,double T,double dT,double L,
						double V ,double A,Py_ssize_t ifType) nogil:
		cdef double t1, t2, t3, t4, t5, tB, tGm
		tB,tGm = self.PRForceDampingCoef(ifType)
		t1 = self.FVCoef(V, dT)*tB*self.sign(V-dT/self.K_SR)
		t2 = self.fabs(V - dT/self.K_SR)**self.a
		t3 = L - self.L0_SR - T/self.K_SR - self.R
		t4 = self.K_PR*(L-self.L0_SR-T/self.K_SR-self.L0_PR)
		t5 = self.M * A + tGm - T + t4
		return self.K_SR/self.M*((t1*t2*t3)+t5)


	cdef double affIa(self,double T,Py_ssize_t ifType) nogil:
		cdef double act, tG
		if ifType == 0: tG = self.gBag1
		if ifType == 1: tG = self.gBag2A1
		if ifType == 2: tG = self.gChainA1
		act = tG * (T/self.K_SR - (self.LN_SR - self.L0_SR))
		if act < 0 : act = 0
		return act

	cdef double affII(self,double L,double T,Py_ssize_t ifType) except -1:
		cdef double t1, t2, t3, t4, act, tG
		if ifType == 1: tG = self.gBag2A2
		if ifType == 2: tG = self.gChainA2
		t1 = self.X * self.Lsec / self.L0_SR
		t2 = (T/self.K_SR - (self.LN_SR - self.L0_SR))
		t3 = (1 - self.X) * self.Lsec/self.L0_PR
		t4 = (L - T/self.K_SR - self.L0_SR - self.LN_PR)
		act = tG * (t1 * t2 + t3 * t4)
		if act < 0 : act = 0
		return act

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef double occlusion(self) except -1:
		cdef double B2C
		B2C = self.IaMid[1][self.tInt] + self.IaMid[2][self.tInt]
		if B2C >= self.IaMid[0][self.tInt]:
			return B2C + self.S*self.IaMid[0][self.tInt]
		else:
			return self.IaMid[0][self.tInt] + self.S*B2C

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	def integrate(	self,double L,double V,double A,double gDyn,
					double gStat):
		self.runge(L, V, A, gDyn, gStat)
		self.tInt = self.tInt + 1
		return self.Ia[self.tInt -1], self.II[self.tInt -1]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void runge(	self,double L,double V,double A,double gDyn,
						double gStat) except *:
		cdef Py_ssize_t i
		self.aBag1[self.tInt+1] = self.rk4Bag1(	self.time[self.tInt],
												self.aBag1[self.tInt],
												gDyn)
		self.aBag2[self.tInt+1] = self.rk4Bag2(	self.time[self.tInt],
												self.aBag2[self.tInt],
												gStat)
		self.aChain[self.tInt]  = self.dfChain(gStat)
		for i in prange(3, nogil=True):
			self.T[i][self.tInt+1],self.dT[i][self.tInt+1]=self.rk4z(
				self.time[self.tInt], self.T[i][self.tInt], 
				self.dT[i][self.tInt], L, V, A, i)

			self.IaMid[i][self.tInt]=self.affIa(self.T[i][self.tInt],i)
		
		for i in range(1,3):
			self.II[self.tInt]=self.II[self.tInt]+self.affII(L,self.T[i][self.tInt],i)
		self.Ia[self.tInt]    = self.occlusion()

	cdef double rk4Bag1(self,double t,double y,double arg1) except? -1:
		cdef double k1, k2, k3, k4
		k1 = self.dfdtBag1(t, y,arg1)
		k2 = self.dfdtBag1(t+self.dt/2,y+self.dt*k1/2,arg1)
		k3 = self.dfdtBag1(t+self.dt/2,y+self.dt*k2/2,arg1)
		k4 = self.dfdtBag1(t+self.dt, y + self.dt*k3,arg1)
		return y+self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

	cdef double rk4Bag2(self,double t,double y,double arg1) except? -1:
		cdef double k1, k2, k3, k4
		k1 = self.dfdtBag2(t, y,arg1)
		k2 = self.dfdtBag2(t+self.dt/2,y+self.dt*k1/2,arg1)
		k3 = self.dfdtBag2(t+self.dt/2,y+self.dt*k2/2,arg1)
		k4 = self.dfdtBag2(t+self.dt, y +self.dt*k3,arg1)
		return y+self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef (double, double) rk4z(	self,double t,double y,double z, 
								double L,double V,double A, 
								Py_ssize_t i) nogil:
		cdef double temp1, temp2
		self.k1y[i] = self.dTen(t,y, z)
		self.k2y[i] = self.dTen(t+self.dt/2, y+self.dt*self.k1y[i]/2, 
								z+self.dt*self.k1z[i]/2)
		self.k3y[i] = self.dTen(t+self.dt/2, y+self.dt*self.k2y[i]/2,
								z+self.dt*self.k2z[i]/2)
		self.k4y[i] = self.dTen(t+self.dt, y + self.dt*self.k3y[i], 
								z+self.dt*self.k3z[i])
		temp1 = self.k1y[i]+2*self.k2y[i]+2*self.k3y[i]+self.k4y[i]
		y = y + self.dt*(temp1)/6
		self.k1z[i] = self.ddTen(t ,y, z, L, V, A, i)
		self.k2z[i] = self.ddTen(t+self.dt/2,y+self.dt*self.k1y[i]/2, 
								z+self.dt*self.k1z[i]/2, L, V, A, i)
		self.k3z[i] = self.ddTen(t+self.dt/2,y+self.dt*self.k2y[i]/2,
								z+self.dt*self.k2z[i]/2, L, V, A, i)
		self.k4z[i] = self.ddTen(t+self.dt, y + self.dt*self.k3y[i], 
								z+self.dt*self.k3z[i], L, V, A, i)
		temp2 = self.k1z[i]+2*self.k2z[i]+2*self.k3z[i]+self.k4z[i]
		z = z + self.dt*(temp2)/6
		return y, z