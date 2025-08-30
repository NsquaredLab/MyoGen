# cython: language_level=3str
import numpy as np
from scipy.optimize import newton
from scipy.signal import lfilter
from cython.parallel import prange
cimport cython
from libc.math cimport exp, log, fabs, asin, sin, cos

cpdef getLHold(double iArtAng, dict hillD):
	cdef double dt 		= 0.0125 	# [ms] Integration time step 	
	cdef int 	tstop 	= 1000 		# [ms] Simlation time
	cdef double[::1] t =  np.arange(0,tstop + dt, dt)
	cdef double[::1] artAng = np.interp(t,[0,tstop],[iArtAng,iArtAng])
	cdef double lHold
	cdef Py_ssize_t i 
	cdef object mus	= _HillMuscleModel__Cython(tstop,dt,hillD,1, 1, artAng0=iArtAng,L0=1)
	for i in range(len(t)-1):
		_ = mus.integrate(artAng[i])
	lHold = mus.L[-1]
	return lHold

class ForceSatParams():
	def __init__(self,dict hillD,int Ntype1,int Ntype2):
		self.RP = hillD["RP"]
		self.fP = hillD["fP"]
		self.RT = hillD["RT"]
		self.durType = hillD["durType"]
		self.Tl = hillD["Tl"]
		self.satType = hillD["satType"]
		self.fsatf =hillD["fsatf"]
		self.lsatf = hillD["lsatf"]
		self.N      = Ntype1 + Ntype2   # Total Number of motor units
		self.P      = self.fPeakAmp()   # Gen. Peak force amplitude
		self.T      = self.fTwitchTime()# Gen. Interval to peak force
		self.c = np.zeros(len(self.P))
		self.tetF = np.zeros(len(self.P))
		self.muSatFreq = self.satInterp()
		self.mapTetanic()

	def fPeakAmp(self):
		# FUNCTION NAME: fPeakAmp
		# FUNCTION DESCRIPTION: Generates the reference twitch peak 
		#                       force for each motor unit in the pool. 
		P = np.zeros(self.N)
		b = np.log(self.RP)/self.N
		for i in range(1, self.N + 1):
			P[i-1] = np.exp(b * i)
		return P

	def  fTwitchTime(self):
		# FUNCTION NAME: fTwitchTime
		# FUNCTION DESCRIPTION: Generates the array for all motor 
		#                       units in the pool with the reference 
		#                       contraction time that takes for the 
		#                       twitch force reach its peak.
		if (self.durType == 1):
			T = np.zeros(self.N)
			c = log(self.RP)/log(self.RT)
			for i in range(1, self.N + 1):
				T[i-1]=self.Tl*(1/self.P[i-1])**(1/c)
		else:
			T = np.random.uniform(self.Tl/self.RT, self.Tl,self.N)
		return T

		# FUNCTION NAME: sat_interpol
		# FUNCTION DESCRIPTION: Interpolate with different styles between 
		#                       first and last MU force saturation freq.
	def satInterp(self):
		satDif = self.lsatf- self.fsatf
		satSum = self.lsatf + self.fsatf
		N = self.N
		if self.satType == 1:
			muSatFreq = np.linspace(self.fsatf,self.lsatf,N)
		if self.satType == 2:
			a=np.log(30)/N
			rte = np.zeros(N)
			muSatFreq = np.zeros(N)
			for i in range(N):
				rte[i]=np.exp(a*(i+1))
				muSatFreq[i] = rte[i]*(satDif)/30 + self.fsatf
		if self.satType == 3:
			mu_ind = np.linspace(1,N,N)
			muSatFreq = (satDif)*(1-np.exp(-mu_ind*5/N)) + self.fsatf
		if self.satType == 4:
			mu_ind = np.linspace(-int(N/4),int(3*N/4),N)
			muSatFreq = (satDif)/2*self.sig(0.2,mu_ind) + (satSum)/2
		return muSatFreq

		# FUNCTION NAME: newton_f
		# FUNCTION DESCRIPTION: function used by the secant method to find
		#                       the zero and c parameter used in the 
		#                       sigmoidal (sig) function.
		# INPUT PARAMS:  1) c: parameter used to adjust the the sigmoidal 
		#                   function so the motor unit twitch force 
		#                   saturates at defined motor unit discharge rate
		#                2) force: The simulated motor unit force
		# OUTPUT PARAMS: 1) s_max: returns the result of the function used
		#                   by the secant method (which should be nearest 
		#                   zero value)
	def newton_f(self,c,force):
		expfc = np.exp(-force*c)
		s_max= max((1-expfc)/(1+expfc)) -0.999
		return s_max

		# FUNCTION NAME: convMuForce
		# FUNCTION DESCRIPTION: This function generates the motor unit 
		#                       force over the simulation time. This 
		#                       function is like the original proposed by 
		#                       Cisi and Kohn (2008).
		# INPUT PARAMS:  1) mu_spikes: MN discharge time List.
		#                2) i : motor unit index [int]
		# OUTPUT PARAMS: 1) mu_force: Motor unit force over time.
	def convMuForce(self,mu_spikes,P,T):
		dt = 0.05
		t = np.arange(0,3e3+dt,dt)
		mu_force = np.zeros(len(t))
		spikes = np.zeros(len(t))
		for spike_times in mu_spikes:
			index = int(spike_times/dt)
			if index >= len(t):
				index = len(t) -1
			spikes[index]= 1/dt
		B = np.array([0,P*dt**2/T*np.exp(1-dt/T)])
		A = np.array([1, -2*np.exp(-dt/T), np.exp(-2*dt/T)])
		mu_force = lfilter(B,A,spikes) 
		return mu_force

	def mapTetanic(self):
		for i in range(self.N):
			self.c[i], self.tetF[i] = self.defTetanicParam(i,0.2)    

		# FUNCTION NAME: defTetanicParam
		# FUNCTION DESCRIPTION: Find the parameter c which saturates the 
		#                       motor unit force at a given frequency.
		# INPUT PARAMS:  1) i: motor unit index
		#                4) c_init: c initial guess used in the secant 
		#                   (newton) method function
		# OUTPUT PARAMS: 1) n_c: c value in which the sigmoidal function 
		#                   will saturate the motor unit force.
		#                2) fsat_freq1_max: maximum force generated by the 
		#                   motor unit after it is passed by the sigmoidal 
		#                   (sig) function.
	def defTetanicParam(self,i,c_init):
		spikes = np.arange(0,3e3,1e3/self.muSatFreq[i])
		mu_force = self.convMuForce(spikes,1,self.T[i])
		n_c = newton(self.newton_f,c_init,args=(mu_force,),tol=1e-5)
		muF1 = self.convMuForce(np.arange(0,3e3,1e3/1),1,self.T[i])
		fsat_freq1_max = max(self.sig(n_c,muF1))
		return n_c,fsat_freq1_max  

		# FUNCTION NAME: sig
		# FUNCTION DESCRIPTION: Sigmoidal function used to simulate the 
		#                       non-linear relation-ship between motor 
		#                       unit discharge rate and generated motor 
		#                       unit twitch force;
		# INPUT PARAMS:  1) c: parameter used to adjust the the sigmoidal 
		#                   function so the motor unit twitch force 
		#                   saturates at a determined motor unit discharge
		#                   rate
		#                2) force: motor unit force over time. 
		# OUTPUT PARAMS: 1) Saturated motor unit force
	def sig(self,c,force):
		expfc = np.exp(-force*c)
		return (1-expfc)/(1+expfc)

cdef class _HillMuscleModel__Cython():
	# Muscle parameters
	# Source: Elias, A.E. PhD tesis, 2013, pg 63
	# Based on the work of [Arnold et al., 2010; Brown et al., 1996;
	#                       Cheng et al., 2000; De Vlugt et al., 2012;
	#                       Delp et al., 2007; Thelen, 2003]
	cdef double alfa0 	# [rad] Initial pennation angle 
	cdef readonly double F0		# [N] Max. isometric force ~ to FCSA
	cdef readonly double L0		# [m] Fascicle optimal length
	cdef double m 		# [Kg] muscle mass 
	cdef double Kpe		# [F0/L0] Passive elastic element-muscle fiber
	cdef double b 		# [F0.s/L0] Muscle fiber Pas. viscous element
	cdef double Em_0   	# Normalized muscle deformation

	# Tendon [T] parameters
	cdef double LT_0 	# [m] T length for maximum isometric force
	cdef double Kse  	# [F0/LT_0] T Non-linear elastic element
	cdef double cT   	# [n.a] Toe region coefficient
	cdef double LT_r 	# [LT_0]F-L Init of T-aponeorosis linear reg.        

	# Force-Length (F-L) and Force-Velocity (F-V) curve parameters
	# for different types of muscle fibers (MF)
	# Source: Elias, A.E. PhD tesis, 2013, pg 65
	# Based on the work of [Brown et al, 1999; Cheng et al., 2000]
	cdef double b1		# F-L Type I MF param.
	cdef double o1		# F-L Type I MF param.
	cdef double r1		# F-L Type I MF param.
	cdef double b2 		# F-L type II MF param.
	cdef double o2 		# F-L type II MF param.
	cdef double r2  	# F-L type II MF param.
	cdef double Vmax1   # F-V Type I MF param.
	cdef double av01	# F-V Type I MF param.
	cdef double av11 	# F-V Type I MF param.
	cdef double av21 	# F-V Type I MF param.
	cdef double bv1 	# F-V Type I MF param.
	cdef double cv01 	# F-V Type I MF param.
	cdef double cv11 	# F-V Type I MF param.
	cdef double Vmax2 	# F-V Type II MF param.
	cdef double av02  	# F-V Type II MF param.
	cdef double av12 	# F-V Type II MF param.
	cdef double av22 	# F-V Type II MF param.
	cdef double bv2     # F-V Type II MF param.
	cdef double cv02  	# F-V Type II MF param.
	cdef double cv12  	# F-V Type II MF param.

	# Muscle-Tendinius Length  (MTL) fitted curve coefficients
	# Elias, A.E. PhD tesis, 2013, pg 63
	# Similar procedure adopted by Menegaldo et al. (2004)
	cdef double[5] Ak
	# Moment Arm Coefficients
	cdef double[5] Bk

	# Motor unit twitch force amplitude params[Fuglevand et al., 1993]
	cdef int RP  		# Range of twitch force amplitude
	cdef double fP		# First peak twitch force [mN]
	# Motor unit twitch duration params [Fuglevand et al., 1993]
	cdef double RT 		# Range of contraction time till peak
	cdef int durType 	# distribution of contraction times. if 
						# durType == 1, exponential distribution, else
						# will be uniformily random distributed.      
	cdef double Tl      # Longest interval for twitch force reach its 
						# peak [ms]
	cdef int satType  	# Options are: 1:'lin',2:'exp',3:'cap',4:'sig'
	cdef double fsatf   # First recruited saturation frequency [imp/s]
	cdef double lsatf   # Last recruited saturation frequency [imp/s]
	cdef int LR 	   	# Last recruited
	cdef int tInt  		# Integration step
	cdef double dt 		# time step [ms]
	cdef int Ntype1 	# Number of typeI motor units
	cdef int Ntype2     # Number of typeII motor units
	cdef int N			# Total Number of motor units
	cdef double[::1] P 	# Gen. Peak force amplitude distribution
	cdef double[::1] T 	# Gen. Interval to peak force distribution
	cdef double[::1] c 	# c value for sigmoidal function
	cdef double[::1] tet_f 		# maximum force generated by 
								# the motor unit [a.u.]
	cdef double[::1] muSatFreq 	# value in which the sigmoidal 
								# function will saturate MU force.
	cdef double[::1] twiAmp 	# Normalized twitch amps.
	cdef double k1y
	cdef double k2y
	cdef double k3y
	cdef double k4y
	cdef double k1z
	cdef double k2z
	cdef double k3z
	cdef double k4z
	cdef readonly double[::1] time 	# time array [ms]
	cdef readonly double[:,::1] f 	# muForce Matrix
	cdef readonly double[::1] F1 	# sum of type1 forces
	cdef readonly double[::1] F2 	# sum of type1 forces
	cdef readonly double[::1] L 	# Muscle length [L0]
	cdef readonly double[::1] V 	# Muscle velocity [L0/s]
	cdef readonly double[::1] A 	# Muscle aceleration [L0/s^2]
	cdef readonly double[::1] force # Muscle torque [F0]
	cdef readonly double[::1] torque# Muscle torque [F0.m]
	cdef list muSpk 				# Spikes list
	cdef list muId 					# motor unit ID

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	def __init__(self, double tstop,double dt,dict hillD,int Ntype1,
				int Ntype2, double artAng0, double L0):
		assert hillD != None
		assert tstop>0
		assert dt>0
		assert tstop > dt
		cdef object params
		cdef Py_ssize_t tlen, i
		cdef double twiSum = 0
		params = ForceSatParams(hillD,Ntype1,Ntype2)
		self.alfa0  = hillD["alfa0"]
		self.F0     = hillD["F0"]
		self.L0     = hillD["L0"]
		self.m      = hillD["m"]
		self.Kpe    = hillD["Kpe"]
		self.b      = hillD["b"]
		self.Em_0  	= hillD["Em_0"] 
		self.LT_0 	= hillD["LT_0"]
		self.Kse  	= hillD["Kse"]
		self.cT   	= hillD["cT"]
		self.LT_r 	= hillD["LT_r"]     
		self.b1    	= hillD["b1"]
		self.o1   	= hillD["o1"]
		self.r1   	= hillD["r1"]
		self.b2    	= hillD["b2"]
		self.o2   	= hillD["o2"]
		self.r2   	= hillD["r2"]
		self.Vmax1 	= hillD["Vmax1"]
		self.av01  	= hillD["av01"]
		self.av11 	= hillD["av11"]
		self.av21 	= hillD["av21"]
		self.bv1   	= hillD["bv1"]
		self.cv01 	= hillD["cv01"]
		self.cv11 	= hillD["cv11"]
		self.Vmax2 	= hillD["Vmax2"]
		self.av02  	= hillD["av02"]
		self.av12 	= hillD["av12"]
		self.av22 	= hillD["av22"]
		self.bv2   	= hillD["bv2"]
		self.cv02 	= hillD["cv02"]
		self.cv12 	= hillD["cv12"]
		self.Ak 	= hillD["Ak"]
		self.Bk 	= hillD["Bk"]
		self.RP   	= hillD["RP"]
		self.fP     = hillD["fP"]
		self.RT     = hillD["RT"]
		self.durType= hillD["durType"]
		self.Tl    	= hillD["Tl"]
		self.satType= hillD["satType"]
		self.fsatf 	= hillD["fsatf"]
		self.lsatf 	= hillD["lsatf"]
		self.muSpk 	= [] 
		self.muId  	= [] 
		self.LR    	= 0 
		self.tInt  	= 0 
		#Simulation configuration
		self.dt      = dt   # time step [ms]
		self.time    = np.arange(0,tstop+dt,dt,dtype= np.double)
		tlen         = self.time.shape[0] 	# Size of time vector
		self.Ntype1 = Ntype1
		self.Ntype2 = Ntype2
		self.N      = Ntype1 + Ntype2
		if self.N > 0 :
			self.fPeakAmp()   
			self.fTwitchTime()
		self.f     	= np.zeros((self.N,tlen),dtype= np.double)
		self.twiAmp = np.zeros(self.N,dtype = np.double)
		self.c 		= np.zeros(self.N,dtype= np.double) 
		self.tet_f 	= np.zeros(self.N,dtype= np.double)
		self.F1     = np.zeros(tlen,dtype= np.double)
		self.F2     = np.zeros(tlen,dtype= np.double)
		self.L      = np.zeros(tlen,dtype= np.double)
		self.V      = np.zeros(tlen,dtype= np.double)
		self.A      = np.zeros(tlen,dtype= np.double)
		self.force 	= np.zeros(tlen,dtype= np.double)
		self.torque = np.zeros(tlen,dtype= np.double)
		self.k1y    = 0
		self.k2y    = 0
		self.k3y    = 0
		self.k4y    = 0
		self.k1z    = 0
		self.k2z    = 0
		self.k3z    = 0
		self.k4z    = 0
		if L0 == -1:
			self.L[0] = getLHold(artAng0, hillD)
		else:
			assert L0 > 0.7
			assert L0 < 1.3
			self.L[0] = L0
		self.c 		= params.c
		self.tet_f 	= params.tetF
		for i in range(self.N):  
			self.twiAmp[i] = self.fP*self.P[i]/self.tet_f[i]
		for i in range(self.N):
			twiSum = twiSum + self.fP*self.P[i]
		for i in range(self.N):
			self.twiAmp[i] = self.twiAmp[i]/twiSum

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void fPeakAmp(self):
		# FUNCTION NAME: fPeakAmp
		# FUNCTION DESCRIPTION: Generates the reference twitch peak 
		#                       force for each motor unit in the pool. 
		cdef double b
		cdef Py_ssize_t i
		self.P = np.zeros(self.N, dtype= np.double)
		b = log(self.RP)/self.N
		for i in range(1, self.N + 1):
			self.P[i-1] = exp(b * i)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void fTwitchTime(self):
		# FUNCTION NAME: fTwitchTime
		# FUNCTION DESCRIPTION: Generates the array for all motor 
		#                       units in the pool with the reference 
		#                       contraction time that takes for the 
		#                       twitch force reach its peak.
		cdef double c
		cdef Py_ssize_t i
		if (self.durType == 1):
			self.T = np.zeros(self.N,dtype= np.double)
			c = log(self.RP)/log(self.RT)
			for i in range(1, self.N + 1):
				self.T[i-1]=self.Tl*(1/self.P[i-1])**(1/c)
		else:
			self.T = np.random.uniform(self.Tl/self.RT,self.Tl,self.N)

		# FUNCTION NAME: sig
		# FUNCTION DESCRIPTION: Sigmoidal function used to simulate the 
		#                       non-linear relation-ship between motor 
		#                       unit discharge rate and generated motor 
		#                       unit twitch force;
		# INPUT PARAMS:  1) c: parameter used to adjust the the sigmoidal 
		#                   function so the motor unit twitch force 
		#                   saturates at a determined motor unit discharge
		#                   rate
		#                2) force: motor unit force over time. 
		# OUTPUT PARAMS: 1) Saturated motor unit force
	cdef double sig(self,double c,double force) nogil:
		cdef double expfc
		expfc = exp(-force*c)
		return (1-expfc)/(1+expfc)

		# FUNCTION NAME: fPE
		# FUNCTION DESCRIPTION: Calculates the passive element contractile
		#                       force contribution to muscle fibers
		# INPUT PARAMS:  1) LM: Muscle Length
		#                2) V : Muscle velocity
		# OUTPUT PARAMS: 1) Fpe: Passive muscle force
	cdef double fPE(self,double LM,double V):
		cdef double term1
		term1 = exp(self.Kpe*(LM-1)/self.Em_0)
		return term1/exp(self.Kpe) + self.b*V

		# Contractile element force:
		# input: 	1) a1 : activation signal type I, i.e., a1(t)
		#           2) a2 : activation signal type II, i.e., a2(t)
		#		 	3) LM : Muscle Length
		#		 	4) V : Muscle velocity
		# output:	1) Fce: Contractile element force
	cdef double fCE(self, double a1,double a2,double LM,double V):
		return a1*self.fCE1(LM, V) + a2*self.fCE2(LM, V)

	def getFCE1(self,double LM,double V):
		return self.fCE1(LM,V)

	cdef double fCE1(self,double LM,double V):
		cdef double Fl1, Fv1
		Fl1 = self.fL(LM, self.b1, self.o1, self.r1)
		Fv1 = self.fV(LM, V, self.bv1, self.av01, self.av11, 
			self.av21, self.cv01, self.cv11, self.Vmax1)
		return Fl1*Fv1

	def getFCE2(self,double LM,double V):
		return self.fCE2(LM,V)

	cdef double fCE2(self,double LM,double V):
		cdef double Fl2, Fv2
		Fl2 = self.fL(LM, self.b2, self.o2, self.r2)
		Fv2 = self.fV(LM, V, self.bv2, self.av02, self.av12, 
			self.av22, self.cv02, self.cv12, self.Vmax2)
		return Fl2*Fv2 

	cdef double fL(self,double LM,double b,double o,double r):
		return exp(-(fabs((LM**b - 1)/o))**r)

	cdef double fV(	self,double LM,double V,double bv,double av0,
					double av1,double av2,double cv0,double cv1,
					double Vmax):
		cdef double fv
		if V > 0 :
		    #o termo av2*LM**2 está diferente do original, que define 
		    #como uma função. Confirmar essa informação.
			fv = (bv- V*(av0 + av1*LM + av2*LM**2))/(bv +V)
		else:
			fv = (Vmax - V)/(Vmax+ V*(cv0 + cv1*LM))
		return fv

		# tendon unit length []
		# output:			1) Tendon length
	cdef double LT_length(self,double art_angle,double LM):
		cdef double term2
		term2 = LM*self.L0*cos(self.penn(LM))
		return  (self.MTU_length(art_angle) - term2)/self.LT_0

		# Serial element force
		# input				1) LT: tendon length
	cdef double fSE(self,double art_angle,double L):
		cdef double LT, logTerm
		LT = self.LT_length(art_angle, L)
		logTerm = exp((LT - self.LT_r)/self.cT) + 1
		return self.Kse*self.cT*log(logTerm)

		#Pennation angle between fibers and tendon
	cdef double penn(self,double LMnorm):
		return asin(sin(self.alfa0)/LMnorm)

		# Musculotendon unit length []
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef double MTU_length(self,double art_angle):
		cdef double result = 0
		cdef Py_ssize_t i
		for i in range(5):
			result = result + self.Ak[i]*(art_angle**i) 
		return result

		# Serial element force
		# input				1) LT: tendon length
	def getfSE(self,double art_angle,double L):
		return self.fSE(art_angle,L)

	def getMTUlength(self,double art_angle):
		return self.MTU_length(art_angle)
	
	def getPennAng(self,double LMnorm):
		return asin(sin(self.alfa0)/LMnorm)

	def getLTlength(self, double art_angle,double LM):
		return self.LT_length(art_angle,LM)


	cdef double dLdt(self,double t,double y,double z):
		return z

	cdef double dVdt(	self,double t,double L,double V,double A1,
						double A2,double art_angle):
		cdef double t1,t2,t3,t4
		t1 = self.F0/self.m
		t2 = self.fSE(art_angle, L)* cos(self.penn(L))
		t3 = (self.fCE(A1, A2, L, V) + self.fPE(L, V))
		t4 = (cos(self.penn(L)))**2
		return t1*(t2 - (t3*t4))

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef double f_n(self,Py_ssize_t i,double f1,double f2) nogil:
		""" FUNCTION NAME: f_n
		FUNCTION DESCRIPTION: Calculates force at simulation step n,
		                      i.e., f[n]
		INPUT PARAMS:  1) i : Motor unit index number
		               2) t : time integration instant [#ms]
		               3) f1: f[n-1] force at step n-1
		               4) f2: f[n-2] force at step n-2
		OUTPUT PARAMS: 1) f : f[n]   force at step n """
		cdef double t1,t2,t3,t4
		t1 = 2*exp(-self.dt/self.T[i]) * f1
		t2 = exp(-2*self.dt/self.T[i]) * f2
		t3 = exp(1- self.dt/self.T[i])
		t4 = (self.dt**2)/self.T[i]*t3
		return t1 - t2 + t4*self.dirac(i)

	def getMomentArm(self,double art_angle):
		return self.moment_arm(art_angle)

		#Moment arm of the muscle as a function of the articulation angle
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef double moment_arm(self,double art_angle)nogil:
		cdef Py_ssize_t i
		cdef double result = 0
		for i in range(5):
			result = result +self.Bk[i]*(art_angle**i)
		return result

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void compForce(self,double artAng):
		self.force[self.tInt] = self.fSE( artAng, self.L[self.tInt])
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void compTorque(self, double artAng):
		self.torque[self.tInt]=self.force[self.tInt]*self.moment_arm(
											artAng)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef double dirac(self,Py_ssize_t i)nogil:
		cdef Py_ssize_t index
		with gil:
			if i in self.muId:
				index = self.muId.index(i)
				if self.muSpk[index] <= self.time[self.tInt]:
					self.muId.pop(index)
					self.muSpk.pop(index)
					return 1/self.dt
				else:
					return 0
			else:
				return 0

		"""
		FUNCTION NAME: addSpike
		FUNCTION DESCRIPTION: Put motoneuron id (which discharged) on the 
		                  list for integration
		INPUT PARAMS:  1) muId : Motoneuron id. (int)
		"""
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	def addSpike(self,Py_ssize_t muId,double delay):
		assert muId < self.N
		self.muId.append(muId)
		self.muSpk.append(self.time[self.tInt] + delay)
		if self.LR < muId:
			self.LR = muId

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	def integrate(self, double artAngle):
		cdef double L, V, A
		if self.N >0:
			self.forceIntegration()
		self.runge(artAngle)
		L = self.L[self.tInt]
		V = self.V[self.tInt]
		A = self.A[self.tInt]
		self.compForce(artAngle)
		self.compTorque(artAngle)
		self.tInt = self.tInt + 1
		return L, V, A

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void forceIntegration(self):
		cdef Py_ssize_t t = self.tInt
		cdef Py_ssize_t j
		cdef double fsat
		for j in prange(self.LR+1,nogil=True):
			if t == 0:
				self.f[j][t+1] = self.f_n(j, 0, 0)
			elif t == 1:
				self.f[j][t+1] = self.f_n(j, self.f[j][t], 0)
			else:
				self.f[j][t+1] = self.f_n(	j, self.f[j][t],
											self.f[j][t-1])

			fsat = self.twiAmp[j]*self.sig(self.c[j],self.f[j][t+1])
			if (j < self.Ntype1):
				self.F1[t+1] = self.F1[t+1] + fsat
			else:
				self.F2[t+1] = self.F2[t+1] + fsat

	cdef (double,double) rk4z(	self,double t,double y,double z,
								double F1,double F2,double angle):
		cdef double dt,t1,t2
		dt = self.dt*1e-3
		self.k1y = self.dLdt(t , y, z)
		self.k2y = self.dLdt(	t + dt/2, y + dt*self.k1y/2, 
								z + dt*self.k1z/2)
		self.k3y = self.dLdt(	t + dt/2, y + dt*self.k2y/2,
								z + dt*self.k2z/2)
		self.k4y = self.dLdt(	t + dt  , y + dt*self.k3y, 
								z + dt*self.k3z)
		t1 = self.k1y + 2*self.k2y + 2*self.k3y + self.k4y
		y = y + dt*(t1)/6
		self.k1z = self.dVdt(t , y, z, F1, F2, angle)
		self.k2z = self.dVdt(	t + dt/2, y + dt*self.k1y/2, 
								z + dt*self.k1z/2,  F1, F2, angle)
		self.k3z = self.dVdt(	t + dt/2, y + dt*self.k2y/2,
								z + dt*self.k2z/2, F1, F2, angle)
		self.k4z = self.dVdt(	t + dt  , y + dt*self.k3y, 
								z + dt*self.k3z,  F1, F2, angle)
		t2 = self.k1z + 2*self.k2z + 2*self.k3z + self.k4z
		z = z + dt*(t2)/6
		return y, z

		#  Forth order Runge-Kutta
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.initializedcheck(False)
	cdef void runge(self, double artAng):
		cdef Py_ssize_t t
		cdef double time
		t = self.tInt
		time = self.time[t]
		self.L[t+1], self.V[t+1] = self.rk4z(time,self.L[t],self.V[t],
								self.F1[t],self.F2[t], artAng)
		self.A[t+1] = (self.V[t+1] - self.V[t])/(self.dt*1e-3)


