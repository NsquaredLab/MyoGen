# cython: language_level=3str
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt, welch, get_window, spectrogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from cython.parallel import prange
from libc.math cimport exp, log, fabs, asin, sin, cos
import cython
cimport numpy as np

@cython.binding(True)
def  exp_crescent( x, a, b, c):
	return a * np.exp(b * np.array(x)) + c

@cython.binding(True)
def  exp_decrescent(x, a, b, c):
	return a * np.exp(-b *  np.array(x)) + c

cdef double[::1] expInterp(first,last,n,curv=0.33):
	cdef double[4] x
	cdef np.ndarray[double, ndim=1] xp,yn,popt
	cdef double[::1] param

	assert curv <= 0.45
	x = [0,2,4,4]
	xp = np.linspace(0,4,n)
	if first <= last:
		yn = np.array([first,first+(last-first)*curv,last,last])/first
		popt, _ = curve_fit(exp_crescent, x, yn)
		param = exp_crescent(xp,popt[0],popt[1],popt[2])*first
	else:
		yn = np.array([first,last+(first-last)*curv,last,last])/last
		popt, _ = curve_fit(exp_decrescent, x, yn)
		param = exp_decrescent(xp,popt[0],popt[1],popt[2])*last
	return param

cdef class _sEMG__Cython():
	cdef int 			morpho 	# [1-4] -> [ring,circle,pizza,ellipse]
	cdef int 			fs     	# [Hz] Sampling Frequency
	cdef int 			n      	# [#] Number of MUs
	cdef int 			t1     	# [#] Number of Type I MUs
	cdef int			t2     	# [#] Number of Type II MUs
	cdef int 			LR 		# [#] Last MU recruited
	cdef int[::1] 		MUFN 	# [#] Array of MU number of fibers
	cdef int[:] 		idvec 	# [#] aMU id array (from neuron)
	cdef double			r 		# [m] muscle radius
	cdef double 		b 		# [m] ellipse muscle morphology param
	cdef double 		a 		# [m] ellipse muscle morphology param
	cdef double 		re 		# [m] Ring muscle morphology param
	cdef double 		ri 		# [m] Ring muscle morphology param
	cdef double 		csa 	# [m] Cross Sectional Area
	cdef double 		fat    	# [m] Adipose tissue tickness
	cdef double 		skin   	# [m] Skin tissue tickness
	cdef double 		theta  	# [rad] Muscle morphology theta param.
	cdef double 		prop   	# [n/a] Muscle morphology prop param.
	cdef double 		first  	# [#] Number of fibers innervated by MU #1
	cdef double 		ratio  	# [n/a] Innervation ratio between first and last MU
	cdef double 		t1m    	# [Mrad] Mean value for Type I MU terr. dist.
	cdef double 		t1dp   	# [Mrad] SD value for Type I MU terr. dist.
	cdef double 		t2m    	# [Mrad] Mean value for Type II MU terr. dist.
	cdef double 		t2dp   	# [Mrad] SD value for Type I MU terr. dist.
	cdef double 		v1     	# [mV] MUAP amplitude Param for MU #1
	cdef double 		v2     	# [mV] MUAP amplitude Param for MU #-1
	cdef double 		d1     	# [ms] MUAP duration param for MU #1
	cdef double 		d2     	# [ms] MUAP duration param for MU #-1
	cdef double 		ampk   	# [mm^-1] Vol. Cond. amp. atten. cte
	cdef double 		durak  	# [mm^-1] Vol. Cond. dur. atten. cte
	cdef double 		noise  	# [mV] sEMG Noise normal dist. SD  
	cdef double 		lc     	# [Hz] Filter low cut
	cdef double 		hc     	# [Hz] Filter high cut
	cdef double 		ecc 	# [mm] Motor unit territory eccentric. 	
	cdef double 		gStep 	# Grid step
	cdef double		 	elec 	# [m] electrode coord.
	cdef double[::1] 	amp 	# [mV] Array of MUAP amplitude factors
	cdef double[::1] 	lm		# [mV] Array of MUAP duration factors
	cdef double[::1] 	emg  	# [mV]sEMG signal 
	cdef double[::1] 	t  		# [ms] Time Vector
	cdef readonly double[::1] 	delay 	# [ms] Spinal Cord-End Plate delay arr
	cdef double[::1] 	circle 	# [m] array to integrate circle
	cdef double[::1] 	ma 		# [m] Muscle boundary coordinate 1
	cdef double[::1] 	mb 		# [m] Muscle boundary coordinate 2
	cdef double[::1] 	fa 		# [m] fat boundary coordinate 1
	cdef double[::1] 	fb 		# [m] fat boundary coordinate 2
	cdef double[::1] 	sa 		# [m] skin boundary coordinate 1
	cdef double[::1] 	sb 		# [m] skin boundary coordinate 2
	cdef double[::1] 	MUrad 	# [m] Motor unit radius
	cdef double[::1] 	x 		# [m] MU terr. center x coord.
	cdef double[::1] 	y 		# [m] MU terr. center y coord.
	cdef double[::1]	gx 		# [] grid x coords.
	cdef double[::1] 	gy 		# [] grid y coords.
	cdef double[::1]	muDist 	# [] Motor unit dist
	cdef double[::1] 	ampV 	# [arr mV] MUAP amplitude pos VolCond
	cdef double[::1] 	lmV 	# [arr mV] MUAP duration pos VolCond
	cdef double[::1] 	spkvec 	# [ms] Spike times array (from neuron)
	cdef double[:] 		rawEMG 	# [mV] raw sEMG array
	cdef double[:,:] 	mu_emg 	# [mV] raw MUAP train
	cdef double[:,:,::1]MUT 	# [m] MUT x,y coord

	cdef bint 			filt 	# Add butterworth band filter to sEMG signal

	def __init__(self,dict cfg):
		cdef double[::1] velcon
		self.morpho 	= cfg['sEMG']['morpho']
		self.csa    	= cfg['sEMG']['csa']
		self.fat    	= cfg['sEMG']['fat']
		self.skin   	= cfg['sEMG']['skin']
		self.theta  	= cfg['sEMG']['theta']
		self.prop   	= cfg['sEMG']['prop']
		self.first  	= cfg['sEMG']['first']
		self.ratio  	= cfg['sEMG']['ratio']
		self.t1m    	= cfg['sEMG']['t1m']
		self.t1dp   	= cfg['sEMG']['t1dp']
		self.t2m    	= cfg['sEMG']['t2m']
		self.t2dp   	= cfg['sEMG']['t2dp']
		self.v1     	= cfg['sEMG']['v1']
		self.v2     	= cfg['sEMG']['v2']
		self.d1     	= cfg['sEMG']['d1']
		self.d2     	= cfg['sEMG']['d2']
		self.ampk   	= cfg['sEMG']['ampk']
		self.durak  	= cfg['sEMG']['durak']
		self.noise  	= cfg['sEMG']['noise']
		self.filt   	= cfg['sEMG']['filt']
		self.lc     	= cfg['sEMG']['lc']
		self.hc     	= cfg['sEMG']['hc']
		self.LR     	= 0
		self.n 			= cfg['pop']['aMN']['n']
		self.amp    	= expInterp(
							cfg['sEMG']['v1'],
							cfg['sEMG']['v2'],
							cfg['pop']['aMN']['n'],
							curv=0.3)
		self.lm     	= expInterp(
							cfg['sEMG']['d1'],
							cfg['sEMG']['d2'],
							cfg['pop']['aMN']['n'],
							curv=0.3) 
		self.fs     	= int(1/(cfg['sim']['dt']*1e-3))
		velcon 			= np.linspace(
							cfg['pop']['aMN']['velcon'][0],
							cfg['pop']['aMN']['velcon'][1],
					 		cfg['pop']['aMN']['n'], 
					 		dtype = np.double)
		self.t1     	= cfg['pop']['aMN']['nType1']
		self.t2     	= cfg['pop']['aMN']['nType2']
		self.MUFN   	= np.zeros(self.n,dtype = np.intc) 
		self.MUrad  	= np.zeros(self.n,dtype = np.double)
		self.muDist 	= np.zeros(self.n,dtype = np.double)
		self.ampV 		= np.zeros(self.n,dtype = np.double)
		self.lmV 	 	= np.zeros(self.n,dtype = np.double)
		self.t      	= cfg['sim']['time']
		self.delay 		= cfg['pop']['aMN']['axonlen']*1e3/np.array(velcon)
		self.defineMorpho()
		self.innervateRatio()
		self.genDistribution()
		self.motorUnitTerritory()
		self.quantification_of_mu_regionalization()
		self.generate_density_grid()
		self.vc_filter()


	def seed(self,int seed):
		np.random.seed(seed)

	# FUNCTION NAME: defineMorpho
	# FUNCTION DESCRIPTION: Defines the muscle morphology and electrode position
	# INPUT PARAMS:  1) morpho: Muscle morphology modeled as the following geometric
	#                   shapes: 'circle', 'ring', 'pizza', 'ellipse' [string]
	#                2) csa: Muscle cross sectional area (m^2) [float]
	#                3) fat: fat tissue thickness (m) [float]
	#                4) skin: skin tissue thickness (m) [float]
	#                5) theta: theta angle which will be used to define the muscle 
	#                   tissue boundaries (rad) [float]
	#                6) prop: for the 'ring' geometry, prop is used as the ratio
	#                   between internal and external muscle radius (m) [float]
	# OUTPUT PARAMS: 1) ma: Muscle boundaries x coordinate (m) [float numpy array]
	#                2) mb: Muscle boundaries y coordinate (m) [float numpy array]
	#                3) fa: fat tissue boundaries x coordinate (m) [float numpy array]
	#                2) fb: fat tissue boundaries y coordinate (m) [float numpy array]
	#                3) sa: skin tissue boundaries x coordinate (m) [float numpy array]
	#                4) sb: skin tissue boundaries y coordinate (m) [float numpy array]
	#                5) elec: electrode position y coordinate (m) [float numpy array]
	cdef void defineMorpho(self):
		if self.morpho == 1: # circle
			self.circle_tissues()
		elif self.morpho == 2: # ring
			self.ring_tissues()
		elif self.morpho == 3: # pizza
			self.pizza_tissues()
		elif self.morpho == 4: # ellipse
			self.prop = 1/self.prop
			self.ellipse_tissues()
		else:
			print('Could not define muscle morphology.')

	# FUNCTION NAME: circle_tissues
	# FUNCTION DESCRIPTION: Draw Circle Tissue limits by creating arrays coordinates
	cdef void circle_tissues(self):
		cdef double[::1] circle
		self.r 	= np.sqrt(self.csa/np.pi)
		circle 	= np.arange(0,2*np.pi,0.01) 
		self.ma = self.r * np.cos(circle)
		self.mb = self.r * np.sin(circle)
		self.fa = (self.r+self.fat)* np.cos(circle)
		self.fb = (self.r+self.fat)* np.sin(circle)
		self.sa = (self.r+self.fat+self.skin)* np.cos(circle)
		self.sb = (self.r+self.fat+self.skin)* np.sin(circle)
		self.elec = self.r+self.fat+self.skin

	# FUNCTION NAME: ring_tissues
	# FUNCTION DESCRIPTION: Draw ring Tissue limits by creating arrays coordinates
	#                       of the tissue boundaries.
	cdef void ring_tissues(self) except *:
		cdef np.ndarray[double, ndim=1] angle
		self.re = np.sqrt((self.csa)/(self.theta*(1-self.prop**2)))
		self.ri = self.re * self.prop
		angle = np.arange(np.pi/2-self.theta,np.pi/2+self.theta,0.01) #0 to 2*pi variation with pase of 0.01 
		self.ma =np.concatenate(([self.ri*np.cos(np.pi/2-self.theta)],self.re \
		                         * np.cos(angle),np.flip(self.ri*np.cos(angle),0)))
		self.mb =np.concatenate(([self.ri*np.sin(np.pi/2-self.theta)],self.re \
		                         * np.sin(angle),np.flip(self.ri*np.sin(angle),0)))
		self.fa = (self.re + self.fat) * np.cos(angle)
		self.fb = (self.re + self.fat) * np.sin(angle)
		self.sa = (self.re + self.fat + self.skin) * np.cos(angle)
		self.sb = (self.re + self.fat + self.skin) * np.sin(angle)
		self.elec = self.re + self.fat + self.skin

	# FUNCTION NAME: pizza_tissues
	# FUNCTION DESCRIPTION: Draw pizza like muscle tissue limits by creating arrays coordinates
	#                       of the tissue boundaries.
	cdef void pizza_tissues(self):
		cdef double angle
		self.r = np.sqrt(self.csa/self.theta)
		angle = np.arange(np.pi/2 - self.theta, np.pi/2 + self.theta, 0.01)
		self.ma = self.r * np.cos(angle)
		self.mb = self.r * np.sin(angle)
		self.ma = np.concatenate(([0],self.ma,[0]))
		self.mb = np.concatenate(([0],self.mb,[0]))
		self.fa = (self.r+self.fat)* np.cos(angle)
		self.fb = (self.r+self.fat)* np.sin(angle)
		self.sa = (self.r+self.fat+self.skin)* np.cos(angle)
		self.sb = (self.r+self.fat+self.skin)* np.sin(angle)
		self.elec = self.r+self.fat+self.skin

	# FUNCTION NAME: ellipse_tissues
	# FUNCTION DESCRIPTION: Draw ellipse like muscle tissue limits by creating arrays 
	#                       of coordinates of the tissue boundaries.
	cdef ellipse_tissues(self):
		cdef double[::1] circle
		self.b = np.sqrt(self.csa/(self.prop*np.pi)) #smaller
		self.a = self.prop*self.b #bigger
		circle = np.arange(0,2*np.pi,0.01) #0 to 2*pi variation with pase of 0.01 
		self.ma = self.a * np.cos(circle) #muscle
		self.mb = self.b * np.sin(circle)
		self.fa = (self.a + self.fat) * np.cos(circle)
		self.fb = (self.b + self.fat) * np.sin(circle)
		self.sa = (self.a + self.fat + self.skin) * np.cos(circle)
		self.sb = (self.b + self.fat + self.skin) * np.sin(circle)
		self.elec = self.b + self.fat + self.skin

	# FUNCTION NAME: innervateRatio
	# FUNCTION DESCRIPTION: Calculates the number of innervated muscle fibers
	#                       for each  motorneuron in the pool and the motor unit
	#                       territory radius. Based on the work of Enoka e Fuglevand, 2001
	cdef void innervateRatio(self):
		cdef int n_fibers
		cdef np.ndarray[double, ndim=1] MUarea
		cdef double fiber_area
		cdef Py_ssize_t i
		n_fibers = 0
		MUarea = np.zeros(len(self.MUFN),dtype = np.double)
		for i in range(self.n):
			self.MUFN[i]=int(self.first*np.exp(np.log(self.ratio)*(i)/self.n))
			n_fibers = n_fibers + self.MUFN[i]
		fiber_area = self.csa/n_fibers
		for i in range(len(self.MUFN)):
			MUarea[i] = self.MUFN[i]*fiber_area
			self.MUrad[i] = np.sqrt(MUarea[i]/np.pi)

	# FUNCTION NAME: gen_distribution
	# FUNCTION DESCRIPTION: Defines the motor units  x and y coordinates 
	# INPUT PARAMS:  1) morpho: Muscle morphology modeled as the following geometric
	#                   shapes: 'circle', 'ring', 'pizza', 'ellipse' [string]
	#                2) t1: Quantity of Type I motor units (MUs) [int]
	#                3) t2a: Quantity of Type IIa MUs [int]
	#                4) t2b: Quantity of Type IIb MUs [int]
	#                5) t1m: Type I MU distance distribution mean (% relative to the
	#                   muscle radius) [float]
	#                6) t1dp: Type I MU distance distribution standard deviation (% 
	#                   relative to the muscle radius) [float]
	#                7) t2m: Type II MU distance distribution mean (% relative to the
	#                   muscle radius) [float]
	#                8) t2dp: Type II MU distance distribution standard deviation (% 
	#                   relative to the muscle radius) [float]
	#                9) csa: Muscle cross sectional area (m^2) [float]
	#               10) proportion: for the 'ring' geometry, prop is used as the ratio
	#                   between internal and external muscle radius (m) [float]
	#               11) theta: theta angle which will be used to define the muscle 
	#                   tissue boundaries (rad) [float]
	#               12) mur: Array with all motor unit territory (MUT) radius [m] 
	#                   [np array]
	# OUTPUT PARAMS: 1) x: All MUT centers x coordinate (m) [float numpy array]
	#                2) y: All MUT centers y coordinate (m) [float numpy array]
	cdef void genDistribution(self):
		if self.morpho == 1: # circle
			self.t1m = self.t1m*self.r
			self.t2m = self.t2m*self.r
			self.circle_normal_distribution_otimize()
		elif self.morpho == 2: # ring
			self.t1m = self.ri + (self.re-self.ri) * self.t1m
			self.t2m = self.ri + (self.re-self.ri) * self.t2m
			self.ring_normal_distribution_otimize()
		elif self.morpho == 3: # pizza
			self.t1m = self.t1m * self.r
			self.t2m = self.t2m * self.r
			self.pizza_normal_distribution_otimize()
		elif self.morpho == 4: # ellipse
			self.ellipse_normal_distribution_otimize()
		else:
			print('Could not identify muscle morphology.')

	# FUNCTION NAME: circle_normal_distribution_otimize
	# FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center coordinates for
	#                       circle cross sectional area (CSA) muscle morpholigies. 
	#                       Verifies MUT placed before to otimize by reducing the 
	#                       distribution variability across CSA.
	# INPUT PARAMS:  1) r: Muscle radius (m) [float]
	#                2) N_s: Quantity of Type I motor units (MUs) [int]
	#                3) N_fr: Quantity of Type IIa MUs [int]
	#                4) N_ff: Quantity of Type IIb MUs [int]
	#                5) D_s : Type I MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list or tuple]
	#                6) D_fr: Type IIa MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list or tuple]
	#                7) D_ff: Type IIb MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list or tuple]
	#                8) mur: Array with all motor unit territory (MUT) radius [m] 
	#                   [np array]
	# OUTPUT PARAMS: 1) x: All MUT centers x coordinate (m) [float numpy array]
	#                2) y: All MUT centers y coordinate (m) [float numpy array]
	def circle_normal_distribution_otimize(self):
		cdef double r_temp
		cdef double t_temp
		cdef double x_temp
		cdef double y_temp
		cdef Py_ssize_t i
		cdef Py_ssize_t j
		cdef double ant_d
		cdef double min_d
		cdef double mur_min
		self.x= np.zeros(self.n)
		self.y= np.zeros(self.n)
		i= self.n - 1
		while (i> self.t1):
			r_temp = np.random.normal(self.t2m,(self.r-self.MUrad[i])*self.t2dp)
			t_temp = np.random.uniform(0,2*np.pi)
			x_temp = r_temp*np.cos(t_temp)
			y_temp = r_temp*np.sin(t_temp)
			if (r_temp <= self.r- self.MUrad[i]) and (r_temp >= 0):
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = self.r
					for j in range(i,self.n):
						d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
						min_d = min(d,ant_d)
						ant_d = min_d
						if min_d == d:
							mur_min = self.MUrad[j]
					if min_d >= self.MUrad[i]+mur_min/2:
						self.x[i] = x_temp
						self.y[i] = y_temp
						i = i - 1
		while (i>=0):
			r_temp = np.random.normal(self.t1m,(self.r-self.MUrad[i])*self.t1dp)
			t_temp = np.random.uniform(0,2*np.pi)
			x_temp = r_temp*np.cos(t_temp)
			y_temp = r_temp*np.sin(t_temp)
			if r_temp <= self.r-self.MUrad[i] and r_temp >= 0:
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = self.r
				for j in range(i,self.n):
					d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
					min_d = min(d,ant_d)
					ant_d = min_d
					if min_d == d:
						mur_min = self.MUrad[j]
				if min_d >= self.MUrad[i]+mur_min/2:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i - 1

	# FUNCTION NAME: motorUnitTerritory
	# FUNCTION DESCRIPTION: Based on motor unit territory centers and muscle radius, 
	#                       creates the motor unit territory boundaries.
	# INPUT PARAMS:  1) x: x coordinate of the MUT centers (m) [float array]
	#                2) y: y coordinate of the MUT centers (m) [float array]
	#                3) MUradius: MUT radius (m) [float array]
	# OUTPUT PARAMS: 1) MUT: MUT boundaries [2d float array]
	cdef void motorUnitTerritory(self) except *:
		cdef np.ndarray[double, ndim=1] theta
		cdef Py_ssize_t i,j
		theta = np.arange(0,2*np.pi,0.01, dtype=np.double)
		self.MUT = np.zeros((self.n,2,len(theta)))
		for i in range(self.n):
			for j in range(len(theta)):
				self.MUT[i][0][j] = self.x[i] + self.MUrad[i]*np.cos(theta[j])
				self.MUT[i][1][j] = self.y[i] + self.MUrad[i]*np.sin(theta[j])

	# FUNCTION NAME: quantification_of_mu_regionalization
	# FUNCTION DESCRIPTION: Calculates the motor unit type II territory 
	#                       radial ecc
	# INPUT PARAMS:  1) x: x coordinate of the MUT centers (m) [float array]
	#                2) y: y coordinate of the MUT centers (m) [float array]
	#                3) n_s: Quantity of Type I motor units [int]
	#                4) n_fr: Quantity of Type IIa motor units [int]
	#                5) n_ff: Quantity of Type IIb motor units [int]
	#                6) mufn: motor units fiber quantity [float numpy array]
	# OUTPUT PARAMS: 1) E: Type II MU ecc (m) [float]
	cdef void quantification_of_mu_regionalization(self):
		cdef int[2] n_f
		cdef int[2] n_i
		cdef double[2] peso
		cdef double[2] r_cg
		cdef Py_ssize_t i,j

		n_f = [self.n,self.n]
		n_i = [0,self.t1]
		peso  = [0,0]
		r_cg = [0,0]
		for j in range(2):
			for i in range (n_i[j],n_f[j]):
				peso[j] = peso[j] + self.MUFN[i]
				r_cg[j] = r_cg[j] + np.sqrt(self.x[i]**2 + self.y[i]**2)*self.MUFN[i]
			r_cg[j] = r_cg[j]/peso[j]
		self.ecc = r_cg[1] - r_cg[0]

	# FUNCTION NAME: generate_density_grid
	# FUNCTION DESCRIPTION: Generates the muscle cross sectional area density
	#                       of motor unit territories (to use with 2d histogram)
	# INPUT PARAMS:  1) mur: motor unit radius (m) [float numpy array]
	#                1) x: x coordinate of the MUT centers (m) [float array]
	#                2) y: y coordinate of the MUT centers (m) [float array]
	#                3) gStep: grid step to create the density matrix
	# OUTPUT PARAMS: 1) gx: x coordinates of the MUT points [numpy array]
	#                2) gy: y coordinates of the MUT points [numpy array]
	cdef void generate_density_grid(self):
		cdef Py_ssize_t i, j, w
		cdef double[::1] x_temp
		cdef double[::1] y_temp
		cdef double Y
		self.gStep = 5e-5 
		self.gx = np.zeros(1,dtype=np.double)
		self.gy = np.zeros(1,dtype=np.double)
		for i in range(self.n):
			x_temp = np.arange(-self.MUrad[i],self.MUrad[i],self.gStep)
			for j in range(len(x_temp)):
				Y = np.sqrt(self.MUrad[i]**2-x_temp[j]**2)
				y_temp = np.arange(-Y,Y,self.gStep)
				for w in range(len(y_temp)):
					y_temp[w] = y_temp[w] +self.y[i]
				self.gx = np.append(self.gx,(x_temp[j]*np.ones(len(y_temp))+self.x[i]))
				self.gy = np.append(self.gy,(y_temp))

	# FUNCTION NAME: ring_normal_distribution_otimize
	# FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center coordinates for
	#                       ring cross sectional area (CSA) muscle morpholigies. 
	#                       Verifies MUT placed before to otimize by reducing the 
	#                       distribution variability across CSA.
	# INPUT PARAMS:  1) re: External muscle radius (m) [float]
	#                2) ri, Internal muscle radius (m) [float]
	#                3) theta: theta angle which will be used to define the muscle 
	#                   tissue boundaries (rad) [float]
	#                4) N_s: Quantity of Type I motor units (MUs) [int]
	#                5) N_fr: Quantity of Type IIa MUs [int]
	#                6) N_ff: Quantity of Type IIb MUs [int]
	#                7) D_s : Type I MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                8) D_fr: Type IIa MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                9) D_ff: Type IIb MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#               10) mur: Array with all motor unit territory (MUT) radius [m] 
	#                   [np array]
	# OUTPUT PARAMS: 1) x: All MUT centers x coordinate (m) [float numpy array]
	#                2) y: All MUT centers y coordinate (m) [float numpy array]
	cdef void ring_normal_distribution_otimize(self):
		cdef double r_temp
		cdef double t_temp
		cdef double x_temp
		cdef double y_temp
		cdef Py_ssize_t i
		cdef Py_ssize_t j
		cdef double theta_c
		cdef double phi_c
		cdef double ant_d
		cdef double min_d
		cdef double mur_min
		cdef double temp
		self.x= np.zeros(self.n)
		self.y= np.zeros(self.n)
		i= self.n - 1
		while (i> self.t1):
			r_temp = np.random.normal(self.t2m,(self.re-self.MUrad[i])*self.t2dp)
			t_temp = np.random.uniform(-self.theta,self.theta)
			x_temp = r_temp*np.sin(t_temp)
			y_temp = r_temp*np.cos(t_temp)
			temp = self.MUrad[i]/r_temp
			if temp >= 1: temp= 1
			if temp <= -1: temp = -1 
			theta_c = np.arcsin(temp)
			phi_c = np.arcsin(x_temp/r_temp)
			if (r_temp <= self.re-self.MUrad[i]) and (r_temp>= self.MUrad[i] + self.ri) \
				and (phi_c <= self.theta - theta_c) and (phi_c >= -(self.theta - theta_c)):
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = self.re
					for j in range(i,self.n):
						d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
						min_d = min(d,ant_d)
						ant_d = min_d
						if min_d == d:
							mur_min = self.MUrad[j]
					if min_d >= self.MUrad[i]+mur_min:
						self.x[i] = x_temp
						self.y[i] = y_temp
						i = i - 1
		while (i>=0):
			#r_temp = r*np.random.uniform()
			r_temp = np.random.normal(self.t1m,(self.re-self.MUrad[i])*self.t1dp)
			t_temp = np.random.uniform(-self.theta,self.theta)
			x_temp = r_temp*np.sin(t_temp)
			y_temp = r_temp*np.cos(t_temp)
			temp = self.MUrad[i]/r_temp
			if temp >= 1: temp  = 1
			if temp <= -1: temp = -1 
			theta_c = np.arcsin(temp)
			phi_c = np.arcsin(x_temp/r_temp)
			if (r_temp <= self.re-self.MUrad[i]) and (r_temp>= self.MUrad[i]+self.ri) \
				and (phi_c <= self.theta - theta_c) and (phi_c >= -(self.theta - theta_c)):
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = self.re
					for j in range(i,self.n):
						d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
						min_d = min(d,ant_d)
						ant_d = min_d
						if min_d == d:
							mur_min = self.MUrad[j]
					if min_d >= self.MUrad[i]+mur_min/2:
						self.x[i] = x_temp
						self.y[i] = y_temp
						i = i - 1

	# FUNCTION NAME: pizza_normal_distribution_otimize
	# FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center coordinates for
	#                       pizza like cross sectional area (CSA) muscle morpholigies. 
	#                       Verifies MUT placed before to otimize by reducing the 
	#                       distribution variability across CSA.
	# INPUT PARAMS:  1) r: muscle radius (m) [float]
	#                2) theta: theta angle which will be used to define the muscle 
	#                   tissue boundaries (rad) [float]
	#                3) N_s: Quantity of Type I motor units (MUs) [int]
	#                4) N_fr: Quantity of Type IIa MUs [int]
	#                5) N_ff: Quantity of Type IIb MUs [int]
	#                6) D_s : Type I MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                7) D_fr: Type IIa MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                8) D_ff: Type IIb MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                9) mur: Array with all motor unit territory (MUT) radius [m] 
	#                   [np array]
	# OUTPUT PARAMS: 1) x: All MUT centers x coordinate (m) [float numpy array]
	#                2) y: All MUT centers y coordinate (m) [float numpy array]
	cdef void pizza_normal_distribution_otimize(self):
		cdef double r_temp
		cdef double t_temp
		cdef double x_temp
		cdef double y_temp
		cdef Py_ssize_t i
		cdef Py_ssize_t j
		cdef double theta_c
		cdef double phi_c
		cdef double ant_d
		cdef double min_d
		cdef double mur_min
		cdef double temp
		self.x= np.zeros(self.n)
		self.y= np.zeros(self.n)
		i= self.n - 1
		while (i> self.t1):
			r_temp = np.random.normal(self.t2m,(self.r-self.MUrad[i])*self.t2dp)
			t_temp = np.random.uniform(-self.theta,self.theta)
			x_temp = r_temp*np.sin(t_temp)
			y_temp = r_temp*np.cos(t_temp)
			theta_c = np.arcsin(self.MUrad[i]/r_temp)
			phi_c = np.arcsin(x_temp/r_temp)
			if (r_temp <= self.r-self.MUrad[i]) and (r_temp >= self.MUrad[i]) \
				and (phi_c <= self.theta - theta_c) and (phi_c >= -(self.theta - theta_c)):
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = self.r
					for j in range(i,self.n):
						d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
						min_d = min(d,ant_d)
						ant_d = min_d
						if min_d == d:
							mur_min = self.MUrad[j]
					if min_d >= self.MUrad[i]+mur_min:
						self.x[i] = x_temp
						self.y[i] = y_temp
						i = i - 1
		while (i>=0):
			#r_temp = r*np.random.uniform()
			r_temp = np.random.normal(self.t1m,(self.r-self.MUrad[i])*self.t1dp)
			t_temp = np.random.uniform(-self.theta,self.theta)
			x_temp = r_temp*np.sin(t_temp)
			y_temp = r_temp*np.cos(t_temp)
			theta_c = np.arcsin(self.MUrad[i]/r_temp)
			phi_c = np.arcsin(x_temp/r_temp)
			if (r_temp <= self.r-self.MUrad[i]) and (r_temp>= self.MUrad[i]) \
				and (phi_c <= self.theta - theta_c) and (phi_c >= -(self.theta - theta_c)):
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = self.r
					for j in range(i,self.n):
						d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
						min_d = min(d,ant_d)
						ant_d = min_d
						if min_d == d:
							mur_min = self.MUrad[j]
					if min_d >= self.MUrad[i]+mur_min/2:
						self.x[i] = x_temp
						self.y[i] = y_temp
						i = i - 1

	# FUNCTION NAME: ellipse_normal_distribution_otimize
	# FUNCTION DESCRIPTION: Generate Motor unit Territory (MUT) center coordinates for
	#                       pizza like cross sectional area (CSA) muscle morpholigies. 
	#                       Verifies MUT placed before to otimize by reducing the 
	#                       distribution variability across CSA.
	# INPUT PARAMS:  1) a: semi-major axis of the muscle tissue boundiries with ellipse
	#                   morphology (m) [float]
	#                2) n: semi-minor axis of the muscle tissue boundiries with ellipse
	#                   morphology (m) [float]
	#                3) N_s: Quantity of Type I motor units (MUs) [int]
	#                4) N_fr: Quantity of Type IIa MUs [int]
	#                5) N_ff: Quantity of Type IIb MUs [int]
	#                6) D_s : Type I MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                7) D_fr: Type IIa MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                8) D_ff: Type IIb MU distance distribution mean (% relative to the
	#                   muscle radius) and standard deviation (% relative to the
	#                   muscle radius) [2 cells list]
	#                9) mur: Array with all motor unit territory (MUT) radius [m] 
	#                   [np array]
	# OUTPUT PARAMS: 1) x: All MUT centers x coordinate (m) [float numpy array]
	#                2) y: All MUT centers y coordinate (m) [float numpy array]
	cdef void ellipse_normal_distribution_otimize(self):
		cdef double r_temp
		cdef double t_temp
		cdef double x_temp
		cdef double y_temp
		cdef double raio
		cdef Py_ssize_t i
		cdef Py_ssize_t j
		cdef double theta_c
		cdef double phi_c
		cdef double ant_d
		cdef double min_d
		cdef double mur_min
		cdef double[::1] temp
		self.x= np.zeros(self.n)
		self.y= np.zeros(self.n)
		i= self.n - 1
		while (i> self.t1):
			t_temp = np.random.uniform(0,2*np.pi)
			raio = self.a * self.b/np.sqrt((self.b*np.cos(t_temp))**2+(self.a*np.sin(t_temp))**2)
			r_temp = np.random.normal(raio*self.t2m,(raio-self.MUrad[i])*self.t2dp)
			x_temp = r_temp*np.sin(t_temp)
			y_temp = r_temp*np.cos(t_temp)
			if r_temp <= raio-self.MUrad[i] and r_temp >= 0:
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = raio
					for j in range(i,self.n):
						d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
						min_d = min(d,ant_d)
						ant_d = min_d
						if min_d == d:
							mur_min = self.MUrad[j]
					if min_d >= self.MUrad[i]+mur_min:
						self.x[i] = x_temp
						self.y[i] = y_temp
						i = i - 1
		while (i>=0):
			t_temp = np.random.uniform(0,2*np.pi)
			raio = self.a*self.b/np.sqrt((self.b*np.cos(t_temp))**2+(self.a*np.sin(t_temp))**2)
			r_temp = np.random.normal(raio*self.t1m,(raio-self.MUrad[i])*self.t1dp)
			x_temp = r_temp*np.sin(t_temp)
			y_temp = r_temp*np.cos(t_temp)
			if r_temp <= raio-self.MUrad[i] and r_temp >= 0:
				if i == 0:
					self.x[i] = x_temp
					self.y[i] = y_temp
					i = i-1
				else:
					ant_d = raio
					for j in range(i,self.n):
						d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
						min_d = min(d,ant_d)
						ant_d = min_d
						if min_d == d:
							mur_min =self.MUrad[j]
					if min_d >= self.MUrad[i]+mur_min/2:
						self.x[i] = x_temp
						self.y[i] = y_temp
						i = i - 1
		temp = self.x
		self.x = self.y
		self.y = temp

	# FUNCTION NAME: vc_filter
	# FUNCTION DESCRIPTION: Apply the filtering effect of the muscle tissue 
	#                       (volume conductor) on the duration and amplitude
	#                       factors of the hermite rodriguez functions
	# INPUT PARAMS:  1) mux: motor unit territory center x coordinates [np array]
	#                2) muy: motor unit territory center y coordinates [np array]
	#                2) electrode: electrode position y coordinate [float]
	#                3) ap_array: HR function amplitude factor array [np array]
	#                4) lm_array: HR function duration factor array [np array]
	#                5) ampk: Volume conductor amplitude attenuation constant 
	#                6) durak: Volume conductor duration widening constant 
	# OUTPUT PARAMS: 1) apvar: HR function amplitude factors attenuated by the
	#                          volume conductor [np array]
	#                2) lmV: HR function duration factors widened by the
	#                          volume conductor filtering effect [np array]
	#                3) muDist: Distance array between motor units and
	#                   electrode.
	def vc_filter(self):
		cdef Py_ssize_t i
		for i in range(len(self.muDist)):
			self.muDist[i] = np.sqrt((self.elec-self.y[i])**2+self.x[i]**2)
			self.ampV[i] = self.amp[i]*np.exp(-self.muDist[i]/self.ampk)
			self.lmV[i] = self.lm[i]*(1+self.durak*self.muDist[i])

	# FUNCTION NAME: hr1_f
	# FUNCTION DESCRIPTION: hermite-rodriguez 1nd order function (Cisi e Kohn, 2008)
	# INPUT PARAMS:  1) t: time simulation array (ms) [numpy array]
	#                2) lm: HR1 function duration factor (ms) [float]
	#                3) Am: HR1 function amplitude factor (mV) [float]
	#                4) tspk: Motorneuron discharge times (ms) [List of floats]
	# OUTPUT PARAMS: 1) hr1: biphasic motor unit action potential train over time 
	#                   (mV) [numpy array]
	cdef double[::1] hr1_f( self,
							double[::1] 	t,
							Py_ssize_t 		i,
							double[:] 		tspk):
		cdef Py_ssize_t n, j, w
		cdef double sbase, t1,t2
		cdef double[::1] hr1
		cdef bint cond1,cond2
		n = len(t)
		hr1= np.zeros(n, dtype=np.double)
		j=0
		sbase = self.lmV[i]*3
		for w in range(n):
			t1 = self.ampV[i]*((t[w]-tspk[j])/self.lmV[i])
			t2 = np.exp(-1*(((t[w]-tspk[j])/self.lm[i])**2))
			hr1[w] = t1*t2
			cond1 = t[w] > tspk[j] + sbase
			cond2 = j < len(tspk)-1
			if cond1 and cond2:
				j = j+1
		return hr1

	# FUNCTION NAME: hr2_f
	# FUNCTION DESCRIPTION: hermite-rodriguez 2nd order function (Cisi e Kohn, 2008)
	# INPUT PARAMS:  1) t: time simulation array (ms) [numpy array]
	#                2) lm: HR1 function duration factor (ms) [float]
	#                3) Am: HR1 function amplitude factor (mV) [float]
	#                4) tspk: Motorneuron discharge times (ms) [List of floats]
	# OUTPUT PARAMS: 1) hr2: triphasic motor unit action potential train 
	#                   over time (mV) [numpy array]
	cdef double[::1] hr2_f(	self,
							double[:] t,
							Py_ssize_t i,
							double[:] tspk):
		cdef Py_ssize_t n, j, w
		cdef double sbase, t1,t2
		cdef double[::1] hr2
		cdef bint cond1,cond2
		n = len(t)
		hr2= np.zeros(n,dtype=np.double)
		j=0
		sbase = self.lmV[i]*3
		for w in range(n):
			cond1 = t[w]>tspk[j]+sbase
			cond2 = j<len(tspk)-1
			if cond1 and cond2:
				j = j+1
			t1 = self.ampV[i]*(1-2*((t[w]-tspk[j])/self.lmV[i])**2)
			t2 = np.exp(-1*(((t[w]-tspk[j])/self.lm[i])**2))
			hr2[w] =  t1*t2
		return hr2

	def sEMG(self, double[::1] idvec,double[::1] spkvec):
		cdef np.ndarray[double, ndim=1] noise
		self.LR = int(max(idvec)) - int(min(idvec))
		self.idvec = np.array(idvec,dtype=np.intc) - int(min(idvec))
		self.spkvec = spkvec
		self.genRawEMG()
		noise = np.random.normal(0, self.noise, len(self.rawEMG))
		self.emg = self.rawEMG+noise

		## BUTTERWORTH BANDPASS FILTER
		# Sample rate and desired cutoff frequencies (in Hz).
		if self.filt:
			self.butter_bandpass_filter()
		return self.emg      

	# FUNCTION NAME: butter_bandpass_filter
	# FUNCTION DESCRIPTION: Apply Butterworth bandpass filter on data 
	# INPUT PARAMS:  1) data: temporal data in which the filter will be applied [np array]
	#                2) lowcut: Butterworth bandpass low cut frequency (Hz) [float]
	#                3) highcut: Butterworth bandpass high cut frequency (Hz) [float]
	#                4) fs: Sampling frequency (hz) [float]
	#                5) order: Butterworth filter oder [int]
	# OUTPUT PARAMS: 1) y: Filtered data [np array]
	cdef void butter_bandpass_filter(self,int order=4):
		cdef double nyq, low, high
		cdef object sos
		nyq = 0.5 * self.fs
		low = self.lc / nyq
		high = self.hc / nyq
		sos = butter(order, (low, high),btype='band', output='sos')
		self.emg = sosfilt(sos, self.emg)

    # FUNCTION NAME: semg
    # FUNCTION DESCRIPTION: Generates the surface EMG signal
    # INPUT PARAMS:  1) apvar: HR function amplitude factors attenuated by the
    #                          volume conductor [np array]
    #                2) lmV: HR function duration factors widened by the
    #                          volume conductor filtering effect [np array]
    #                3) t: simulation time array
    #                4) lastrec: Last recruited motor unit for each time step array
    #                5) rasterplot: All motorneurons discharge times. 
    # OUTPUT PARAMS: 1) emg: surface (or interference) emg signal
    #                2) mu_emg: motor unit action potential train of each motor unit
	cdef void genRawEMG(self) except *:
		cdef Py_ssize_t i
		cdef np.ndarray[double, ndim=1] spktrain
		cdef np.ndarray[int, 	ndim=1] index
		cdef double[::1] temp
		temp = np.zeros(len(self.t),dtype=np.double)
		self.rawEMG = np.zeros(len(self.t))
		self.mu_emg = np.zeros((self.LR+1,len(self.t)))
		for i in range (self.LR+1):
			index = np.asarray(np.asarray(
				np.array(self.idvec)==i,dtype=np.double).nonzero()[0],
				dtype=np.intc)
			spktrain = np.array(self.spkvec)[index]
			spktrain = spktrain+self.delay[i]
			if len(spktrain) > 0 :
				if np.random.randint(2) == 1: #
					temp = self.hr2_f(self.t,i,spktrain)
				else:
					temp = self.hr1_f(self.t,i,spktrain)
				self.mu_emg[i] = temp
				self.rawEMG = np.array(self.rawEMG) + np.array(temp)

	# FUNCTION NAME: MedFreq
	# FUNCTION DESCRIPTION: Calculates the median frequency and the power of the 
	#                       median frequency of a Power spectral density data
	# INPUT PARAMS:  1) freq: frequency vector generated with the PSD
	#                2) psd: Power spectrum density
	# OUTPUT PARAMS: 1) f: Median frequency of the PSD
	#                2) mfpsdvalue: Power of the median frequency of the PSD
	def MedFreq(self,freq,psd):
		cdef double[:] cum
		cdef double f
		cdef double mfp
		cum = cumulative_trapezoid(psd,freq,initial = 0)
		f = np.interp(cum[-1]/2,cum,freq)
		mfpsdvalue = cum[-1]/2
		return f,mfpsdvalue

	def movingAverage(self,double[:] emg,double  window):
		cdef int rms_length
		rms_length = int(window * self.fs/1e3)
		return np.convolve(
			abs(np.array(emg)), 
			np.ones((rms_length,),dtype=np.double) / rms_length, 
			mode = 'same')

	def spectrogram(self,emg,window,window_type,overlap):
		cdef int spec_w_size
		cdef int spec_ol
		cdef double[:] f,tf
		cdef double[:,:] Sxx
		spec_w_size = int(window*self.fs / 1e3)
		spec_ol = int(overlap*self.fs / 1e3)
		spec_window = get_window(window_type, spec_w_size)
		return spectrogram(	emg, 
							fs = self.fs, 
							window = spec_window, 
							nperseg = spec_w_size, 
							noverlap = spec_ol)


	def welch(self,emg,window,window_type,overlap):
		cdef int welch_w_size
		cdef int welch_ol
		cdef double[:] fwelch,PSDwelch
		cdef double emgfm,psdfm
		welch_w_size = int(window * self.fs / 1e3)
		welch_ol = int(overlap * self.fs / 1e3)
		fwelch, PSDwelch = welch(emg * 1e-3, self.fs, window =window, nperseg = welch_w_size, noverlap = welch_ol)
		emgfm, psdfm = self.MedFreq(fwelch, PSDwelch)
		return fwelch,PSDwelch,emgfm,psdfm

	def plot_rms(self,at,aemg,moving_average,title,ylabel):
		plt.ylabel(ylabel)
		plt.xlabel('Time [ms]')
		plt.title(title)
		plt.plot(at,aemg,lw=0.5,label='Raw EMG')
		plt.plot(at,moving_average,label='Moving RMS',lw=2,color='red')
		plt.annotate("EMG RMS = %.3f mV" %(np.sqrt(np.mean(np.square(aemg)))), xy=(0.1,0.90), xycoords = ("axes fraction"))
		plt.legend()
		plt.xlim(at[0],at[-1])

	def plot_spec(self,tf,f,Sxx,spec_axis,ylim):
		cf=plt.contourf(tf,f,Sxx*10**12, levels = 20, cmap=plt.cm.jet)
		plt.title('Spectrogram')
		plt.ylabel("Frequency [Hz]")
		plt.xlabel("Time [ms]")
		plt.ylim(0,ylim)
		ax_in = inset_axes(
			spec_axis, 
			width="5%",
			height="100%", 
			loc=3, 
			bbox_to_anchor=(1.01, 0., 1, 1),
			bbox_transform=spec_axis.transAxes, 
			borderpad=0)
		plt.colorbar(cf,cax = ax_in)

	def plot_welch(self,fwelch,PSDwelch,emgfm,psdfm,xlim,ylabel):
		plt.title("sEMG Power Spectrum Density")
		plt.plot(fwelch,PSDwelch*10**6)
		plt.axvline(x=emgfm,ls = '--',lw=0.5, c = 'k')
		plt.annotate(
			"Median Freq  = %.3fHz" %emgfm,
			xy=(0.6,0.95), 
			xycoords = ("axes fraction"))
		plt.ylabel(ylabel)
		plt.xlabel("Frequency [Hz]")
		plt.xlim(0,xlim)
		plt.ylim(bottom=0)

	def plot_mu_cont(self,at,mu_emg,mu_c_index):
		plt.plot(at,mu_emg)
		plt.xlabel('Time [ms]')
		plt.ylabel('Amplitude [mV]')
		plt.title('Motor unit # {} surface EMG contribution '.format(mu_c_index))
		#    print('Couldn\'t save EMG model parameters, try to run the simulation first.')
		return self.config

	def view_morpho(self):
		plt.figure(figsize=(6,6))
		plt.plot(
			np.array(self.ma) * 1000, 
			np.array(self.mb) * 1000,
			ls = '-.', label = 'muscle boundaries')
		plt.plot(
			np.array(self.fa) * 1000, 
			np.array(self.fb) * 1000, 
			ls = '--', label = 'adipose tissue')
		plt.plot(
			np.array(self.sa)*1000, 
			np.array(self.sb)*1000, 
			label = 'skin boundaries')
		plt.plot(
			self.elec*1000, 
			marker = 7, 
			ms = '15', 
			label = 'electrode')
		plt.legend()
		plt.axis('equal')
		plt.xlabel('[mm]')
		plt.ylabel('[mm]')
		plt.title('Muscle Cross-Sectional Area')

	def view_distribution(self):
		if self.LR == None:
			self.LR = 0 
		hst_step = 0.1
		hist_bins = [np.arange(min(self.ma) * 1e3, max(self.ma) * 1e3, hst_step), 
		             np.arange(min(self.mb) * 1e3, max(self.mb) *1e3, hst_step)]
		hist,xedges,yedges = np.histogram2d(
			np.array(self.gx) * 1e3, 
			np.array(self.gy) * 1e3, 
			bins = hist_bins)
		f, axes = plt.subplots(1, 2, figsize=(14, 7), sharex = 'all')
		plt.subplots_adjust(wspace = 0.1, hspace = 0.05)
		plt.sca(axes[0])
		plt.ylabel('[mm]')
		plt.xlabel('[mm]')
		plt.title('Motor unit territory distribution')
		##incluir aqui
		plt.annotate(
			"MU type II radial ecc:{:.2f} [mm]".format(self.ecc*1e3),
			xy=(0.1,0.1),
			xycoords = ("axes fraction")
		)

		fill_blue = mpl.patches.Patch( label = 'Recruited Type I MUs',  fc = (0, 0, 1, 0.4) )
		fill_red = mpl.patches.Patch( label = 'Recruited Type II MUs', fc = (1, 0, 0, 0.4))
		blue_line = mpl.lines.Line2D([], [], color= 'b', label = 'Type I MU')
		red_line = mpl.lines.Line2D([], [], color=  'r', ls = '--', label = 'Type II MU')
		plt.legend(handles = [fill_blue, fill_red, blue_line, red_line])
		for i in range(self.t1):
			if (i <= self.LR):
				plt.fill(
					np.array(self.MUT[i, 0]) * 1e3,
					np.array(self.MUT[i, 1]) * 1e3, 
					fc = (0, 0, 1, 0.4), 
					lw = 0.5)
			plt.plot(
				np.array(self.MUT[i, 0]) * 1e3, 
				np.array(self.MUT[i, 1]) * 1e3,
				color = 'b')
		for i in range(self.t1, self.n):
			if (i <= self.LR):
				plt.fill(
					np.array(self.MUT[i, 0]) * 1e3, 
					np.array(self.MUT[i, 1]) * 1e3, 
					fc = (1, 0, 0, 0.4))
			plt.plot(
				np.array(self.MUT[i, 0]) * 1e3, 
				np.array(self.MUT[i, 1]) * 1e3, 
				color = 'r', ls = '--')
		plt.plot(
			np.array(self.ma) * 1e3, 
			np.array(self.mb) * 1e3, 
			np.array(self.fa) * 1e3, 
			np.array(self.fb) * 1e3, 
			np.array(self.sa) * 1e3, 
			np.array(self.sb) * 1e3)
		plt.plot(self.elec * 1e3, marker = 7, ms = '15')
		plt.axis('equal')
		plt.sca(axes[1])
		axes1 = plt.gca()
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
		with plt.style.context('ggplot'):
			im1 = plt.imshow(hist.T, extent = extent, interpolation = 'nearest', origin = 'lower', cmap=plt.cm.jet)
			axins1 = inset_axes(axes1, width = "5%", height = "100%", loc = 3, bbox_to_anchor=(1.01, 0., 1, 1),
			                    bbox_transform = axes1.transAxes, borderpad = 0)
			plt.colorbar(im1, cax = axins1)
			axins1.xaxis.set_ticks_position("bottom")
			axes1.axis('equal')
			axes1.set_xlabel('[mm]')
			axes1.set_title('Motor unit territories (2D Histogram)')

	def view_attenuation(self):
		step = 1e-4
		ga = np.arange(min(self.ma), max(self.ma), step)
		gb = np.arange(min(self.mb), max(self.mb), step)
		Ga, Gb = np.meshgrid(ga, gb)
		muDist2d = np.sqrt((self.elec-Gb) ** 2 + Ga ** 2)
		apvar2d = np.exp(-muDist2d / self.ampk)
		lmvar2d = 1 + self.durak * muDist2d
		f = plt.figure(figsize = (15, 7))
		f.subplots_adjust(wspace=0.05)
		axes1 = plt.subplot(121)
		axes1.axis('equal')
		plt.xlabel('[mm]')
		plt.ylabel('[mm]')
		plt.title('MUAP Amplitude Attenuation')
		plt.plot(
			np.array(self.ma) * 1e3, 
			np.array(self.mb) * 1e3, 
			np.array(self.fa) * 1e3, 
			np.array(self.fb) * 1e3, 
			np.array(self.sa) * 1e3, 
			np.array(self.sb) *1e3)
		plt.plot(self.elec * 1e3, marker = 7, ms = '15')
		CS = plt.contour(Ga * 1e3, Gb * 1e3, apvar2d, 10, cmap=plt.cm.jet_r)
		ax1= plt.gca()
		plt.colorbar(CS, ax = ax1)
		axes1.clabel(CS, inline = 1, fontsize = 12)
		ax2 = plt.subplot(122)
		plt.plot(
			np.array(self.ma) * 1e3, 
			np.array(self.mb) * 1e3, 
			np.array(self.fa) * 1e3, 
			np.array(self.fb) * 1e3, 
			np.array(self.sa) * 1e3, 
			np.array(self.sb) * 1e3)
		plt.plot(self.elec * 1e3, marker = 7, ms = '15')
		CS2 = plt.contour(Ga * 1e3, Gb * 1e3, lmvar2d, 8, cmap=plt.cm.jet)
		ax2 = plt.gca()
		plt.colorbar(CS2, ax = ax2)
		ax2.clabel(CS2, inline = 1, fontsize = 12)
		ax2.axis('equal')
		plt.xlabel('[mm]')
		plt.title('MUAP Widening')