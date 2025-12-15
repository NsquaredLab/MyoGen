TITLE na3rp
: Na current 
: modified from Jeff Magee. M.Migliore may97
: added sh to account for higher threshold M.Migliore, Apr.2002
: modified by RP to have slow inactivation given in Fleiderivsh et al.

NEURON {
	SUFFIX na3rp
	USEION na READ ena WRITE ina
	RANGE  gbar, ar, sh,ina
	RANGE minf, hinf, mtau, htau, sinf, taus, qinf, thinf
}

PARAMETER {
	sh = 		8 (mV)
	gbar = 		0.010 (mho/cm2)	
								
	tha =  		-30	(mV)		: v 1/2 for act	
	qa = 		7.2	(mV)		: act slope (4.5)		
	Ra = 		0.4	(/ms)		: open (v)		
	Rb = 		0.124 (/ms)		: close (v)		

	thi1 = 		-45	(mV)		: v 1/2 for inact 	
	thi2 = 		-45 (mV)		: v 1/2 for inact 	
	qd = 		1.5	(mV)		: inact tau slope
	qg = 		1.5 (mV)
	mmin =		0.02	
	hmin =		0.5			
	q10 =		2
	Rg = 		0.01 (/ms)		: inact recov (v) 	
	Rd = 		.03 (/ms)		: inact (v)	
	qq = 		10 (mV)
	tq = 		-55 (mV)

	thinf = 	-50 (mV)		: inact inf slope	
	qinf = 		4 (mV)			: inact inf slope 

	a0s =		0.001 (/ms)	
	b0s =		0.0034 (/ms)
	asvh =		-85 (mV) 
	bsvh =		-17 (mV) 
	avs =		30 (mV)
	bvs =		10 (mV)
	ar =		1 (1)			: 1=no inact., 0=max inact.
	ena			(mV)            : must be explicitly def. in hoc
	celsius
	v 			(mV)
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ina 		(mA/cm2)
	thegna		(mho/cm2)
	minf 		hinf 		
	mtau (ms)	htau (ms) 	
	sinf (ms)	taus (ms)
}

STATE { m h s}

BREAKPOINT {
	SOLVE states METHOD cnexp
	thegna = gbar*m*m*m*h*s
	ina = thegna * (v - ena)
} 

INITIAL {
	trates(v,ar,sh)
	m=minf  
	h=hinf
	s=sinf
}
 
FUNCTION alps(v(mV)) {  
   alps = a0s*exp((asvh-v)/avs)
}

FUNCTION bets(v(mV)) {
  bets = b0s/(exp((bsvh-v)/bvs)+1)
}

LOCAL mexp, hexp, sexp

DERIVATIVE states {   
	trates(v,ar,sh)      
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
	s' = (sinf - s)/taus
}

PROCEDURE trates(vm,a2,sh2) {  
	LOCAL  a, b, c, qt
	qt=q10^((celsius-24)/10)
	a = trap0(vm,tha+sh2,Ra,qa)
	b = trap0(-vm,-tha-sh2,Rb,qa)
	mtau = 1/(a+b)/qt
	if (mtau<mmin) {mtau=mmin}
	minf = a/(a+b)

	a = trap0(vm,thi1+sh2,Rd,qd)
	b = trap0(-vm,-thi2-sh2,Rg,qg)
	htau =  1/(a+b)/qt
	if (htau<hmin) {htau=hmin}
	hinf = 1/(1+exp((vm-thinf-sh2)/qinf))
	taus = 1/(alps(vm)+bets(vm))
	c = alps(vm)*taus
	sinf = c+a2*(1-c)
}

FUNCTION trap0(v,th,a,q) {
	if (fabs(v-th) > 1e-6) {
		trap0 = a * (v - th) / (1 - exp(-(v - th)/q))
	} else {
		trap0 = a * q
 	}
}	
