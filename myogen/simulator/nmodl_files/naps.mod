TITLE naps		:modified to have slow inactivation described in Fleidervish and to make slope a global parameter

NEURON {
	SUFFIX naps
	USEION na READ ena WRITE ina
	RANGE  gbar, thegna, sh, ar,minf,sinf,taus
	GLOBAL mtau,vslope
}

PARAMETER {
	gbar = .0052085   	(mho/cm2)
	sh = 0  (mV)
	vslope=6.8   (mV)	:activation slope
	mtau = 1 (ms)
	ena		(mV)       :must be explicitly defined in hoc     
        a0s=0.001	(/ms)	
        b0s=0.0034	(/ms)
        asvh=-85	(mV) 
        bsvh=-17	(mV) 
        avs=30		(mV)
        bvs=10		(mV)
        ar=1		(1)		: 1=no inact., 0=max inact.
	celsius (degC)
	v 		(mV)
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
	minf 	
	sinf
	taus (ms)
}
 

STATE { m s }

UNITSOFF

BREAKPOINT {
    SOLVE states METHOD cnexp
	        		
	thegna =gbar*m*s       
	ina = thegna * (v - ena)
	} 

INITIAL {
	trates(v,ar,sh)
	m=minf  
	s=sinf
}

DERIVATIVE states {   
    	trates(v,ar,sh)
        s' = (sinf - s)/taus
	m' = (minf-m)/mtau
}

PROCEDURE trates(vm,a2,sh2) {  
        LOCAL   c 

	minf = (1/(1+exp(-(vm+52.3-sh2)/vslope)))      	
        taus = 1/(alps(vm)+bets(vm))
	c=alps(vm)*taus
        sinf = c+a2*(1-c)
 }




FUNCTION alps(v(mV)) {  
  alps = a0s*exp((asvh-v)/avs)
}

FUNCTION bets(v(mV)) {
  bets = b0s/(exp((bsvh-v)/bvs)+1)
}

UNITSON
