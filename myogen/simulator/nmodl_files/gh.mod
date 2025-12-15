TITLE gh channel channel
: Hodgkin - Huxley h channel


NEURON {
	SUFFIX gh
	NONSPECIFIC_CURRENT i
	RANGE i, eh, ghbar,half,slp
	GLOBAL htau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v 				(mV)
	ghbar =	.001	(mho/cm2) <0,1e9>
	htau = 	50 		(ms)
	half=	-80 	(mV)
	slp=	8 		(mV)
	eh = 	-41 	(mV)
}

STATE {n}

ASSIGNED {
	i (mA/cm2)
	inf
}

INITIAL {
	rate(v)
	n = inf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	i=ghbar*n*(v - eh)
}

DERIVATIVE states {	
	rate(v)
	n' = (inf - n)/htau
}

UNITSOFF
	PROCEDURE rate(v(mV)) {	
		TABLE inf DEPEND half,slp FROM -100 TO 100 WITH 200
		inf = 1/(1+exp((v-half)/slp))
	}
UNITSON
