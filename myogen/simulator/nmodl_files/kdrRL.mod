TITLE Potassium Delayed Rectifier Channel
	:This channel is a Voltage Dependent Potassium Channel
	: and will create a current (ik) based on the voltage 
	:Simplied by RKP 3/22/07 to exlude references to different
	: parts of Bob's split dendrite model

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

NEURON {
	SUFFIX kdrRL
	USEION k READ ek WRITE ik
	RANGE ik, g, gMax
	GLOBAL  mVh,mslp, tVh, tslp, tmin,taumax
}

PARAMETER {
	gMax = 0.1 (S/cm2)
	mVh = -25 (mV)
	mslp = 20 (mV)
	tVh = -39 (mV)			
	tslp = 5.5 (mV)			
	tmin = 1.4 (ms)		
	taumax = 11.9(ms)
}			

ASSIGNED {
	v   (mV)
	ek  (mV)
	ik  (mA/cm2)
	g   (S/cm2)
	mtau (ms)
	minf
}

STATE {
	m
}

INITIAL {
	rate(v)
	m = minf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = gMax * m^4
	ik = g*(v - ek)
}

DERIVATIVE state {
	rate(v)
	m' = (minf - m)/mtau
}

PROCEDURE rate(v (mV)) {
	LOCAL b, f TABLE minf,mtau 
	DEPEND mVh,mslp,tVh,tslp,tmin,taumax 
	FROM -100 TO 100 WITH 200

	b = exp((v - tVh)/tslp)
	f = (1 + b)^2
	
	minf = 1/(1+exp(-(v-mVh)/mslp))
	mtau = tmin + taumax*b/f
}
