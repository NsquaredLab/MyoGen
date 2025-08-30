TITLE caL.mod   

COMMENT
ENDCOMMENT

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}

? interface
NEURON {
        SUFFIX caL
	USEION caL READ ecaL WRITE icaL VALENCE 2
	NONSPECIFIC_CURRENT il
        RANGE  gl, el, gcaL, gcaLbar, vtraub, icaL, gama, Ltau, ecaL
        GLOBAL Linf
	THREADSAFE : assigned GLOBALs will be per thread
}

PARAMETER {
        gcaLbar = .030 (S/cm2)	<0,1e9>
	gl = .0003 (S/cm2)	<0,1e9>
        el = -54.3 (mV)

	vtraub = 70.0 (mV)
	gama = 1
	Ltau = 20 (ms)
}

STATE {
        L 
}

ASSIGNED {
        v (mV)
        celsius (degC)
        ecaL (mV)
	
    	gcaL (S/cm2)
        icaL (mA/cm2)
        il (mA/cm2)
        Linf
	
}

? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
        gcaL = gcaLbar*L
	icaL = gcaL*gama*(v-ecaL)
        il = gl*(v - el)
}


INITIAL {
	rates(v)
	L = Linf
	
}

? states
DERIVATIVE states {
        rates(v)
        L' =  (Linf-L)/Ltau
}

? rates
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL v2

UNITSOFF

	v2 = v - vtraub

                :"L" calcium activation system
   

        Linf = 1/(exp((v2+30)/(-1))+1)
}


UNITSON
