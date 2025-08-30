TITLE napp.mod   squid sodium, potassium, and leak channels

COMMENT
 This is the original Hodgkin-Huxley treatment for the set of sodium,
  potassium, and leakage channels found in the squid giant axon membrane.
  ("A quantitative description of membrane current and its application
  conduction and excitation in nerve" J.Physiol. (Lond.) 117:500-544 (1952).)
 Membrane voltage is in absolute mV and has been reversed in polarity
  from the original HH convention and shifted to reflect a resting potential
  of -65 mV.
 Remember to set celsius=6.3 (or whatever) in your HOC file.
 See squid.hoc for an example of a simulation using this model.
 SW Jaslove  6 March, 1992
ENDCOMMENT

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}

? interface
NEURON {
    SUFFIX napp
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    NONSPECIFIC_CURRENT il
    RANGE gnabar,gnapbar, gkfbar, gksbar, gl, el, gna, gnap, gkf, gks, vtraub, mact, rinact, inap, inaf, ikf, iks, ek, ena
    GLOBAL minf, hinf, pinf, ninf, rinf, mtau, htau, ptau, ntau, rtau
    THREADSAFE : assigned GLOBALs will be per thread
}

PARAMETER {
    gnabar = .030 (S/cm2)	<0,1e9>
    gnapbar = .000033 (S/cm2) <0,1e9> 
    gkfbar = .016 (S/cm2)	<0,1e9>
    gksbar = .004 (S/cm2)	<0,1e9>
    
    gl = .0003 (S/cm2)	<0,1e9>
    el = -54.3 (mV)
    
    vtraub = 50.0 (mV)
    mact = 15.0 (mV)
    rinact = 0.05 (/ms)
	
  	
}

STATE {
    m h p n r 
}

ASSIGNED {
    v (mV)
    celsius (degC)
    
    gna (S/cm2)
    gnap (S/cm2)
    gkf (S/cm2)
    gks (S/cm2)
    ena (mV)
    ek (mV)
    ina (mA/cm2)
    inap (mA/cm2)
    inaf (mA/cm2)
    ik (mA/cm2)
    ikf (mA/cm2)
    iks (mA/cm2)
    il (mA/cm2)
    minf hinf pinf ninf rinf
    mtau (ms) htau (ms) ptau (ms) ntau (ms) rtau (ms)
}

? currents
BREAKPOINT {
    SOLVE states METHOD cnexp
    gna = gnabar*m*m*m*h
    gnap = gnapbar*p*p*p
    inaf = gna*(v-ena)
    inap = gnap*(v-ena)
    ina = inaf+inap
    gkf = gkfbar*n*n*n*n
    gks = gksbar*r*r
    ikf = (gkf)*(v-ek)
    iks = (gks)*(v-ek)
    ik = ikf+iks
    il = gl*(v - el)
}


INITIAL {
    rates(v)
    m = minf
    h = hinf
    p = pinf
    n = ninf
    r = rinf
}

? states
DERIVATIVE states {
    rates(v)
    m' =  (minf-m)/mtau
    h' = (hinf-h)/htau
    p' = (pinf-p)/ptau
    n' = (ninf-n)/ntau
    r' = (rinf-r)/rtau
}

? rates
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
    LOCAL  alpha, beta, sum, v2
    TABLE minf, mtau, hinf, htau, pinf, ptau, ninf, ntau, rinf, rtau DEPEND celsius FROM -100 TO 100 WITH 200

UNITSOFF

    v2 = v - vtraub

    :"m" sodium activation system
    alpha = .64 * vtrap(mact-v2,4)
    beta = .56 * vtrap(v2-40,5)
    sum = alpha + beta
    mtau = 1/sum
    minf = alpha/sum
    :"h" sodium inactivation system
    alpha = .928 * exp((17-v2)/18)
    beta = 9 / (exp((40-v2)/5) + 1)
    sum = alpha + beta
    htau = 1/sum
    hinf = alpha/sum
    :"p" sodium persistent activation system
    alpha = .64 * vtrap(5-v2,4)
    beta = .56 * vtrap(v2-30,5)
    sum = alpha + beta
    ptau = 1/sum
    pinf = alpha/sum
    :"n" fast potassium activation system
    alpha = .08*vtrap(15-v2,7)
    beta = 2*exp((10-v2)/40)
    sum = alpha + beta
    ntau = 1/sum
    ninf = alpha/sum
    :"r" slow potassium activation system
    alpha = (3.5)/(exp((55-v2)/4) + 1)
    beta = rinact
    sum = alpha + beta
    rtau = 1/sum
    rinf = alpha/sum
}

FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
        }else{
            vtrap = x/(exp(x/y) - 1)
        }
}

UNITSON
