
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
    RANGE m_alpha_A, m_alpha_v_offset, m_alpha_k, m_beta_A, m_beta_v_offset, m_beta_k
    RANGE h_alpha_A, h_alpha_v_offset, h_alpha_tau, h_beta_A, h_beta_v_offset, h_beta_k
    RANGE p_alpha_A, p_alpha_v_offset, p_alpha_k, p_beta_A, p_beta_v_offset, p_beta_k
    RANGE n_alpha_A, n_alpha_v_offset, n_alpha_k, n_beta_A, n_beta_v_offset, n_beta_tau
    RANGE r_alpha_A, r_alpha_v_offset, r_alpha_k
    GLOBAL minf, hinf, pinf, ninf, rinf, mtau, htau, ptau, ntau, rtau
    THREADSAFE : assigned GLOBALs will be per thread
}

PARAMETER {
    gnabar =            .030 (S/cm2) <0,1e9>
    gnapbar =           .000033 (S/cm2) <0,1e9>
    gkfbar =            .016 (S/cm2)	<0,1e9>
    gksbar =            .004 (S/cm2)	<0,1e9>

    gl =                .0003 (S/cm2)	<0,1e9>
    el =                -54.3 (mV)
    vtraub =            50.0 (mV)
    mact =              15.0 (mV)
    rinact =            0.05 (/ms)

    : Alpha and beta parameters for m gate (sodium activation)
    m_alpha_A =         0.64
    m_alpha_v_offset =  15.0 (mV)
    m_alpha_k =         4.0 (mV)
    m_beta_A =          0.56
    m_beta_v_offset =   40.0 (mV)
    m_beta_k =          5.0 (mV)

    : Alpha and beta parameters for h gate (sodium inactivation)
    h_alpha_A =         0.928
    h_alpha_v_offset =  17.0 (mV)
    h_alpha_tau =       18.0 (mV)
    h_beta_A =          9.0
    h_beta_v_offset =   40.0 (mV)
    h_beta_k =          5.0 (mV)

    : Alpha and beta parameters for p gate (persistent sodium activation)
    p_alpha_A =         0.64
    p_alpha_v_offset =  5.0 (mV)
    p_alpha_k =         4.0 (mV)
    p_beta_A =          0.56
    p_beta_v_offset =   30.0 (mV)
    p_beta_k =          5.0 (mV)
    : Alpha and beta parameters for n gate (fast potassium activation)
    n_alpha_A =         0.08
    n_alpha_v_offset =  15.0 (mV)
    n_alpha_k =         7.0 (mV)
    n_beta_A =          2.0
    n_beta_v_offset =   10.0 (mV)
    n_beta_tau =        40.0 (mV)

    : Alpha and beta parameters for r gate (slow potassium activation)
    r_alpha_A =         3.5
    r_alpha_v_offset =  55.0 (mV)
    r_alpha_k =         4.0 (mV)

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
PROCEDURE rates(v(mV)) {  
    :Computes rate and other constants at current v.
    :Call once from HOC to initialize inf at resting v.
    LOCAL alpha, beta, sum, v2

UNITSOFF

    v2 = v - vtraub

    :"m" sodium activation system
    alpha = m_alpha_A * vtrap(m_alpha_v_offset-v2, m_alpha_k)
    beta = m_beta_A * vtrap(v2-m_beta_v_offset, m_beta_k)
    sum = alpha + beta
    mtau = 1/sum
    minf = alpha/sum
    :"h" sodium inactivation system
    alpha = h_alpha_A * exp((h_alpha_v_offset-v2)/h_alpha_tau)
    beta = h_beta_A / (exp((h_beta_v_offset-v2)/h_beta_k) + 1)
    sum = alpha + beta
    htau = 1/sum
    hinf = alpha/sum
    :"p" sodium persistent activation system
    alpha = p_alpha_A * vtrap(p_alpha_v_offset-v2, p_alpha_k)
    beta = p_beta_A * vtrap(v2-p_beta_v_offset, p_beta_k)
    sum = alpha + beta
    ptau = 1/sum
    pinf = alpha/sum
    :"n" fast potassium activation system
    alpha = n_alpha_A*vtrap(n_alpha_v_offset-v2, n_alpha_k)
    beta = n_beta_A*exp((n_beta_v_offset-v2)/n_beta_tau)
    sum = alpha + beta
    ntau = 1/sum
    ninf = alpha/sum
    :"r" slow potassium activation system
    alpha = (r_alpha_A)/(exp((r_alpha_v_offset-v2)/r_alpha_k) + 1)
    beta = rinact
    sum = alpha + beta
    rtau = 1/sum
    rinf = alpha/sum
}

FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
    } else {
        vtrap = x/(exp(x/y) - 1)
    }
}

UNITSON
