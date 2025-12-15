: Motoneuron model implementing Hodgkin-Huxley type dynamics
: This model includes:
: - Sodium (Na+) channels with activation (m) and inactivation (h) gates
: - Fast potassium (K+) channels with activation gate (n)
: - Slow potassium (K+) channels with activation gate (p)
: - Leak current
: The model uses standard Hodgkin-Huxley formulations for ion channels
: with voltage-dependent rate constants and conductances.
:
: Documentation:
: - USEION syntax: https://nrn.readthedocs.io/en/8.2.6/guide/faq.html#what-units-does-neuron-use-for-current-concentration-etc
: - Units: https://nrn.readthedocs.io/en/8.2.6/guide/units.html
: - State variables: https://nrn.readthedocs.io/en/8.2.6/guide/faq.html#how-do-i-create-a-neuron-model
: - Rate constants: https://nrn.readthedocs.io/en/8.2.6/guide/faq.html#is-there-a-list-of-functions-that-are-built-into-nmodl

NEURON {
    SUFFIX motoneuron
    : Sodium ion channel
    : READ ena: reads the sodium reversal potential from the extracellular space
    : WRITE ina: writes the computed sodium current back to the extracellular space
    USEION na READ ena WRITE ina
    : Slow potassium ion channel
    : READ eks: reads the slow potassium reversal potential from the extracellular space
    : WRITE iks: writes the computed slow potassium current back to the extracellular space
    USEION ks READ eks WRITE iks
    : Fast potassium ion channel
    : READ ekf: reads the fast potassium reversal potential from the extracellular space
    : WRITE ikf: writes the computed fast potassium current back to the extracellular space
    USEION kf READ ekf WRITE ikf
    : Leak current
    NONSPECIFIC_CURRENT il
    : Channel conductances and parameters
    RANGE gna, gk_fast, gk_slow, gl, vt, el
    : Rate constants
    GLOBAL alpha_m, alpha_h, alpha_n, pinf, beta_m, beta_h, beta_n, ptau
    : Allows parallel execution
    THREADSAFE
}

UNITS {
    : Current unit
    (mA) = (milliamp)
    : Voltage unit
    (mV) = (millivolt)
    : Conductance unit
    (S) = (siemens)
}

PARAMETER {
    : Leak conductance
    gl = 0.0003 (nS/um2) <0,1e9>
    : Sodium channel conductance
    gna = 0.0003 (nS/um2) <0,1e9>
    : Fast potassium channel conductance
    gk_fast = 0.0003 (nS/um2) <0,1e9>
    : Slow potassium channel conductance
    gk_slow = 0.0003 (nS/um2) <0,1e9>
    : Leak reversal potential
    el = -70 (mV)
    : Maximum time constant for slow K+ channel
    tau_max_p = 4
    : Voltage threshold for activation
    vt = -58
}

ASSIGNED {
    : Membrane potential
    v (mV)
    : Sodium reversal potential
    ena (mV)
    : Slow potassium reversal potential
    eks (mV)
    : Fast potassium reversal potential
    ekf (mV)
    : Sodium current
    ina (mA/cm2)
    : Slow potassium current
    iks (mA/cm2)
    : Fast potassium current
    ikf (mA/cm2)
    : Leak current
    il (mA/cm2)
}

STATE {
    : m,h: Na+ channel gates, n: fast K+ gate, p: slow K+ gate
    m h n p
}

INITIAL {
    rates(v)
    : Na+ activation gate starts closed
    m = 0
    : Na+ inactivation gate starts open
    h = 1
    : Fast K+ gate starts closed
    n = 0
    : Slow K+ gate starts at steady state
    p = pinf
}

? currents
BREAKPOINT {
    : Solve differential equations
    SOLVE states METHOD cnexp
    : Sodium current (m^3*h formulation)
    ina = gna*m*m*m*h*(v - ena)
    : Fast potassium current (n^4 formulation)
    ikf = gk_fast*n*n*n*n*(v - ekf)
    : Slow potassium current (p^2 formulation)
    iks = gk_slow*p*p*(v - eks)
    : Leak current
    il = gl*(v - el)
    :printf("ina = %g", ina)
}

DERIVATIVE states {
    rates(v)
    : Na+ activation gate
    m' = alpha_m*(1-m) - beta_m*m
    : Na+ inactivation gate
    h' = 0.1*alpha_h*(1-h) - 0.1*beta_h*h
    : Fast K+ gate
    n' = 0.1*alpha_n*(1-n) - 0.1*beta_n*n
    : Slow K+ gate
    p' = (pinf - p) / ptau
}

PROCEDURE rates(v(mV)) {
    : Na+ activation forward rate
    alpha_m = (-0.32*(v-vt-13))/(exp(-(v-vt-13)/4)-1)
    : Na+ activation backward rate
    beta_m = 0.28*(v-vt-40)/(exp((v-vt-40)/5)-1)
    : Na+ inactivation forward rate
    alpha_h = 0.128*exp(-(v-vt-17)/18)
    : Na+ inactivation backward rate
    beta_h = 4/(1+exp(-(v-vt-40)/5))

    : Fast K+ activation forward rate
    alpha_n = (-0.032*(v-vt-15))/(exp(-(v-vt-15)/5)-1)
    : Fast K+ activation backward rate
    beta_n = 0.5*exp(-(v-vt-10)/40)

    : Slow K+ steady-state activation
    pinf = 1/(1+exp(-(v+35)/10))
    : Slow K+ time constant
    ptau = tau_max_p/(3.3*exp((v+35)/20)+exp(-(v+35)/20))
}