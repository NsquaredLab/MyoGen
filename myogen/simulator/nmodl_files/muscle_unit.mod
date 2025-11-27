NEURON {
    POINT_PROCESS muscle_unit
    RANGE Tc, Fmax, spike, F
}


PARAMETER {
    Tc =    100
    Fmax =  1
}

ASSIGNED {
    spike
    F
}

STATE {
    x1 x2
}

INITIAL {
    x1 = 0
    x2 = 0
    spike = 0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    F = Fmax*x1
}

DERIVATIVE states {
    x1' = x2
    x2' = -2/Tc*x2-1/(Tc*Tc)*x1+spike/Tc
    spike = 0
}

NET_RECEIVE (weight) {
	spike = 2.7182818/dt
}

