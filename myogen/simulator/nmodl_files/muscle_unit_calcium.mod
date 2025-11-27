NEURON {
    POINT_PROCESS muscle_unit_calcium
    RANGE Tc, Fmax, spike, F, A, temp
	RANGE k1, k2, k3, k4, k5, k6, k, k5i, k6i
	RANGE Umax, Rmax, tau1, tau2, R, R1, R2
	RANGE phi0, phi1, phi2, phi3, phi4
	RANGE AMinf, AMtau, SF_AM, T0
	RANGE c1i, c1n1, c1n2, c1n3, tauc1, c2i, c2n1, c2n2, c2n3, tauc2, c3, c4, c5, c1inf, c2inf
	RANGE acm, alpha :alpha1, alpha2, alpha3, beta, gamma
}

PARAMETER{
    Tc = 100
    Fmax = 1
    :: Calcium dynamics ::
	k1 = 3000		: M-1*ms-1
	k2 = 3			: ms-1
	k3 = 400		: M-1*ms-1
	k4 = 1			: ms-1
	k5i = 4e5		: M-1*ms-1
	k6i = 150		: ms-1
	k = 850			: M-1	
	Rmax = 10		: ms-1
	Umax = 2000		: M-1*ms-1
	tau1 = 1			: ms
	tau2 = 13			: ms
	phi1 = 0.004
	phi2 = 0.98
	phi3 = 0.0002
	phi4 = 0.999
	SF_AM = 5
	CS0 = 0.03     	:[M]
	B0 = 0.00043	:[M]
	T0 = 0.00007 	:[M]
    R = 0
	R1 = 0
	R2 = 0 

	:: Muscle activation::
	c1i = 0.154
	c1n1 = 0.01
	c1n2 = 0.15
	c1n3 = 0.01
	tauc1 = 85
	c2i = 0.11
	c2n1 = -0.0315
	c2n2 = 0.27
	c2n3 = 0.015
	tauc2 = 70
	c3 = 54.717
	c4 = -18.847
	c5 = 3.905
	alpha = 2
	:alpha1 = 4.77
	:alpha2 = 400
	:alpha3 = 160
	:beta = 0.47
	:gamma = 0.001
	temp = 0
}

ASSIGNED{
    spike
    F
	A
	k5
	k6
	AMinf
	AMtau
	c1inf
	c2inf
	:xm_temp1
	:xm_temp2
	:vm
	:acm
}

STATE{
    CaSR CaT AM x1 x2 xm CaSRCS Ca CaB c1 c2
}

INITIAL{
    x1 = 0.0
    x2 = 0.0
    spike = 0
	CaSR = 0.0025  		:[M]
	CaSRCS = 0.0		    :[M]
	Ca = 1e-10		    :[M]
	CaT = 0.0				:[M]
	AM = 0.0				:[M]
	CaB = 0.0				:[M]
	c1 = 0.154
	c2 = 0.11
}


BREAKPOINT{
	R1 = R1*exp(-dt/tau2)
	R2 = R2*exp(-dt/tau2)*exp(-dt/tau1)
	R = CaSR*Rmax*(R1 - R2)
	:printf("R = %g", R)
	rate (CaT, AM, t)
    SOLVE states_force METHOD derivimplicit
    F = Fmax*x1
	spike = 0
	A = AM^alpha
	:printf("CaSR = %g", CaSR)
}

DERIVATIVE states_force{
    x1' = x2
    x2' = -2/Tc*x2 - 1/(Tc*Tc)*x1 + CaT/0.0001/Tc    	
	CaSR' = - R + U(Ca) -k1*CS0*CaSR + (k1*CaSR+k2)*CaSRCS 
	CaSRCS' = k1*CS0*CaSR - (k1*CaSR+k2)*CaSRCS
	Ca' = -k5*T0*Ca + (k5*Ca+k6)*CaT + R - U(Ca) - k3*B0*Ca +(k3*Ca+k4)*CaB 
	CaB' = k3*B0*Ca -(k3*Ca+k4)*CaB
	CaT' = k5*T0*Ca - (k5*Ca+k6)*CaT
	AM' = 0 :(AMinf -AM)/AMtau
	c1' = 0 :(c1inf - c1)/tauc1
	c2' = 0 :(c2inf - c2)/tauc2
	
}

PROCEDURE rate(CaT (M), AM (M), t(ms)) {
	k5 = phi(5)*k5i
	k6 = k6i/(1 + SF_AM*AM)
	AMinf = 0.5*(1+tanh((CaT/T0-c1)/c2))
	AMtau = c3/(cosh((CaT/T0-c4)/(2*c5)))
	c1inf = c1n1*(1+tanh((CaT/T0-c1n2)/c1n3))+c1i
	c2inf = c2n1*(1+tanh((CaT/T0-c2n2)/c2n3))+c2i
}

FUNCTION U(x) {
	if (x >= 0) {U = Umax*(x^2*k^2/(1+x*k+x^2*k^2))^2}
	else {U = 0}
}

FUNCTION phi(x) {
	if (x <= 5) {phi = phi1*x + phi2}
	else {phi = phi3*x + phi4}
}

NET_RECEIVE (weight) {
	spike = 2.7182818/dt
    R1 = R1 + 1
 	R2 = R2 + 1
	:temp = (AMinf -AM)/AMtau
	:printf("%g ", temp)
	:printf("R1 = %g, R2 = %g, R = %g, CaSR = %g", R1, R2, R, CaSR)
}

