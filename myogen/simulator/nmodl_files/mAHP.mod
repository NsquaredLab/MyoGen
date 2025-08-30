
 COMMENT
 
 mAHP.mod
 
 Calcium-dependent potassium channel responsible for mAHP in motoneurons
 Simplified calcium channel that provides Ca for the KCa conductance is included
 	
 ENDCOMMENT

 NEURON {
 	SUFFIX mAHP
 	USEION k READ ek WRITE ik
 	USEION ca READ eca WRITE ica
 	RANGE n, gkcamax,gcamax,ik,cai,ica,depth,taur
 	GLOBAL fKCa, bKCa, caix
 }

 
 UNITS {
 	(mA) = (milliamp)
 	(mV) = (millivolt)
 	(S) = (siemens)
 	(um) = (micron)
 	(molar) = (1/liter)			: moles do not appear in units
 	(mM)	= (millimolar)
 	(msM)	= (ms mM)
 	FARADAY = (faraday) (coulomb)
 } 
 
 PARAMETER {
 	gkcamax = 0.03   	(S/cm2)	
	gcamax = 3e-5		(S/cm2)
	mvhalfca = -30		(mV)
	mslpca = 4 		(mV)
	mtauca = 1		(ms)	
 	caix = 2	
  	cainf=0.0001		(mM)
 	depth	= .1		(um)		: depth of shell
 	taur	= 20		(ms)		: rate of calcium removal
								
  	fKCa   = 0.1		: max act rate  
 	bKCa   = 0.1		: max deact rate 
 
 	celsius		(degC)
 } 
 
 
 ASSIGNED {
 	ik 		(mA/cm2)
 	v 		(mV)
	ica 		(mA/cm2)
 	ek		(mV)
	eca		(mV)
 	ninf
 	ntau 		(ms)
	minfca	
	drive_channel
 }
  
 
 STATE {
 mca 
 n 
 cai (mM)
}
 
 INITIAL { 
	cai=cainf
 	rates(cai)
	mcarate(v)
 	n = ninf
	mca=minfca
 }
 
 BREAKPOINT {
         SOLVE states METHOD cnexp
	ica = gcamax*mca*(v - eca)
 	ik =  gkcamax *n* (v - ek)
 } 
 

DERIVATIVE states { 
	 
 	drive_channel =  - (10000) * ica/ (2 * FARADAY * depth)
 	if (drive_channel <= 0.) { drive_channel = 0. }	: cannot pump inward
 	cai' = drive_channel + (cainf-cai)/taur

         rates(cai)    
         n' = (ninf-n)/ntau
         mcarate(v)    
         mca' = (minfca-mca)/mtauca
}
PROCEDURE rates(cai(mM)) {  LOCAL a,b
							UNITSOFF
         a = fKCa * (1e3*(cai  -cainf))^caix		: rate constant depends on cai in uM
         b = bKCa
         ntau = 1/(a+b)
         ninf = a*ntau
					UNITSON
 }

PROCEDURE mcarate(v (mV)) {
	TABLE minfca
	DEPEND mvhalfca,mslpca 
	FROM -100 TO 100 WITH 200
	
	minfca = 1/(1+exp(-(v-mvhalfca)/mslpca))
}
