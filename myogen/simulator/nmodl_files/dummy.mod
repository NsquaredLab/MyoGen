: $Id: nsloc.mod,v 1.7 2013/06/20  salvad $
: from nrn/src/nrnoc/netstim.mod
: modified to use as proprioceptive units in arm2dms model

NEURON	{ 
  ARTIFICIAL_CELL DUMMY
  RANGE interval

  THREADSAFE : only true if every instance has its own distinct Random
  POINTER donotuse
}

PARAMETER {
	interval	= 100 (ms) <1e-9,1e9>: time between spikes (msec)

}

ASSIGNED {
	on
	donotuse
}


INITIAL {
	on = 0 : off

}	

NET_RECEIVE (w) {
	net_event(t)
}