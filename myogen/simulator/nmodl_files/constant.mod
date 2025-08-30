: constant current for custom initialization
NEURON {
  SUFFIX Constant
  NONSPECIFIC_CURRENT i
  RANGE i, ic
}
UNITS {
  (mA) = (milliamp)
}
PARAMETER {
  ic = 0.000 (mA/cm2)
}
ASSIGNED {
  i (mA/cm2)
}
BREAKPOINT {
  i = ic
}
