#!/usr/bin/env python3
"""
Debug script to check what attributes are available on a soma section with napp mechanism.
"""

from myogen.utils.nmodl import compile_nmodl_files, load_nmodl_mechanisms
from myogen.simulator.neuron.cells import AlphaMN

# Compile and load NMODL mechanisms
compile_nmodl_files()
load_nmodl_mechanisms()

# Create a simple AlphaMN cell
cell = AlphaMN(
    segments__count=1,
    mode="active",
    dendrites__count=1,
    model="NERLab",
)

print("Available attributes on soma section:")
soma_attrs = [attr for attr in dir(cell.soma) if not attr.startswith('_')]
print(f"Total attributes: {len(soma_attrs)}")

# Look for napp-related attributes
napp_attrs = [attr for attr in soma_attrs if 'napp' in attr.lower()]
print(f"\nNapp-related attributes ({len(napp_attrs)}):")
for attr in sorted(napp_attrs):
    print(f"  {attr}")

# Look for alpha-related attributes
alpha_attrs = [attr for attr in soma_attrs if 'alpha' in attr.lower()]
print(f"\nAlpha-related attributes ({len(alpha_attrs)}):")
for attr in sorted(alpha_attrs):
    print(f"  {attr}")

# Look for m_alpha attributes specifically
m_alpha_attrs = [attr for attr in soma_attrs if 'm_alpha' in attr.lower()]
print(f"\nM_alpha-related attributes ({len(m_alpha_attrs)}):")
for attr in sorted(m_alpha_attrs):
    print(f"  {attr}")

# Try to access some known working parameters
print(f"\nTesting known working parameters:")
try:
    print(f"  gl_napp: {cell.soma.gl_napp}")
except AttributeError as e:
    print(f"  gl_napp: ERROR - {e}")

try:
    print(f"  gnabar_napp: {cell.soma.gnabar_napp}")
except AttributeError as e:
    print(f"  gnabar_napp: ERROR - {e}")

# Try to access the problematic parameters
print(f"\nTesting problematic parameters:")
try:
    print(f"  m_alpha_A: {cell.soma.m_alpha_A}")
except AttributeError as e:
    print(f"  m_alpha_A: ERROR - {e}")

try:
    print(f"  m_alpha_A_napp: {cell.soma.m_alpha_A_napp}")
except AttributeError as e:
    print(f"  m_alpha_A_napp: ERROR - {e}")
