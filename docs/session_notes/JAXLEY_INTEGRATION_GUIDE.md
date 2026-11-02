# Jaxley Integration Guide for MyoGen

## ✅ Correct Integration Approach

### What's Integrated vs What's a Dependency

```
MyoGen (your code)
├── myogen/simulator/jaxley/      ✅ YOURS - Keep in repo
│   ├── cells.py                  ✅ Custom wrapper (560 lines)
│   ├── channels/constant.py      ✅ Custom channel
│   └── __init__.py               ✅ Integration code
│
└── Dependencies (don't copy)     🔵 EXTERNAL - Keep as dependencies
    ├── jaxley                    🔵 ~50,000 lines - pip install
    ├── jax                       🔵 ~100,000 lines - pip install
    └── jaxlib                    🔵 ~500 MB binary - pip install
```

## 📦 How Users Install

### Option 1: With Jaxley Support (Recommended)

```bash
pip install myogen[jaxley]
```

This installs:
- MyoGen core
- Jaxley + JAX + JAXlib as dependencies

### Option 2: Without Jaxley (NEURON only)

```bash
pip install myogen
```

This installs:
- MyoGen core
- NEURON simulator only

Users can add Jaxley later:
```bash
pip install jaxley jax jaxlib
```

## 🔧 What You Did

### 1. Added Optional Dependency

**File: `pyproject.toml`**
```toml
[project.optional-dependencies]
jaxley = [
    "jaxley>=0.13.0",
    "jax>=0.8.0",
    "jaxlib>=0.8.0",
]
```

### 2. Created MyoGen Wrapper Layer

**File: `myogen/simulator/jaxley/cells.py`**
- ✅ Base class `_JaxleyCell` with template methods
- ✅ 8 cell types: INgII, INgIb, DD, DD_Gamma, AffIa, AffII, AffIb, AlphaMN
- ✅ Geometry and parameters for each cell type
- 🔵 Uses `jaxley.Cell` and `jaxley.channels.Leak` (dependencies)

### 3. Made Import Optional

**File: `myogen/simulator/jaxley/__init__.py`**
```python
try:
    import jaxley as _jx
    JAXLEY_AVAILABLE = True
except ImportError:
    JAXLEY_AVAILABLE = False
```

Gives helpful error if user tries to use Jaxley without installing it.

## 📝 Usage Examples

### Check if Jaxley is Available

```python
from myogen.simulator import jaxley

if jaxley.JAXLEY_AVAILABLE:
    print("Jaxley is installed!")
    from myogen.simulator.jaxley.cells import INgII
else:
    print("Jaxley not installed. Use NEURON instead.")
    from myogen.simulator.neuron.cells import INgII
```

### Use Jaxley Cells

```python
try:
    from myogen.simulator.jaxley.cells import AlphaMN
    import jaxley as jx
    
    # Create cell
    mn = AlphaMN(mode="passive")
    
    # Stimulate
    current = jx.step_current(i_delay=10, i_dur=30, i_amp=0.1,
                              delta_t=0.025, t_max=50.0)
    mn.cell.stimulate(current)
    mn.cell.record("v")
    mn.cell.set("v", -71.0)
    
    # Integrate
    voltages = jx.integrate(mn.cell, delta_t=0.025, t_max=50.0)
    
except ImportError:
    print("Install with: pip install myogen[jaxley]")
```

## ⚠️ What NOT to Do

### ❌ DON'T Copy Jaxley Code

```python
# ❌ WRONG - Don't do this!
myogen/
├── third_party/
│   └── jaxley/           # ❌ Copied Jaxley code
│       ├── integrate.py  # 50,000+ lines
│       └── ...
```

**Why it's wrong:**
1. Legal: Must include all Apache 2.0 license notices
2. Maintenance: You'd have to update 50,000 lines when Jaxley updates
3. Size: Adds 100+ MB to your repo
4. Dependencies: Jaxley has its own dependencies
5. Best practice: Libraries should be dependencies

### ✅ DO Use as Dependency

```toml
# ✅ CORRECT - pyproject.toml
[project.optional-dependencies]
jaxley = [
    "jaxley>=0.13.0",  # ✅ Reference, don't copy
]
```

## 📋 Files You Created (Keep These)

These are YOUR code - keep them in your repo:

```
✅ myogen/simulator/jaxley/
   ├── __init__.py                    # 50 lines - yours
   ├── cells.py                       # 560 lines - yours
   ├── channels/
   │   ├── __init__.py               # 10 lines - yours
   │   └── constant.py               # 150 lines - yours
   ├── README.md                      # Documentation - yours
   ├── INSTALL.md                     # Installation guide - yours
   └── PROGRESS.md                    # Development notes - yours

✅ Root level benchmarks/tests:
   ├── benchmark_standalone.py        # 300 lines - yours
   ├── benchmark_batch.py             # 300 lines - yours
   ├── test_all_cells.py              # 70 lines - yours
   └── WHY_NEURON_FASTER.md          # Documentation - yours
```

**Total:** ~1,500 lines of custom code

## 🎯 Summary

**What you're integrating:**
- ✅ Your custom MyoGen wrapper classes (cells.py)
- ✅ Your custom channels (constant.py)
- ✅ Your benchmarks and tests
- 🔵 Jaxley as a pip dependency (not copied)

**This is the RIGHT way** to integrate external libraries:
1. Add as optional dependency in pyproject.toml
2. Create your custom wrapper layer
3. Let pip handle installation
4. Never copy library source code

**Users install with:**
```bash
pip install myogen[jaxley]
```

And get:
- Your MyoGen code
- Jaxley automatically installed as dependency
- All working together seamlessly

## 🚀 Next Steps

To finish the integration:

1. ✅ **Done**: Added Jaxley to pyproject.toml
2. ✅ **Done**: Created wrapper classes  
3. ✅ **Done**: Made imports optional
4. **TODO**: Update main documentation to mention Jaxley option
5. **TODO**: Add examples showing NEURON vs Jaxley usage
6. **TODO**: Implement active ion channels in Jaxley cells

## 📚 References

- **Jaxley GitHub**: https://github.com/jaxleyverse/jaxley
- **Jaxley License**: Apache 2.0 (requires attribution, allows modification)
- **JAX**: https://github.com/google/jax
- **Best Practice**: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-optional-dependencies
