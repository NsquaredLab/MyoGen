# Implementation Notes - January 22, 2026

## Session Summary

This session focused on integrating **jaxley-mech** Hodgkin-Huxley ion channels into the MyoGen Jaxley backend and resolving critical package compatibility issues to support **numpy 2.x**.

---

## 1. Jaxley-Mech Integration

### Goal
Use standardized Hodgkin-Huxley channels from `jaxley-mech` instead of custom ion channel implementations in the Jaxley backend.

### Files Modified

#### `/myogen/simulator/jaxley/cells_api.py`

**Added jaxley-mech imports with version compatibility:**
```python
# Try importing jaxley-mech channels (Hodgkin-Huxley based)
# Note: In jaxley-mech 0.1.0, the module is 'hh', in 0.3.1+ it's 'hodgkin52'
JAXLEY_MECH_AVAILABLE = False
JaxleyMechNa = None
JaxleyMechK = None
JaxleyMechLeak = None

try:
    # Try newer module path first (jaxley-mech >= 0.3.0)
    from jaxley_mech.channels.hodgkin52 import (
        Na as JaxleyMechNa,
        K as JaxleyMechK,
        Leak as JaxleyMechLeak,
    )
    JAXLEY_MECH_AVAILABLE = True
except ImportError:
    try:
        # Try older module path (jaxley-mech 0.1.0 - 0.2.0)
        from jaxley_mech.channels.hh import (
            Na as JaxleyMechNa,
            K as JaxleyMechK,
            Leak as JaxleyMechLeak,
        )
        JAXLEY_MECH_AVAILABLE = True
    except ImportError:
        pass
```

**Added `use_jaxley_mech` parameter to `AlphaMN` class:**
```python
class AlphaMN(_Cell):
    def __init__(
        self,
        segments__count: int = 1,
        mode: Literal["active", "passive"] = "active",
        dendrites__count: int = 4,
        model: Literal["NERLab", "Powers2017"] = "NERLab",
        use_jaxley_mech: bool = False,  # NEW PARAMETER
        class__ID: Optional[int] = None,
        pool__ID: int | None = None,
    ):
        # ... validation and initialization
        if use_jaxley_mech and not JAXLEY_MECH_AVAILABLE:
            raise ImportError(
                "jaxley-mech is not installed. Install it with: pip install jaxley-mech"
            )
```

**Added channel insertion method:**
```python
def _insert_jaxley_mech_channels(self):
    """Insert jaxley-mech Hodgkin-Huxley channels."""
    soma = self.cell.branch(self.soma_branch_idx)
    
    # Insert HH channels from jaxley-mech
    na = JaxleyMechNa(name="Na")
    k = JaxleyMechK(name="K")
    leak = JaxleyMechLeak(name="Leak")
    
    soma.insert(na)
    soma.insert(k)
    soma.insert(leak)
    
    # Set standard HH conductances (S/cm²)
    self.cell.set("Na_gNa", 120e-3)   # 120 mS/cm²
    self.cell.set("K_gK", 36e-3)      # 36 mS/cm²
    self.cell.set("Leak_gLeak", 0.3e-3)  # 0.3 mS/cm²
    
    # Set reversal potentials
    self.cell.set("Na_eNa", 50.0)     # mV
    self.cell.set("K_eK", -77.0)      # mV
    self.cell.set("Leak_eLeak", -54.3)  # mV
```

#### `/myogen/simulator/jaxley/populations/motor_neurons.py`

**Added `use_jaxley_mech` parameter to `AlphaMN__Pool`:**
```python
class AlphaMN__Pool(_Pool):
    def __init__(
        self,
        n: int,
        recruitment_thresholds__array: Quantity__nA,
        model: Literal["NERLab", "Powers2017"] = "NERLab",
        mode: Literal["active", "passive"] = "active",
        use_jaxley_mech: bool = False,  # NEW PARAMETER
    ):
        self.use_jaxley_mech = use_jaxley_mech
        # ... rest of initialization
```

---

## 2. Poisson Process Generator Fix

### Problem
The Jaxley Poisson process implementation was producing ~4x higher firing rates than expected (247 pps vs 55 pps).

### Root Cause
The original implementation used a simple probability-per-timestep formula that was incorrect:
```python
# WRONG - old implementation
self._lambda = y * self.N * self.dt / 1000.0
return 1 if self.rng.random() < self._lambda else 0
```

### Solution
Rewrote to match the Cython implementation's algorithm using accumulated intensity and exponential thresholds:

```python
class _PoissonProcessGenerator__Jaxley:
    """
    Poisson process spike generator for Jaxley cells.
    
    Uses the same algorithm as the Cython implementation: accumulates input
    intensity and compares against an exponentially-distributed threshold.
    """
    
    def __init__(self, seed, N, dt):
        self.N = N  # Batch size for threshold (not max rate!)
        self.dt = dt  # Time step in ms
        self.rng = np.random.RandomState(seed)
        
        # State variables matching Cython implementation
        self.yi = 0.0  # Accumulated input intensity
        self.thres = self._generate_threshold()  # Exponential threshold
    
    def _generate_threshold(self):
        """Generate exponential threshold using batch method."""
        aux = 1.0
        for _ in range(self.N):
            aux *= self.rng.random()
        return -(1.0 / self.N) * np.log(aux)
    
    def compute(self, y):
        """
        Generate Poisson spike for given input rate.
        
        Parameters
        ----------
        y : float
            Input intensity (rate) in Hz at current time step
            
        Returns
        -------
        int
            1 if spike occurs, 0 otherwise
        """
        spk = 0
        # Integrate input intensity: y is rate in Hz, dt is in ms
        self.yi += y * self.dt * 1e-3
        
        # Check if threshold crossed
        if self.yi >= self.thres:
            spk = 1
            self.yi = 0.0  # Reset accumulator
            self.thres = self._generate_threshold()  # New threshold
        
        return spk
```

### Result
DD firing rate now correctly matches NEURON reference: **55.1 ± 4.4 pps** (expected ~55 pps)

---

## 3. Package Compatibility Updates (NumPy 2.x Support)

### The Problem: Diamond Dependency Conflict

```
                    MyoGen
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    Scientific Stack          JAX Stack
    (elephant, neo,           (jaxley, jaxley-mech)
     quantities)                     │
           │                         │
           ▼                         ▼
    numpy < 2.0               numpy >= 2.0
```

### Solution

1. **Install elephant from numpy 2.0 support branch:**
   ```bash
   pip install git+https://github.com/INM-6/elephant.git@support_numpy_2
   ```

2. **Update `/pyproject.toml` numpy constraint:**
   ```toml
   # Changed from:
   "numpy>=1.26,<2.0",
   # To:
   "numpy>=1.26",
   ```

3. **Use numpy < 2.4 for numba compatibility:**
   ```bash
   pip install "numpy<2.4"
   ```

### Final Compatible Package Versions

| Package | Version | Notes |
|---------|---------|-------|
| numpy | 2.3.5 | < 2.4 for numba |
| jax | 0.9.0 | Latest |
| jaxlib | 0.9.0 | Latest |
| jaxley | 0.13.0 | Latest |
| jaxley-mech | 0.3.1 | Latest |
| elephant | 1.2.0b1 | From INM-6/support_numpy_2 branch |
| quantities | 0.16.4 | Latest |
| neo | 0.14.3 | Latest |
| numba | 0.63.1 | Latest |

### Installation Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install elephant with numpy 2.0 support
pip install git+https://github.com/INM-6/elephant.git@support_numpy_2

# Upgrade JAX ecosystem
pip install "jax>=0.4.38" "jaxlib>=0.4.38" jaxley jaxley-mech --upgrade

# Ensure numpy < 2.4 for numba compatibility
pip install "numpy<2.4"
```

---

## 4. Example Usage

### Using jaxley-mech channels in Example 3:

```python
from myogen.simulator.jaxley import AlphaMN__Pool

motor_neuron_pool = AlphaMN__Pool(
    n=100,
    recruitment_thresholds__array=recruitment_thresholds,
    model="NERLab",
    mode="active",
    use_jaxley_mech=True,  # Use jaxley-mech HH channels
)
```

---

## 5. Known Issues / Future Work

1. **MN Firing Rate**: Motor neuron firing rates are still higher than expected (~172 pps vs ~17 pps). The MN model in the Jaxley example uses a simplified integrate-and-fire approach that doesn't fully match NEURON's biophysical model.

2. **Elephant Release**: Once elephant 1.2.0 is officially released with numpy 2.0 support, update the dependency to use PyPI instead of the GitHub branch.

3. **Performance**: The Jaxley DD simulation is slower than NEURON (~7 minutes vs faster). Consider optimizing the Python Poisson generator or using JAX JIT compilation.

---

## 6. Testing

Run the Jaxley Example 3 to verify the integration:

```bash
cd MyoGen
source ../.venv/bin/activate
python examples/01_basic/03_simulate_spike_trains_descending_drive_jaxley.py
```

Expected output:
```
Descending Drive neurons:
    Active neurons: 100/100
    Mean firing rate: 55.1 ± 4.4 pps  ← Should be ~55 pps
    Rate range: 48.7 - 65.4 pps
```

---

## Summary of Files Changed

1. `/myogen/simulator/jaxley/cells_api.py`
   - Added jaxley-mech imports with version fallback
   - Added `use_jaxley_mech` parameter to `AlphaMN`
   - Added `_insert_jaxley_mech_channels()` method
   - **Fixed Poisson process generator algorithm**

2. `/myogen/simulator/jaxley/populations/motor_neurons.py`
   - Added `use_jaxley_mech` parameter propagation

3. `/pyproject.toml`
   - Updated numpy constraint to allow 2.x

4. `/examples/01_basic/03_simulate_spike_trains_descending_drive_jaxley.py`
   - Added `use_jaxley_mech=True` to AlphaMN__Pool creation
