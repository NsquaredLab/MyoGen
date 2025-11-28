"""Profile to see where time is actually spent."""

import cProfile
import pstats
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray

# Medium configuration
N, M = 128, 32
Fs, v = 2.048, 4.0
r, r_bone = 40.0, 0.0
th_fat, th_skin = 5.0, 2.0
R, L1, L2, zi = 10.0, 50.0, 50.0, 0.0

electrode_array = SurfaceElectrodeArray(
    num_rows=4,
    num_cols=4,
    inter_electrode_distances__mm=5.0,
    electrode_radius__mm=1.0,
)

sig_muscle_rho, sig_muscle_z = 0.5, 0.1
sig_fat, sig_skin, sig_bone = 0.04, 0.2, 0.0

print("Profiling Python version...")
profiler = cProfile.Profile()
profiler.enable()

for _ in range(10):
    simulate_fiber_v2(
        Fs, v, N, M, r, r_bone, th_fat, th_skin, R, L1, L2, zi,
        electrode_array, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin,
        sig_bone, use_cython=False
    )

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
print("\nTop 20 time-consuming operations:")
stats.print_stats(20)
