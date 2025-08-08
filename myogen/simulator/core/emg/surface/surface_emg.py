import warnings

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.muscle import Muscle
from myogen.simulator.core.spike_train import MotorNeuronPool
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.utils.types import (
    SURFACE_MUAP_SHAPE__TENSOR,
    SURFACE_EMG__TENSOR,
    beartowertype,
)

# Suppress warnings for cleaner output during batch simulations
warnings.filterwarnings("ignore")


@beartowertype
class SurfaceEMG:
    """
    Surface Electromyography (sEMG) Simulation.

    This class provides a simulation framework for generating
    surface electromyography signals from the muscle. It implements the
    multi-layered cylindrical volume conductor model from Farina et al. 2004 [1]_.

    Parameters
    ----------
    muscle_model : Muscle
        Pre-computed muscle model (see :class:`myogen.simulator.Muscle`).
    electrode_arrays : list[SurfaceElectrodeArray]
        List of electrode arrays to use for simulation (see :class:`myogen.simulator.SurfaceElectrodeArray`).
    sampling_frequency__Hz : float, default=2048.0
        Sampling frequency in Hz. Default is set to 2048 Hz as used by the Quattrocento (OT Bioelettronica, Turin, Italy) system.
    sampling_points_in_t_and_z_domains : int, default=256
        Spatial and temporal discretization resolution for numerical integration.
        Controls the accuracy of the volume conductor calculations but significantly
        impacts computational cost (scales quadratically).
        Higher values provide better numerical accuracy at the expense of simulation time.
        Default is set to 256 samples.
    sampling_points_in_theta_domain : int, default=180
        Angular discretization for cylindrical coordinate system in degrees.
        Higher values provide better spatial resolution for circumferential electrode placement studies.
        Default is set to 180 points, which provides 2° angular resolution.
        This is suitable for most EMG studies.
    MUs_to_simulate : list[int], optional
        Indices of motor units to simulate. If None, all motor units are simulated.
        Default is None. For computational efficiency, consider
        simulating subsets for initial analysis.
        Indices correspond to the recruitment order (0 is recruited first).

    References
    ----------
    .. [1] Farina, D., Mesin, L., Martina, S., Merletti, R., 2004. A surface EMG generation model with multilayer cylindrical description of the volume conductor. IEEE Transactions on Biomedical Engineering 51, 415–426. https://doi.org/10.1109/TBME.2003.820998
    """

    def __init__(
        self,
        muscle_model: Muscle,
        electrode_arrays: list[SurfaceElectrodeArray],
        sampling_frequency__Hz: float = 2048.0,
        sampling_points_in_t_and_z_domains: int = 256,
        sampling_points_in_theta_domain: int = 180,
        MUs_to_simulate: list[int] | None = None,
    ):
        self.muscle_model = muscle_model
        self.electrode_arrays = electrode_arrays
        self.sampling_frequency__Hz = sampling_frequency__Hz
        self.mean_conduction_velocity__m_s = (
            self.muscle_model.mean_conduction_velocity__m_s
        )
        self.sampling_points_in_t_and_z_domains = sampling_points_in_t_and_z_domains
        self.sampling_points_in_theta_domain = sampling_points_in_theta_domain
        self.MUs_to_simulate = MUs_to_simulate
        self.mean_fiber_length__mm = self.muscle_model.mean_fiber_length__mm
        self.var_fiber_length__mm = self.muscle_model.var_fiber_length__mm
        self.radius_bone__mm = self.muscle_model.radius_bone__mm
        self.fat_thickness__mm = self.muscle_model.fat_thickness__mm
        self.skin_thickness__mm = self.muscle_model.skin_thickness__mm
        self.muscle_conductivity_radial__S_m = (
            self.muscle_model.muscle_conductivity_radial__S_m
        )
        self.muscle_conductivity_longitudinal__S_m = (
            self.muscle_model.muscle_conductivity_longitudinal__S_m
        )
        self.fat_conductivity__S_m = self.muscle_model.fat_conductivity__S_m
        self.skin_conductivity__S_m = self.muscle_model.skin_conductivity__S_m

        # Calculate total radius
        self.radius_total = (
            self.muscle_model.radius__mm
            + self.fat_thickness__mm
            + self.skin_thickness__mm
        )

    def simulate_muaps(self) -> list[SURFACE_MUAP_SHAPE__TENSOR]:
        """
        Simulate MUAPs for all electrode arrays using the provided muscle model.

        Returns
        -------
        list[SURFACE_MUAP_SHAPE__TENSOR]
            List of generated MUAP templates for each electrode array.
        """
        # Set default MUs to simulate
        if self.MUs_to_simulate is None:
            self.MUs_to_simulate = list(
                range(len(self.muscle_model.resulting_number_of_innervated_fibers))
            )

        # Calculate innervation zone variance
        innervation_zone_variance = (
            self.mean_fiber_length__mm * 0.1
        )  # 10% of the mean fiber length (see Botelho et al. 2019 [6]_)

        # Extract fiber counts
        number_of_fibers_per_MUs = (
            self.muscle_model.resulting_number_of_innervated_fibers
        )

        # Create time array
        t = np.linspace(
            0,
            (self.sampling_points_in_t_and_z_domains - 1)
            / self.sampling_frequency__Hz
            * 1e-3,
            self.sampling_points_in_t_and_z_domains,
        )

        # Pre-calculate innervation zones
        innervation_zones = RANDOM_GENERATOR.uniform(
            low=-innervation_zone_variance / 2,
            high=innervation_zone_variance / 2,
            size=len(self.MUs_to_simulate),
        )

        # Initialize output arrays for each electrode array
        # Each electrode array can have different dimensions
        electrode_array_results = []

        for array_idx, electrode_array in enumerate(self.electrode_arrays):
            # Initialize array for this electrode configuration
            array_result = np.zeros(
                (
                    len(self.MUs_to_simulate),
                    electrode_array.num_rows,
                    electrode_array.num_cols,
                    len(t),
                )
            )

            # Matrix optimization variables
            A_matrix = None
            B_incomplete = None

            # Process each motor unit
            for MU_number, MU_index in enumerate(self.MUs_to_simulate):
                number_of_fibers = number_of_fibers_per_MUs[MU_index]

                if number_of_fibers == 0:
                    continue

                # Get fiber positions
                position_of_fibers = self.muscle_model.resulting_fiber_assignment(
                    MU_index
                )
                innervation_zone = innervation_zones[MU_number]

                # Process each fiber
                for fiber_number in tqdm(
                    range(number_of_fibers),
                    desc=f"Electrode Array {array_idx + 1}/{len(self.electrode_arrays)} MU {MU_number + 1}/{len(self.MUs_to_simulate)}",
                    unit="fibers",
                ):
                    fiber_position = position_of_fibers[fiber_number]

                    # Calculate fiber distance from center
                    R = np.sqrt(fiber_position[0] ** 2 + fiber_position[1] ** 2)

                    # Generate fiber length
                    fiber_length__mm = (
                        self.mean_fiber_length__mm
                        + RANDOM_GENERATOR.uniform(
                            low=-self.var_fiber_length__mm,
                            high=self.var_fiber_length__mm,
                        )
                    )

                    # Calculate fiber end positions
                    L2 = abs(innervation_zone - fiber_length__mm / 2)
                    L1 = abs(innervation_zone + fiber_length__mm / 2)

                    # Use the new simulate_fiber_v2 function
                    if fiber_number == 0 or A_matrix is None:
                        phi_temp, A_matrix, B_incomplete = simulate_fiber_v2(
                            Fs=self.sampling_frequency__Hz * 1e-3,
                            v=self.mean_conduction_velocity__m_s,
                            N=self.sampling_points_in_t_and_z_domains,
                            M=self.sampling_points_in_theta_domain,
                            r=self.radius_total,
                            r_bone=self.radius_bone__mm,
                            th_fat=self.fat_thickness__mm,
                            th_skin=self.skin_thickness__mm,
                            R=R,
                            L1=L1,
                            L2=L2,
                            zi=innervation_zone,
                            electrode_array=electrode_array,
                            sig_muscle_rho=self.muscle_conductivity_radial__S_m,
                            sig_muscle_z=self.muscle_conductivity_longitudinal__S_m,
                            sig_skin=self.skin_conductivity__S_m,
                            sig_fat=self.fat_conductivity__S_m,
                        )
                    else:
                        phi_temp, _, _ = simulate_fiber_v2(
                            Fs=self.sampling_frequency__Hz * 1e-3,
                            v=self.mean_conduction_velocity__m_s,
                            N=self.sampling_points_in_t_and_z_domains,
                            M=self.sampling_points_in_theta_domain,
                            r=self.radius_total,
                            r_bone=self.radius_bone__mm,
                            th_fat=self.fat_thickness__mm,
                            th_skin=self.skin_thickness__mm,
                            R=R,
                            L1=L1,
                            L2=L2,
                            zi=innervation_zone,
                            electrode_array=electrode_array,
                            sig_muscle_rho=self.muscle_conductivity_radial__S_m,
                            sig_muscle_z=self.muscle_conductivity_longitudinal__S_m,
                            sig_skin=self.skin_conductivity__S_m,
                            sig_fat=self.fat_conductivity__S_m,
                            A_matrix=A_matrix,
                            B_incomplete=B_incomplete,
                        )

                    array_result[MU_number] += phi_temp

            electrode_array_results.append(array_result)

        # Store results
        self.resulting_muaps_arrays = electrode_array_results

        return electrode_array_results

    def simulate_surface_emg(
        self, motor_neuron_pool: MotorNeuronPool
    ) -> list[SURFACE_EMG__TENSOR]:
        """
        Generate surface EMG signals for all electrode arrays using the provided motor neuron pool.

        Parameters
        ----------
        motor_neuron_pool : MotorNeuronPool
            Motor neuron pool with spike trains computed (see :class:`myogen.simulator.MotorNeuronPool`).

        Returns
        -------
        list[SURFACE_EMG__TENSOR]
            Surface EMG signals for each electrode array

        Raises
        ------
        AttributeError
            If MUAP templates have not been generated. Call simulate_muaps() first.
        """
        if not hasattr(self, "resulting_muaps_arrays"):
            raise AttributeError(
                "MUAP templates have not been generated. Call simulate_muaps() first."
            )

        self.motor_neuron_pool = motor_neuron_pool

        # Handle MUs to simulate
        if self.MUs_to_simulate is None:
            MUs_to_simulate = set(
                range(len(self.muscle_model.resulting_number_of_innervated_fibers))
            )
        else:
            MUs_to_simulate = set(self.MUs_to_simulate)

        emg_results = []

        for array_idx, muap_array in enumerate(self.resulting_muaps_arrays):
            # Temporal resampling
            muap_shapes = np.zeros(
                (
                    muap_array.shape[0],
                    muap_array.shape[1],
                    muap_array.shape[2],
                    int(
                        muap_array.shape[3]
                        / self.sampling_frequency__Hz
                        * 1
                        / self.motor_neuron_pool.timestep__ms
                        * 1000
                    ),
                )
            )

            for muap_nr in range(muap_shapes.shape[0]):
                for row in range(muap_shapes.shape[1]):
                    for col in range(muap_shapes.shape[2]):
                        muap_shapes[muap_nr, row, col] = np.interp(
                            np.arange(
                                0,
                                muap_array.shape[-1] / self.sampling_frequency__Hz,
                                self.motor_neuron_pool.timestep__ms / 1000,
                            ),
                            np.arange(
                                0,
                                muap_array.shape[-1] / self.sampling_frequency__Hz,
                                1 / self.sampling_frequency__Hz,
                            ),
                            muap_array[muap_nr, row, col],
                        )

            n_pools = motor_neuron_pool.spike_trains.shape[0]
            n_rows = muap_shapes.shape[1]
            n_cols = muap_shapes.shape[2]

            # Initialize result array
            sample_conv = np.convolve(
                motor_neuron_pool.spike_trains[0, 0],
                muap_shapes[0, 0, 0],
                mode="same",
            )

            surface_emg = np.zeros((n_pools, n_rows, n_cols, len(sample_conv)))

            muap_shapes /= np.max(np.abs(muap_shapes))  # Normalize MUAP shapes

            # Perform convolution for each pool using GPU acceleration if available
            if HAS_CUPY:
                # Use GPU acceleration with CuPy
                spike_gpu = cp.asarray(motor_neuron_pool.spike_trains)
                muap_gpu = cp.asarray(muap_shapes)
                surface_emg_gpu = cp.zeros((n_pools, n_rows, n_cols, len(sample_conv)))

                for pool_idx in tqdm(
                    range(n_pools),
                    desc=f"Electrode Array {array_idx + 1}/{len(self.resulting_muaps_arrays)} Surface EMG (GPU)",
                    unit="pools",
                ):
                    active_neuron_indices = set(
                        motor_neuron_pool.active_neuron_indices[pool_idx]
                    )

                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all active MUs on GPU
                            convolutions = cp.array(
                                [
                                    cp.correlate(
                                        spike_gpu[pool_idx, mu_idx],
                                        muap_gpu[i, row_idx, col_idx],
                                        mode="same",
                                    )
                                    for i, mu_idx in enumerate(
                                        MUs_to_simulate.intersection(
                                            active_neuron_indices
                                        )
                                    )
                                ]
                            )
                            # Sum across MUAPs on GPU
                            if len(convolutions) > 0:
                                surface_emg_gpu[pool_idx, row_idx, col_idx] = cp.sum(
                                    convolutions, axis=0
                                )

                # Transfer results back to CPU
                surface_emg = cp.asnumpy(surface_emg_gpu)
            else:
                # Fallback to CPU computation with NumPy
                for pool_idx in tqdm(
                    range(n_pools),
                    desc=f"Electrode Array {array_idx + 1}/{len(self.resulting_muaps_arrays)} Surface EMG (CPU)",
                    unit="pools",
                ):
                    active_neuron_indices = set(
                        motor_neuron_pool.active_neuron_indices[pool_idx]
                    )

                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all active MUs
                            convolutions = []
                            for i, mu_idx in enumerate(
                                MUs_to_simulate.intersection(active_neuron_indices)
                            ):
                                conv = np.correlate(
                                    motor_neuron_pool.spike_trains[pool_idx, mu_idx],
                                    muap_shapes[i, row_idx, col_idx],
                                    mode="same",
                                )
                                convolutions.append(conv)

                            if convolutions:
                                surface_emg[pool_idx, row_idx, col_idx] = np.sum(
                                    convolutions, axis=0
                                )

            surface_emg_resampled = np.zeros(
                (
                    n_pools,
                    n_rows,
                    n_cols,
                    int(
                        surface_emg.shape[-1]
                        / (1 / self.motor_neuron_pool.timestep__ms * 1000)
                        * self.sampling_frequency__Hz
                    ),
                )
            )
            for pool_idx in range(n_pools):
                for row_idx in range(n_rows):
                    for col_idx in range(n_cols):
                        surface_emg_resampled[pool_idx, row_idx, col_idx] = np.interp(
                            np.arange(
                                0,
                                surface_emg.shape[-1]
                                * (self.motor_neuron_pool.timestep__ms / 1000),
                                1 / self.sampling_frequency__Hz,
                            ),
                            np.arange(
                                0,
                                surface_emg.shape[-1]
                                * (self.motor_neuron_pool.timestep__ms / 1000),
                                self.motor_neuron_pool.timestep__ms / 1000,
                            ),
                            surface_emg[pool_idx, row_idx, col_idx],
                        )

            emg_results.append(surface_emg_resampled)

        self.surface_emg__tensors = emg_results
        return emg_results

    def add_noise(
        self, snr_db: float, noise_type: str = "gaussian"
    ) -> list[SURFACE_EMG__TENSOR]:
        """
        Add noise to all electrode arrays.

        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in dB
        noise_type : str, default="gaussian"
            Type of noise to add

        Returns
        -------
        list[SURFACE_EMG__TENSOR]
            Noisy EMG signals for each electrode array
        """
        if not hasattr(self, "surface_emg__tensors"):
            raise ValueError(
                "Surface EMG has not been simulated. Call simulate_surface_emg() first."
            )

        noisy_emg_results = []

        for _, emg_array in enumerate(self.surface_emg__tensors):
            # Calculate signal power
            signal_power = np.mean(emg_array**2)

            # Calculate noise power
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear

            # Generate noise
            if noise_type.lower() == "gaussian":
                noise_std = np.sqrt(noise_power)
                noise = RANDOM_GENERATOR.normal(
                    loc=0.0, scale=noise_std, size=emg_array.shape
                )
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            # Add noise
            noisy_emg = emg_array + noise
            noisy_emg_results.append(noisy_emg)

        self.noisy_surface_emg__tensors = noisy_emg_results
        return noisy_emg_results


# For backward compatibility - run simulation if script is executed directly
if __name__ == "__main__":
    from myogen import simulator
    from myogen.utils.plotting.muscle import show_mf_centers, show_innervation_areas_2d
    import seaborn as sns

    # Generate recruitment thresholds#
    N_MU = 100  # Number of motor units
    r = 4.9  # Radius of the muscle in mm
    fiber_density = 400  # fibers per mm²
    max_innervation_ratio = 1 / 4  # Maximum innervation area to total muscle area ratio
    grid_resolution = 256  # Grid resolution for muscle simulation

    recruitment_thresholds, recruitment_thresholds_starting_by_zeros = (
        simulator.generate_mu_recruitment_thresholds(
            N=N_MU,
            recruitment_range=50,  # Recruitment range in % MVC
            mode="konstantin",  # Recruitment pattern model
            konstantin__max_threshold=1,
        )
    )

    plt.plot(recruitment_thresholds, marker="o", linestyle="None")
    plt.show()

    # #Create muscle object
    # muscle = simulator.Muscle(
    #     recruitment_thresholds=recruitment_thresholds,
    #     radius__mm=r,
    #     fiber_density__fibers_per_mm2=fiber_density,
    #     max_innervation_area_to_total_muscle_area__ratio=max_innervation_ratio,
    #     grid_resolution=grid_resolution,
    #     autorun=True,
    # )
    # joblib.dump(muscle, "muscle.pkl")  # Save for reuse

    # load muscle object
    muscle = joblib.load("muscle.pkl")

    with plt.xkcd():
        plt.rcParams.update({"font.size": 24})
        plt.rcParams.update({"axes.titlesize": 24})
        plt.rcParams.update({"axes.labelsize": 24})
        plt.rcParams.update({"xtick.labelsize": 24})
        plt.rcParams.update({"ytick.labelsize": 24})
        show_mf_centers(muscle, ax=plt.gca())

        sns.despine(trim=True, offset=10)
    plt.show()

    plt.figure(figsize=(6, 6))

    plt.rcParams.update({"font.size": 24})
    plt.rcParams.update({"axes.titlesize": 24})
    plt.rcParams.update({"axes.labelsize": 24})
    plt.rcParams.update({"xtick.labelsize": 24})
    plt.rcParams.update({"ytick.labelsize": 24})

    selected_indices = np.array(list(reversed(range(120))))
    with plt.xkcd():
        show_innervation_areas_2d(
            muscle, indices_to_plot=selected_indices, ax=plt.gca()
        )
    plt.title("Motor Unit Innervation Areas (First 10 MUs)")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.axis("equal")

    plt.show()

    surface_emg = SurfaceEMG(
        muscle_model=muscle,
        MUs_to_simulate=[0],
    )

    MUAP = surface_emg.simulate_muaps(
        verbose=True,
        show_plots=False,  # Center at the muscle belly
    )

    np.save("FDI_MUAPs_multi.npy", MUAP)


#######################################################################################################
###################################### References #####################################################
#######################################################################################################

# BOTELHO, Diego; CURRAN, Kathleen; LOWERY, Madeleine M. Anatomically accurate
# model of EMG during index finger flexion and abduction derived from diffusion tensor
# imaging. PLOS Computational Biology, [S. l.], v. 15, n. 8, p. e1007267, 2019. DOI:
# 10.1371/journal.pcbi.1007267.

# BRODAR, Vida. Observations on Skin Thickness and Subcutaneous Tissue in Man. Zeitschrift
# Für Morphologie Und Anthropologie, [S. l.], v. 50, n. 3, p. 386–95, 1960. Available in:
# https://about.jstor.org/terms

# ENOKA, Roger M.; FUGLEVAND, Andrew J. Motor unit physiology: Some unresolved issues. Muscle & Nerve,
# [S. l.], v. 24, n. 1, p. 4–17, 2001. DOI: 10.1002/1097-4598(200101)24:1<4::AID-MUS13>3.0.CO;2-F.

# FEINSTEIN, Bertram; LINDEGARD, Bengt; NYMAN, Eberhard; WOHLFART, Gunnar.
# MORPHOLOGIC STUDIES OF MOTOR UNITS IN NORMAL HUMAN MUSCLES. Cells Tissues
# Organs, [S. l.], v. 23, n. 2, p. 127–142, 1955. DOI: 10.1159/000140989.

# HWANG, Kun; HUAN, Fan; KIM, Dae Joong. Muscle fibre types of the lumbrical, interossei,
# flexor, and extensor muscles moving the index finger. Journal of Plastic Surgery and Hand
# Surgery, [S. l.], v. 47, n. 4, p. 268–272, 2013. DOI: 10.3109/2000656X.2012.755988.

# JACOBSON, Mark D.; RAAB, Rajnik; FAZELI, Babak M.; ABRAMS, Reid A.; BOTTE, Michael J.;
# LIEBER, Richard L. Architectural design of the human intrinsic hand muscles. The Journal of
# Hand Surgery, [S. l.], v. 17, n. 5, p. 804–809, 1992. DOI: 10.1016/0363-5023(92)90446-V.

# ROELEVELD, K.; BLOK, J. H.; STEGEMAN, D. F.; VAN OOSTEROM, A. Volume conduction
# models for surface EMG; confrontation with measurements. Journal of Electromyography
# and Kinesiology, [S. l.], v. 7, n. 4, p. 221–232, 1997. DOI: 10.1016/S1050-6411(97)00009-6.

# STÖRCHLE, Paul; MÜLLER, Wolfram; SENGEIS, Marietta; LACKNER, Sonja; HOLASEK, Sandra;
# FÜRHAPTER-RIEGER, Alfred. Measurement of mean subcutaneous fat thickness: eight
# standardised ultrasound sites compared to 216 randomly selected sites. Scientific Reports,
# [S. l.], v. 8, n. 1, p. 16268, 2018. DOI: 10.1038/s41598-018-34213-0.
