"""
Alpha motor neuron populations for motor output.

This module contains the population class for alpha motor neurons, which form
the final common pathway for motor control and drive muscle contraction.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from myogen.simulator.neuron import cells
from myogen.utils.config_loader import load_yaml_config, merge_configs
from myogen.utils.decorators import beartowertype
from myogen.utils.types import RECRUITMENT_THRESHOLDS__ARRAY

from .base import _exp_interp, _Pool


@beartowertype
class AlphaMN__Pool(_Pool):
    """
    Container for a population of alpha motor neurons.

    Manages a collection of AlphaMN (alpha motor neuron) cells with different
    biophysical models: ModALS or Powers2017. These cells form the final
    common pathway for motor control.

    Parameters
    ----------
    n : int
        Number of alpha motor neurons to create.
    recruitment_thresholds__array : RECRUITMENT_THRESHOLDS__ARRAY, optional
        Array of recruitment thresholds for each motor neuron, by default None.
    config_file : str or Path, optional
        Path to YAML configuration file containing model parameters.
        If provided, parameters from this file will be used as defaults,
        which can be overridden by explicitly passed parameters.
        Can be a filename (searches in myogen/config/), relative path, or absolute path.
        By default uses "alpha_mn_default.yaml".
    model : str, optional
        Motor neuron model type ("NERLab" or "Powers2017"), by default "NERLab".
    mode : str, optional
        Simulation mode ("active" or "passive"), by default "active".
    axon_velocities : tuple[float, float], optional
        Min and max axon conduction velocities (m/s), by default (50, 65).
    axon_length : float, optional
        Length of the axon (mm), by default 0.6.
    gamma : float, optional
        Neuromodulation level (a.u.), by default 0.2.
    cell_index : Optional[int], optional
        Specific cell index to create (creates only one cell), by default None.
    lambda_factor : float, optional
        Lambda factor for Powers2017 model persistent sodium scaling, by default 1.0.
    initial_voltage__mV : float or list[float], optional
        Initial membrane voltage (mV), by default -67.
    spike_threshold__mV : float, optional
        Spike detection threshold for recording motor neuron spikes, by default 50.0.
        Motor neurons have large action potentials (80-100 mV) requiring higher thresholds.

    Powers2017 Model Parameters (required when model="Powers2017")
    --------------------------------------------------------
    soma_length_range : tuple[float, float, float], optional
        Soma length [min, max, curve] (um).
    soma_diameter_range : tuple[float, float, float], optional
        Soma diameter [min, max, curve] (um).
    soma_capacitance_range : tuple[float, float, float], optional
        Soma capacitance [min, max, curve] (uF/cm²).
    soma_passive_conductance_range : tuple[float, float, float], optional
        Soma passive conductance [min, max, curve] (S/cm²).
    soma_passive_reversal_range : tuple[float, float, float], optional
        Soma passive reversal potential [min, max, curve] (mV).
    soma_na3rp_conductance_range : tuple[float, float, float], optional
        Soma Na3RP conductance [min, max, curve] (S/cm²).
    soma_naps_conductance_range : tuple[float, float, float], optional
        Soma NaPS conductance [min, max, curve] (S/cm²).
    soma_kdrrl_conductance_range : tuple[float, float, float], optional
        Soma KDRRL conductance [min, max, curve] (S/cm²).
    soma_mahp_ca_conductance_range : tuple[float, float, float], optional
        Soma mAHP calcium conductance [min, max, curve] (S/cm²).
    soma_mahp_k_conductance_range : tuple[float, float, float], optional
        Soma mAHP potassium conductance [min, max, curve] (S/cm²).
    soma_mahp_tau_range : tuple[float, float, float], optional
        Soma mAHP time constant [min, max, curve] (ms).
    soma_gh_conductance_range : tuple[float, float, float], optional
        Soma h-current conductance [min, max, curve] (S/cm²).
    dendrite_length_range : tuple[float, float, float], optional
        Dendrite length [min, max, curve] (um).
    dendrite_diameter_range : tuple[float, float, float], optional
        Dendrite diameter [min, max, curve] (um).
    dendrite_passive_conductance_range : tuple[float, float, float], optional
        Dendrite passive conductance [min, max, curve] (S/cm²).
    dendrite_passive_reversal_range : tuple[float, float, float], optional
        Dendrite passive reversal potential [min, max, curve] (mV).
    dendrite_resistance_range : tuple[float, float, float], optional
        Dendrite axial resistance [min, max, curve] (Ω·cm).
    dendrite_capacitance_range : tuple[float, float, float], optional
        Dendrite capacitance [min, max, curve] (uF/cm²).
    dendrite_gh_conductance_range : tuple[float, float, float], optional
        Dendrite h-current conductance [min, max, curve] (S/cm²).
    dendrite_ca_conductance_ranges : tuple[tuple[float, float, float], ...], optional
        L-type Ca conductance ranges for each dendrite (4 tuples).
    dendrite_ca_theta_m_range : tuple[float, float, float], optional
        Ca channel activation threshold [min, max, curve] (mV).
    dendrite_ca_theta_h_range : tuple[float, float, float], optional
        Ca channel inactivation threshold [min, max, curve] (mV).
    """

    def __init__(
        self,
        n: int | None = None,
        recruitment_thresholds__array: RECRUITMENT_THRESHOLDS__ARRAY | None = None,
        config_file: Union[str, Path, None] = None,
        model: str | None = None,
        mode: str | None = None,
        axon_velocities: tuple[float, float] | None = None,
        axon_length: float | None = None,
        gamma: float | None = None,
        cell_index: Optional[int] = None,
        lambda_factor: float | None = None,
        initial_voltage__mV: Union[float, list[float], None] = None,
        spike_threshold__mV: float | None = None,
        # Powers2017 parameters
        # Soma parameters
        soma_length_range: tuple[float, float, float] | None = None,
        soma_diameter_range: tuple[float, float, float] | None = None,
        soma_capacitance_range: tuple[float, float, float] | None = None,
        soma_passive_conductance_range: tuple[float, float, float] | None = None,
        soma_passive_reversal_range: tuple[float, float, float] | None = None,
        soma_na3rp_conductance_range: tuple[float, float, float] | None = None,
        soma_naps_conductance_range: tuple[float, float, float] | None = None,
        soma_kdrrl_conductance_range: tuple[float, float, float] | None = None,
        soma_mahp_ca_conductance_range: tuple[float, float, float] | None = None,
        soma_mahp_k_conductance_range: tuple[float, float, float] | None = None,
        soma_mahp_tau_range: tuple[float, float, float] | None = None,
        soma_gh_conductance_range: tuple[float, float, float] | None = None,
        # Dendrite parameters
        dendrite_length_range: tuple[float, float, float] | None = None,
        dendrite_diameter_range: tuple[float, float, float] | None = None,
        dendrite_passive_conductance_range: tuple[float, float, float] | None = None,
        dendrite_passive_reversal_range: tuple[float, float, float] | None = None,
        dendrite_resistance_range: tuple[float, float, float] | None = None,
        dendrite_capacitance_range: tuple[float, float, float] | None = None,
        dendrite_gh_conductance_range: tuple[float, float, float] | None = None,
        dendrite_ca_conductance_ranges: tuple[tuple[float, float, float], ...] | None = None,
        dendrite_ca_theta_m_range: tuple[float, float, float] | None = None,
        dendrite_ca_theta_h_range: tuple[float, float, float] | None = None,
    ):
        # Load configuration from YAML files with fallback mechanism
        if config_file is None:
            config_file = "alpha_mn_default.yaml"

        try:
            # Always load the default configuration first
            default_config = load_yaml_config("alpha_mn_default.yaml")

            if config_file == "alpha_mn_default.yaml":
                config = default_config
            else:
                # Load the specific config file and merge with defaults
                specific_config = load_yaml_config(config_file)
                config = merge_configs(default_config, specific_config)

        except ImportError:
            # PyYAML not installed
            raise ImportError(
                f"PyYAML is required to load config file '{config_file}'. "
                "Install it with: pip install pyyaml"
            )

        # Helper function to get parameter value (explicit param > config > required)
        def get_param(param_value, config_key):
            if param_value is not None:
                return param_value
            if config_key not in config:
                # Debug information
                available_keys = (
                    list(config.keys()) if isinstance(config, dict) else "config is not a dict"
                )
                raise ValueError(
                    f"Parameter '{config_key}' not found in merged config and not provided explicitly. "
                    f"Available top-level keys: {available_keys}. "
                    f"Config file used: '{config_file}'"
                )
            return config[config_key]

        def get_nested_param(param_value, *config_keys):
            """Get parameter from nested config structure."""
            if param_value is not None:
                return param_value
            value = config
            for key in config_keys:
                if isinstance(value, dict):
                    value = value.get(key, {})
                else:
                    raise ValueError(
                        f"Parameter path {' -> '.join(config_keys)} not found in merged config and not provided explicitly. "
                        f"Failed at key '{key}'. Config file used: '{config_file}'"
                    )
            if value == {} or value is None:
                raise ValueError(
                    f"Parameter path {' -> '.join(config_keys)} not found in merged config and not provided explicitly. "
                    f"Config file used: '{config_file}'"
                )
            return value

        # Set basic parameters
        self.n = n
        self.recruitment_thresholds__array = recruitment_thresholds__array

        if self.recruitment_thresholds__array is not None:
            self.n = len(self.recruitment_thresholds__array)

        if self.n is None and self.recruitment_thresholds__array is None:
            raise ValueError("Either n or recruitment_thresholds__array must be provided.")

        self.model = get_param(model, "model")
        self.mode = get_param(mode, "mode")
        self.axon_velocities = get_param(axon_velocities, "axon_velocities")
        self.axon_length = get_param(axon_length, "axon_length")
        self.gamma = get_param(gamma, "gamma")
        self.cell_index = cell_index
        self.lambda_factor = get_param(lambda_factor, "lambda_factor")

        # Store Powers2017 parameters
        self.soma_length_range = get_nested_param(
            soma_length_range, "powers2017", "soma", "length_range"
        )
        self.soma_diameter_range = get_nested_param(
            soma_diameter_range, "powers2017", "soma", "diameter_range"
        )
        self.soma_capacitance_range = get_nested_param(
            soma_capacitance_range, "powers2017", "soma", "capacitance_range"
        )
        self.soma_passive_conductance_range = get_nested_param(
            soma_passive_conductance_range, "powers2017", "soma", "passive_conductance_range"
        )
        self.soma_passive_reversal_range = get_nested_param(
            soma_passive_reversal_range, "powers2017", "soma", "passive_reversal_range"
        )
        self.soma_na3rp_conductance_range = get_nested_param(
            soma_na3rp_conductance_range, "powers2017", "soma", "na3rp_conductance_range"
        )
        self.soma_naps_conductance_range = get_nested_param(
            soma_naps_conductance_range, "powers2017", "soma", "naps_conductance_range"
        )
        self.soma_kdrrl_conductance_range = get_nested_param(
            soma_kdrrl_conductance_range, "powers2017", "soma", "kdrrl_conductance_range"
        )
        self.soma_mahp_ca_conductance_range = get_nested_param(
            soma_mahp_ca_conductance_range, "powers2017", "soma", "mahp_ca_conductance_range"
        )
        self.soma_mahp_k_conductance_range = get_nested_param(
            soma_mahp_k_conductance_range, "powers2017", "soma", "mahp_k_conductance_range"
        )
        self.soma_mahp_tau_range = get_nested_param(
            soma_mahp_tau_range, "powers2017", "soma", "mahp_tau_range"
        )
        self.soma_gh_conductance_range = get_nested_param(
            soma_gh_conductance_range, "powers2017", "soma", "gh_conductance_range"
        )
        self.dendrite_length_range = get_nested_param(
            dendrite_length_range, "powers2017", "dendrite", "length_range"
        )
        self.dendrite_diameter_range = get_nested_param(
            dendrite_diameter_range, "powers2017", "dendrite", "diameter_range"
        )
        self.dendrite_passive_conductance_range = get_nested_param(
            dendrite_passive_conductance_range,
            "powers2017",
            "dendrite",
            "passive_conductance_range",
        )
        self.dendrite_passive_reversal_range = get_nested_param(
            dendrite_passive_reversal_range, "powers2017", "dendrite", "passive_reversal_range"
        )
        self.dendrite_resistance_range = get_nested_param(
            dendrite_resistance_range, "powers2017", "dendrite", "resistance_range"
        )
        self.dendrite_capacitance_range = get_nested_param(
            dendrite_capacitance_range, "powers2017", "dendrite", "capacitance_range"
        )
        self.dendrite_gh_conductance_range = get_nested_param(
            dendrite_gh_conductance_range, "powers2017", "dendrite", "gh_conductance_range"
        )
        self.dendrite_ca_conductance_ranges = get_nested_param(
            dendrite_ca_conductance_ranges, "powers2017", "dendrite", "ca_conductance_ranges"
        )
        self.dendrite_ca_theta_m_range = get_nested_param(
            dendrite_ca_theta_m_range, "powers2017", "dendrite", "ca_theta_m_range"
        )
        self.dendrite_ca_theta_h_range = get_nested_param(
            dendrite_ca_theta_h_range, "powers2017", "dendrite", "ca_theta_h_range"
        )

        # Store NERLab napp parameters
        self.napp_m_alpha_A = get_nested_param(None, "nerlab", "napp", "m_alpha_A")
        self.napp_m_alpha_v_offset = get_nested_param(None, "nerlab", "napp", "m_alpha_v_offset")
        self.napp_m_alpha_k = get_nested_param(None, "nerlab", "napp", "m_alpha_k")
        self.napp_m_beta_A = get_nested_param(None, "nerlab", "napp", "m_beta_A")
        self.napp_m_beta_v_offset = get_nested_param(None, "nerlab", "napp", "m_beta_v_offset")
        self.napp_m_beta_k = get_nested_param(None, "nerlab", "napp", "m_beta_k")

        self.napp_h_alpha_A = get_nested_param(None, "nerlab", "napp", "h_alpha_A")
        self.napp_h_alpha_v_offset = get_nested_param(None, "nerlab", "napp", "h_alpha_v_offset")
        self.napp_h_alpha_tau = get_nested_param(None, "nerlab", "napp", "h_alpha_tau")
        self.napp_h_beta_A = get_nested_param(None, "nerlab", "napp", "h_beta_A")
        self.napp_h_beta_v_offset = get_nested_param(None, "nerlab", "napp", "h_beta_v_offset")
        self.napp_h_beta_k = get_nested_param(None, "nerlab", "napp", "h_beta_k")

        self.napp_p_alpha_A = get_nested_param(None, "nerlab", "napp", "p_alpha_A")
        self.napp_p_alpha_v_offset = get_nested_param(None, "nerlab", "napp", "p_alpha_v_offset")
        self.napp_p_alpha_k = get_nested_param(None, "nerlab", "napp", "p_alpha_k")
        self.napp_p_beta_A = get_nested_param(None, "nerlab", "napp", "p_beta_A")
        self.napp_p_beta_v_offset = get_nested_param(None, "nerlab", "napp", "p_beta_v_offset")
        self.napp_p_beta_k = get_nested_param(None, "nerlab", "napp", "p_beta_k")

        self.napp_n_alpha_A = get_nested_param(None, "nerlab", "napp", "n_alpha_A")
        self.napp_n_alpha_v_offset = get_nested_param(None, "nerlab", "napp", "n_alpha_v_offset")
        self.napp_n_alpha_k = get_nested_param(None, "nerlab", "napp", "n_alpha_k")
        self.napp_n_beta_A = get_nested_param(None, "nerlab", "napp", "n_beta_A")
        self.napp_n_beta_v_offset = get_nested_param(None, "nerlab", "napp", "n_beta_v_offset")
        self.napp_n_beta_tau = get_nested_param(None, "nerlab", "napp", "n_beta_tau")

        self.napp_r_alpha_A = get_nested_param(None, "nerlab", "napp", "r_alpha_A")
        self.napp_r_alpha_v_offset = get_nested_param(None, "nerlab", "napp", "r_alpha_v_offset")
        self.napp_r_alpha_k = get_nested_param(None, "nerlab", "napp", "r_alpha_k")

        # Store NERLab soma parameters
        self.nerlab_soma_diameter_range = get_nested_param(None, "nerlab", "soma", "diameter_range")
        self.nerlab_soma_gnabar_range = get_nested_param(None, "nerlab", "soma", "gnabar_range")
        self.nerlab_soma_gnapbar_range = get_nested_param(None, "nerlab", "soma", "gnapbar_range")
        self.nerlab_soma_gkfbar_range = get_nested_param(None, "nerlab", "soma", "gkfbar_range")
        self.nerlab_soma_gksbar_range = get_nested_param(None, "nerlab", "soma", "gksbar_range")
        self.nerlab_soma_mact_range = get_nested_param(None, "nerlab", "soma", "mact_range")
        self.nerlab_soma_rinact_range = get_nested_param(None, "nerlab", "soma", "rinact_range")
        self.nerlab_soma_gls_range = get_nested_param(None, "nerlab", "soma", "gls_range")
        self.nerlab_soma_ena = get_nested_param(None, "nerlab", "soma", "ena")
        self.nerlab_soma_ek = get_nested_param(None, "nerlab", "soma", "ek")
        self.nerlab_soma_el_napp = get_nested_param(None, "nerlab", "soma", "el_napp")
        self.nerlab_soma_vtraub_napp = get_nested_param(None, "nerlab", "soma", "vtraub_napp")
        self.nerlab_soma_Ra = get_nested_param(None, "nerlab", "soma", "Ra")
        self.nerlab_soma_cm = get_nested_param(None, "nerlab", "soma", "cm")

        # Store NERLab dendrite parameters
        self.nerlab_dendrite_diameter_range = get_nested_param(
            None, "nerlab", "dendrite", "diameter_range"
        )
        self.nerlab_dendrite_length_range = get_nested_param(
            None, "nerlab", "dendrite", "length_range"
        )
        self.nerlab_dendrite_gcaLbar_range = get_nested_param(
            None, "nerlab", "dendrite", "gcaLbar_range"
        )
        self.nerlab_dendrite_vtraub_caL_range = get_nested_param(
            None, "nerlab", "dendrite", "vtraub_caL_range"
        )
        self.nerlab_dendrite_ltau_caL_range = get_nested_param(
            None, "nerlab", "dendrite", "ltau_caL_range"
        )
        self.nerlab_dendrite_gl_caL_range = get_nested_param(
            None, "nerlab", "dendrite", "gl_caL_range"
        )
        self.nerlab_dendrite_Ra = get_nested_param(None, "nerlab", "dendrite", "Ra")
        self.nerlab_dendrite_cm = get_nested_param(None, "nerlab", "dendrite", "cm")
        self.nerlab_dendrite_ecaL = get_nested_param(None, "nerlab", "dendrite", "ecaL")
        self.nerlab_dendrite_el_caL = get_nested_param(None, "nerlab", "dendrite", "el_caL")

        if self.model == "NERLab":
            _cells = self._create_nerlab_cells()
        elif self.model == "Powers2017":
            _cells = self._create_powers2017_cells()
        else:
            raise ValueError("Could not find the specific model for alpha MNs.")

        # Get initial voltage and spike threshold from config if not provided
        _initial_voltage = get_param(initial_voltage__mV, "initial_voltage__mV")
        _spike_threshold = get_param(spike_threshold__mV, "spike_threshold__mV")

        super().__init__(
            cells=_cells,
            initial_voltage__mV=_initial_voltage,
            spike_threshold__mV=_spike_threshold,
        )

    def _create_nerlab_cells(self) -> list:
        """Create motor neurons using the NERLab model."""

        def special_interp(x, y, n, curv=None, negative=None):
            if negative:
                # Reverse the interpolation direction
                return self.recruitment_thresholds__array * (x - y) + y
            else:
                # Normal direction
                return self.recruitment_thresholds__array * (y - x) + x

        if self.recruitment_thresholds__array is None:
            interpF = _exp_interp
        else:
            interpF = special_interp

        self.t = self.n

        # Soma parameters (using parameters from YAML config)
        Diam_soma = interpF(
            self.nerlab_soma_diameter_range[0],
            self.nerlab_soma_diameter_range[1],
            self.t,
            curv=1.0 / self.nerlab_soma_diameter_range[2],
            negative=self.nerlab_soma_diameter_range[3],
        )
        Gnabar = interpF(
            self.nerlab_soma_gnabar_range[0],
            self.nerlab_soma_gnabar_range[1],
            self.t,
            curv=1 / self.nerlab_soma_gnabar_range[2],
            negative=self.nerlab_soma_gnabar_range[3],
        )
        Gnapbar = interpF(
            self.nerlab_soma_gnapbar_range[0],
            self.nerlab_soma_gnapbar_range[1],
            self.t,
            curv=1 / self.nerlab_soma_gnapbar_range[2],
            negative=self.nerlab_soma_gnapbar_range[3],
        )
        Gkfbar = interpF(
            self.nerlab_soma_gkfbar_range[0],
            self.nerlab_soma_gkfbar_range[1],
            self.t,
            curv=1 / self.nerlab_soma_gkfbar_range[2],
            negative=self.nerlab_soma_gkfbar_range[3],
        )
        Gksbar = interpF(
            self.nerlab_soma_gksbar_range[0],
            self.nerlab_soma_gksbar_range[1],
            self.t,
            curv=1.0 / self.nerlab_soma_gksbar_range[2],
            negative=self.nerlab_soma_gksbar_range[3],
        )
        Mact = interpF(
            self.nerlab_soma_mact_range[0],
            self.nerlab_soma_mact_range[1],
            self.t,
            curv=1 / self.nerlab_soma_mact_range[2],
            negative=self.nerlab_soma_mact_range[3],
        )
        Rinact = interpF(
            self.nerlab_soma_rinact_range[0],
            self.nerlab_soma_rinact_range[1],
            self.t,
            curv=1 / self.nerlab_soma_rinact_range[2],
            negative=self.nerlab_soma_rinact_range[3],
        )
        Gls = interpF(
            self.nerlab_soma_gls_range[0],
            self.nerlab_soma_gls_range[1],
            self.t,
            curv=1 / self.nerlab_soma_gls_range[2],
            negative=self.nerlab_soma_gls_range[3],
        )

        # Dendrite parameters
        Diam_dend = interpF(
            self.nerlab_dendrite_diameter_range[0],
            self.nerlab_dendrite_diameter_range[1],
            self.t,
            curv=1.0 / self.nerlab_dendrite_diameter_range[2],
            negative=self.nerlab_dendrite_diameter_range[3],
        )
        L_dend = interpF(
            self.nerlab_dendrite_length_range[0],
            self.nerlab_dendrite_length_range[1],
            self.t,
            curv=1.0 / self.nerlab_dendrite_length_range[2],
            negative=self.nerlab_dendrite_length_range[3],
        )
        GcaLbar = interpF(
            self.nerlab_dendrite_gcaLbar_range[0],
            self.nerlab_dendrite_gcaLbar_range[1],
            self.t,
            curv=1 / self.nerlab_dendrite_gcaLbar_range[2],
            negative=self.nerlab_dendrite_gcaLbar_range[3],
        )
        Vtraub_caL = interpF(
            self.nerlab_dendrite_vtraub_caL_range[0],
            self.nerlab_dendrite_vtraub_caL_range[1],
            self.t,
            curv=1 / self.nerlab_dendrite_vtraub_caL_range[2],
            negative=self.nerlab_dendrite_vtraub_caL_range[3],
        )
        LTAU_caL = interpF(
            self.nerlab_dendrite_ltau_caL_range[0],
            self.nerlab_dendrite_ltau_caL_range[1],
            self.t,
            curv=1 / self.nerlab_dendrite_ltau_caL_range[2],
            negative=self.nerlab_dendrite_ltau_caL_range[3],
        )
        Gl_caL = interpF(
            self.nerlab_dendrite_gl_caL_range[0],
            self.nerlab_dendrite_gl_caL_range[1],
            self.t,
            curv=1 / self.nerlab_dendrite_gl_caL_range[2],
            negative=self.nerlab_dendrite_gl_caL_range[3],
        )

        vcon = np.linspace(self.axon_velocities[0], self.axon_velocities[1], self.n)

        # Determine cell creation range
        if self.cell_index is not None:
            init, end = self.cell_index, self.cell_index + 1
        else:
            init, end = 0, self.n

        _cells = []
        for i in range(init, end):
            cell = cells.AlphaMN(
                segments__count=1,
                mode=self.mode,
                dendrites__count=1,
                model=self.model,
                class__ID=self.cell_index,
                pool__ID=i,
            )
            # Convert to quantities for create_axon
            import quantities as pq

            cell.create_axon(
                length__m=self.axon_length * pq.m,
                conduction_velocity__m_per_s=vcon[i] * pq.m / pq.s,
            )

            # Soma biophysical parameters
            cell.soma.L = Diam_soma[i]
            cell.soma.diam = Diam_soma[i]
            cell.soma.ena = self.nerlab_soma_ena
            cell.soma.ek = self.nerlab_soma_ek
            cell.soma.el_napp = self.nerlab_soma_el_napp
            cell.soma.vtraub_napp = self.nerlab_soma_vtraub_napp
            cell.soma.Ra = self.nerlab_soma_Ra
            cell.soma.cm = self.nerlab_soma_cm
            cell.soma.gl_napp = Gls[i]
            cell.soma.gnabar_napp = Gnabar[i]
            cell.soma.gnapbar_napp = Gnapbar[i]
            cell.soma.gkfbar_napp = Gkfbar[i]
            cell.soma.gksbar_napp = Gksbar[i]
            cell.soma.mact_napp = Mact[i]
            cell.soma.rinact_napp = Rinact[i]

            # Set napp alpha and beta parameters from config
            # These parameters control the kinetics of sodium and potassium channels
            cell.soma.m_alpha_A_napp = self.napp_m_alpha_A
            cell.soma.m_alpha_v_offset_napp = self.napp_m_alpha_v_offset
            cell.soma.m_alpha_k_napp = self.napp_m_alpha_k
            cell.soma.m_beta_A_napp = self.napp_m_beta_A
            cell.soma.m_beta_v_offset_napp = self.napp_m_beta_v_offset
            cell.soma.m_beta_k_napp = self.napp_m_beta_k

            cell.soma.h_alpha_A_napp = self.napp_h_alpha_A
            cell.soma.h_alpha_v_offset_napp = self.napp_h_alpha_v_offset
            cell.soma.h_alpha_tau_napp = self.napp_h_alpha_tau
            cell.soma.h_beta_A_napp = self.napp_h_beta_A
            cell.soma.h_beta_v_offset_napp = self.napp_h_beta_v_offset
            cell.soma.h_beta_k_napp = self.napp_h_beta_k

            cell.soma.p_alpha_A_napp = self.napp_p_alpha_A
            cell.soma.p_alpha_v_offset_napp = self.napp_p_alpha_v_offset
            cell.soma.p_alpha_k_napp = self.napp_p_alpha_k
            cell.soma.p_beta_A_napp = self.napp_p_beta_A
            cell.soma.p_beta_v_offset_napp = self.napp_p_beta_v_offset
            cell.soma.p_beta_k_napp = self.napp_p_beta_k

            cell.soma.n_alpha_A_napp = self.napp_n_alpha_A
            cell.soma.n_alpha_v_offset_napp = self.napp_n_alpha_v_offset
            cell.soma.n_alpha_k_napp = self.napp_n_alpha_k
            cell.soma.n_beta_A_napp = self.napp_n_beta_A
            cell.soma.n_beta_v_offset_napp = self.napp_n_beta_v_offset
            cell.soma.n_beta_tau_napp = self.napp_n_beta_tau

            cell.soma.r_alpha_A_napp = self.napp_r_alpha_A
            cell.soma.r_alpha_v_offset_napp = self.napp_r_alpha_v_offset
            cell.soma.r_alpha_k_napp = self.napp_r_alpha_k

            # Dendrite parameters
            cell.dend[0].Ra = self.nerlab_dendrite_Ra
            cell.dend[0].cm = self.nerlab_dendrite_cm
            cell.dend[0].L = L_dend[i]
            cell.dend[0].diam = Diam_dend[i]
            cell.dend[0].ecaL = self.nerlab_dendrite_ecaL
            cell.dend[0].gama_caL = self.gamma
            cell.dend[0].gcaLbar_caL = GcaLbar[i]
            cell.dend[0].vtraub_caL = Vtraub_caL[i]
            cell.dend[0].Ltau_caL = LTAU_caL[i]
            cell.dend[0].gl_caL = Gl_caL[i]
            cell.dend[0].el_caL = self.nerlab_dendrite_el_caL
            _cells.append(cell)

        return _cells

    def _create_powers2017_cells(self) -> list:
        """Create motor neurons using the Powers2017 model."""
        if self.recruitment_thresholds__array is None:
            interpF = lambda x, y, z: _exp_interp(first=x, last=y, n=self.n, curv=z)
        else:
            interpF = lambda x, y, _: self.recruitment_thresholds__array * (y - x) + x

        # Geometry parameters
        sL = interpF(*self.soma_length_range)
        sdiam = interpF(*self.soma_diameter_range)
        scm = interpF(*self.soma_capacitance_range)

        # Biophysics parameters
        sg_pas = interpF(*self.soma_passive_conductance_range)
        se_pas = interpF(*self.soma_passive_reversal_range)
        sgbar_na3rp = interpF(*self.soma_na3rp_conductance_range)
        sgbar_naps = interpF(*self.soma_naps_conductance_range)
        sgMax_kdrRL = interpF(*self.soma_kdrrl_conductance_range)
        sgcamax_mAHP = interpF(*self.soma_mahp_ca_conductance_range)
        sgkcamax_mAHP = interpF(*self.soma_mahp_k_conductance_range)
        staur_mAHP = interpF(*self.soma_mahp_tau_range)
        sghbar_gh = interpF(*self.soma_gh_conductance_range)

        # Dendrite parameters
        dL = interpF(*self.dendrite_length_range)
        ddiam = interpF(*self.dendrite_diameter_range)
        dg_pas = interpF(*self.dendrite_passive_conductance_range)
        de_pas = interpF(*self.dendrite_passive_reversal_range)
        dRa = interpF(*self.dendrite_resistance_range)
        dcm = interpF(*self.dendrite_capacitance_range)
        dghbar_gh = interpF(*self.dendrite_gh_conductance_range)

        # L-type calcium channels for each dendrite
        d_ca_conductances = [interpF(*ca_range) for ca_range in self.dendrite_ca_conductance_ranges]
        dtheta_m_L_Ca_inact = interpF(*self.dendrite_ca_theta_m_range)
        dtheta_h_L_Ca_inact = interpF(*self.dendrite_ca_theta_h_range)

        vcon = np.linspace(*self.axon_velocities, self.n)

        if self.cell_index is not None:
            init, end = self.cell_index, self.cell_index + 1
        else:
            init, end = 0, self.n

        _cells = []
        for i in range(init, end):
            cell = cells.AlphaMN(
                segments__count=1,
                mode=self.mode,
                dendrites__count=4,
                model=self.model,
                class__ID=self.cell_index,
                pool__ID=i,
            )

            # Set soma parameters
            cell.soma.L = sL[i]
            cell.soma.diam = sdiam[i]
            # Convert to quantities for create_axon
            import quantities as pq

            cell.create_axon(
                length__m=self.axon_length * pq.m,
                conduction_velocity__m_per_s=vcon[i] * pq.m / pq.s,
            )
            cell.soma.g_pas = sg_pas[i]
            cell.soma.e_pas = se_pas[i]
            cell.soma.cm = scm[i]
            cell.soma.gbar_na3rp = sgbar_na3rp[i]
            cell.soma.gbar_naps = sgbar_naps[i] * self.lambda_factor
            cell.soma.gMax_kdrRL = sgMax_kdrRL[i]
            cell.soma.gcamax_mAHP = sgcamax_mAHP[i]
            cell.soma.gkcamax_mAHP = sgkcamax_mAHP[i]
            cell.soma.taur_mAHP = staur_mAHP[i]
            cell.soma.ghbar_gh = sghbar_gh[i]

            # Set dendrite parameters
            for j, d in enumerate(cell.dend):
                d.L = dL[i]
                d.diam = ddiam[i]
                d.g_pas = dg_pas[i]
                d.e_pas = de_pas[i]
                d.Ra = dRa[i]
                d.cm = dcm[i]
                d.ghbar_gh = dghbar_gh[i]

                if self.mode == "active":
                    d.gcabar_L_Ca_inact = d_ca_conductances[j][i] * self.gamma
                    d.theta_m_L_Ca_inact = dtheta_m_L_Ca_inact[i]
                    d.theta_h_L_Ca_inact = dtheta_h_L_Ca_inact[i]

            _cells.append(cell)

        return _cells
