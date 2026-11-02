"""Regression test for Hill muscle activation normalisation."""

import pytest
import quantities as pq

pytest.importorskip("neuron")
pytest.importorskip("myogen.simulator.neuron._cython._hill")

from myogen.simulator.neuron.muscle import HillModel


def test_hill_activation_at_full_saturation():
    parameters = HillModel.create_default_muscle_parameters("FDI")
    dt_ms = 0.05
    model = HillModel(
        simulation_time__ms=3000.0 * pq.ms,
        time_step__ms=dt_ms * pq.ms,
        muscle_parameters=parameters,
        n_motor_units_type1=10,
        n_motor_units_type2=10,
        initial_joint_angle__deg=0.0,
        initial_muscle_length__L0=1.0,
    )
    # 100 Hz saturates every FDI unit; 3 s exceeds 30 longest twitch times.
    steps_per_spike = round(1000.0 / parameters["lsatf"] / dt_ms)
    for step in range(len(model.time_vector) - 1):
        if step % steps_per_spike == 0:
            for unit in range(20):
                model.add_spike(unit)
        model.integrate(0.0)

    peak_activation = (model.type1_activation + model.type2_activation).max()
    assert peak_activation <= 1.002
    assert peak_activation > 0.95
