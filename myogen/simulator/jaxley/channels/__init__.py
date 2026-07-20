"""Custom ion channels for Jaxley simulations."""

from myogen.simulator.jaxley.channels.constant import Constant
from myogen.simulator.jaxley.channels.motor_neuron_channels import (
    Na3rp,
    Naps,
    KdrRL,
    MAHP,
    Gh,
    LCaInact,
    LeakChannel,
    get_motor_neuron_channels_soma,
    get_motor_neuron_channels_dendrite,
    safe_exp,
    vtrap,
    trap0,
)
from myogen.simulator.jaxley.channels.nerlab_channels import (
    napp,
    caL,
)

__all__ = [
    # Utility
    "Constant",
    # Powers2017 motor-neuron channels
    "Na3rp",
    "Naps",
    "KdrRL",
    "MAHP",
    "Gh",
    "LCaInact",
    "LeakChannel",
    # NERLab motor-neuron channels (1952 HH voltage convention; V_rest ≈ 0 mV)
    "napp",
    "caL",
    # Helper functions
    "get_motor_neuron_channels_soma",
    "get_motor_neuron_channels_dendrite",
    "safe_exp",
    "vtrap",
    "trap0",
]
