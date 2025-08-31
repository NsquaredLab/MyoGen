"""
Type validation decorators for MyoGen.

This module provides decorators for runtime type checking and validation.
"""

from beartype import BeartypeConf, beartype

# See https://beartype.readthedocs.io/en/latest/api_decor/#beartype.BeartypeConf.is_pep484_tower
beartowertype = beartype(conf=BeartypeConf(is_pep484_tower=True))

__all__ = ["beartowertype"]
