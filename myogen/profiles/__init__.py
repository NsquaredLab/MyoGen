"""Named simulation profiles ("cards"): every validated setting of a simulation in one place.

``load_profile()`` gives the ``fdi_semg`` card (FDI surface EMG); ``load_profile("fdi_semg.published")`` its
variant with each component as its source paper describes it. The card builds the library objects and runs the
calibrated drive; keywords passed to a builder override the card, and a note is printed when an overridden value
is one the card's calibration or fit depends on.
"""
from myogen.profiles._card import CardField, Profile, ProfileError, available_profiles, load_profile

__all__ = ["CardField", "Profile", "ProfileError", "available_profiles", "load_profile"]
