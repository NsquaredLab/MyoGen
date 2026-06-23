# Top-level API (`myogen`)

## Reproducibility

Module-level state:

- **`myogen.RANDOM_GENERATOR`** — the global [`numpy.random.Generator`][numpy.random.Generator] used across MyoGen simulations, initialized with the default seed. Change it with `set_random_seed`.
- **`myogen.SEED`** — the current random seed value (default `180319`).

::: myogen.set_random_seed
::: myogen.get_random_generator
::: myogen.get_random_seed
::: myogen.derive_subseed
::: myogen.load_nmodl_mechanisms
::: myogen.get_mechanism_parameters
::: myogen.validate_mechanism_parameter
::: myogen.set_mechanism_param
