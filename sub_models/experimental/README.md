# Experimental Sub-Models

️ **WARNING**: This directory contains experimental modules that are under active development. They may have incomplete implementations, unstable APIs, or missing tests.

## Status Legend

| Status | Meaning |
|--------|---------|
|  Research | Early research phase, not ready for use |
|  Development | Active development, API may change |
|  Testing | Feature complete, needs more testing |
|  Stable | Production ready |

## Module Status

| Module | Status | Description |
|--------|--------|-------------|
| `dual_process.py` |  Development | Dual-process thinking (System 1/2) |
| `liquid.py` |  Research | Liquid neural networks |
| `snns_LiCell.py` |  Research | Spiking neural networks with LiCell |
| `spike_ssm.py` |  Research | Spike-based state space models |
| `meta_bamdp.py` |  Research | Meta-learning with BAMDPs |
| `adaptive_vq_submodel.py` |  Development | Adaptive vector quantization |

## Usage Guidelines

1. **Do not use in production** without explicit approval
2. **Expect breaking changes** in APIs
3. **Report issues** to help improve these modules
4. **Contribute tests** if using experimentally

## Dependencies

Some modules require additional dependencies not in the main requirements:
- `liquid.py`: Requires `pytorch` with specific versions
- `snns_LiCell.py`: Requires spiking neural network libraries
- `spike_ssm.py`: Requires JAX with specific configurations

## Migration Path

When a module becomes stable:
1. It will be moved to `sub_models/` root or appropriate subdirectory
2. A deprecation notice will be added here
3. Import paths will be aliased for backwards compatibility

## Contributing

To contribute to experimental modules:
1. Create comprehensive tests
2. Document the API
3. Add usage examples
4. Update this README with status changes
