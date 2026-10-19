# Distributed Module (Core)

This subpackage provides **minimal, working** mesh configuration helpers.
It focuses on JAX sharding setup and avoids fake “enterprise” features.

## Available API

- `TPUDistributionConfig`
- `create_mesh_config(...)`
- `setup_mesh(...)`
- `PartitionSpec` exposed as `P`

## Example

```python
from capibara.core.distributed import TPUDistributionConfig

cfg = TPUDistributionConfig(
    mesh_shape=(1,),
    mesh_axis_names=("data",),
    device_type="cpu",
)
system = cfg.setup_distributed_training()
print(system.mesh_shape)
```

## Notes

- The module **raises clear errors** if the requested device type or mesh
  shape is not available.
- Advanced auto‑tuning, monitoring, or fault‑tolerance features are **not**
  implemented here yet.
