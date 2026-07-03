# training/

## Camino canonico de entrenamiento (produccion)

| Modulo | Que hace |
|---|---|
| `byte_level_training.py` | Tokenizador y dataloader byte-level (CPU) — usado por todos los scripts de entrenamiento CPU |
| `data_loader.py` | ShardDataLoader para corpus fragmentados |
| `lmtp_flax_trainer.py` | Entrenador L-MTP (JAX/Flax) sobre models/lmtp_flax |
| `tpu/` | Trainer TPU v6e robusto (preemptibles, checkpoints) — `python -m training.tpu.tpu_v6e_trainer --config config/configs_toml/production/training.toml` |
| `data_capture/` | Captura automatica de pares de entrenamiento desde inferencia (feature flag) |
| `data_preprocessing/` | Limpieza y filtrado de corpus |
| `optimizations/` | Optimizaciones TPU compartidas |
| `safety/` | Filtros de seguridad en entrenamiento |
| `jax_utils.py` | Helpers JAX |

## Investigacion (NO es el camino de entrenamiento)

Todo lo experimental vive en `research/` — ver `research/README.md`.
No importar `training.research.*` desde codigo de produccion.
