# Perfiles de producción (TPU)

Se han creado perfiles fusionados:
- `fusion_base.toml`: base común con parámetros del modelo, memoria, router, módulos, servicios, etc.
- `fusion_production.toml`: perfil de producción (TPU v6e / MoE) que complementa la base.

Recomendado: en código, apuntar a `fusion_production.toml` o a `fusion_base.toml` según necesidad.

Archivos anteriores pueden mantenerse por compatibilidad, pero se sugiere migrar.
