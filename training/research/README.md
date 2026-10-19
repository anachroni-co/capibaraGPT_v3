# training/research/

Codigo de INVESTIGACION experimental. No forma parte del camino canonico
de entrenamiento y no debe importarse desde codigo de produccion.

| Modulo | Estado |
|---|---|
| `consensus/` | Meta-consensus multi-modelo (~13k LOC). Mocks pendientes: BACKLOG ISSUE-001/002 |
| `federated_consensus/` | Consenso federado distribuido |
| `strategies/` | Estrategias de entrenamiento alternativas |
| `hierarchical_strategy/` | Estrategia jerarquica |

Eliminados en la limpieza de 2026-07 (recuperables via git history):
data_lineage (audit blockchain, import roto), cython_kernels (stubs sin
compilar), unified_trainer (non-canonical), y los routers/bridges huerfanos
(hybrid_expert_router, btx_training_system, config_manager,
monitoring_dashboard, moe_hierarchical_router, core_training_bridge,
cascade_training_integration, data_preprocessing_integration).
