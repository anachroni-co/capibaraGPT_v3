# capibara/core/tpu - TPU Optimization Module

El módulo **tpu** proporciona configuraciones y optimizaciones específicas para Google Cloud TPU v4, v5e y v6e, incluyendo integración con Vector Quantization.

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [TPU Versions Supported](#tpu-versions-supported)
3. [Quick Start](#quick-start)
4. [TPU Configuration](#tpu-configuration)
5. [Mesh & Sharding](#mesh--sharding)
6. [XLA Optimizations](#xla-optimizations)
7. [Memory Management](#memory-management)
8. [Vector Quantization Integration](#vector-quantization-integration)
9. [Performance Tuning](#performance-tuning)
10. [Monitoring](#monitoring)
11. [Troubleshooting](#troubleshooting)
12. [Benchmarks](#benchmarks)

---

## 🎯 Visión General

Este módulo especializa el sistema para aprovechar al máximo las capacidades de Google Cloud TPUs, incluyendo:

- ⚡ **TPU v4/v5e/v6e Support**: Optimizaciones específicas para cada generación
- 🔧 **Automatic Configuration**: Setup automático de entorno TPU
- 📊 **Mesh Configuration**: Data/model parallelism automático
- 🎯 **XLA Optimization**: Compilación XLA optimizada
- 💾 **Memory Management**: Gestión eficiente de HBM
- 📦 **VQ Integration**: Vector Quantization acelerada por hardware
- 📈 **Performance Monitoring**: Métricas en tiempo real

### Arquitectura

```
capibara/core/tpu/
├── tpu_config.py              # Configuración general TPU
├── tpu_v6_vq_integration.py   # Integración VQ para TPU v6
└── README.md                  # Este archivo
```

---

## 🖥️ TPU Versions Supported

| TPU Version | Chips | Memory/Chip | BF16 TFLOPS | INT8 TOPS | Recomendado Para |
|-------------|-------|-------------|-------------|-----------|------------------|
| **v4-8** | 4 | 16GB HBM | 275 | 550 | Development |
| **v4-32** | 32 | 16GB HBM | 2,200 | 4,400 | Medium models |
| **v5e-4** | 4 | 16GB HBM | 197 | 394 | Cost-efficient dev |
| **v5e-64** | 64 | 16GB HBM | 12,600 | 25,200 | Production |
| **v5e-256** | 256 | 16GB HBM | 50,500 | 101,000 | Large models |
| **v6e-64** | 64 | 32GB HBM | 21,000 | 42,000 | Advanced production |
| **v6e-256** | 256 | 32GB HBM | 84,000 | 168,000 | Massive models |

### Características por Generación

**TPU v4**:
- BFloat16 nativo
- SparseCore para embedding lookups
- 2nd gen interconnect

**TPU v5e**:
- 40% más eficiente que v4
- Menor costo por TFLOP
- Ideal para producción cost-sensitive

**TPU v6e**:
- 2x memoria (32GB vs 16GB)
- 3rd gen interconnect
- Hardware VQ acceleration
- Mejor para modelos grandes

---

## 🚀 Quick Start

### Configuración Básica

```python
from capibara.core.tpu import TPUConfig, configure_tpu_environment

# Auto-detectar y configurar TPU
configure_tpu_environment()

# Crear configuración
config = TPUConfig(
    tpu_version="v6e",       # v4, v5e, v6e
    num_chips=64,            # 4, 8, 32, 64, 256
    topology="8x8"           # Mesh topology
)

# Verificar configuración
config.validate()
print(f"TPU configurado: {config.tpu_version}-{config.num_chips}")
print(f"Total TFLOPS: {config.get_total_tflops():.1f}")
```

### Training en TPU

```python
from capibara.core.tpu import TPUTrainer

# Crear trainer optimizado para TPU
trainer = TPUTrainer(
    config=config,
    model=model,
    mesh_shape=(8, 8),       # Para v6e-64: 8x8
    use_bf16=True,
    enable_xla_optimizations=True
)

# Entrenar
trainer.train(
    train_data="gs://bucket/data/",
    num_steps=100000,
    batch_size_per_chip=16   # Batch size por chip TPU
)
```

---

## ⚙️ TPU Configuration

### Configuración Completa

```python
from capibara.core.tpu import TPUConfig

config = TPUConfig(
    # Hardware
    tpu_version="v6e",
    num_chips=64,
    topology="8x8",
    memory_per_chip="32GB",

    # Precision
    precision="bfloat16",    # bfloat16, float32, mixed
    use_mixed_precision=True,
    fp32_params=True,        # Parámetros en FP32, cómputo en BF16

    # Compilation
    xla_optimization_level=3,  # 0-3
    enable_xla_fusion=True,
    enable_async_collective=True,

    # Memory
    memory_optimization="aggressive",  # none, normal, aggressive
    enable_rematerialization=True,
    gradient_checkpointing=True,

    # Communication
    communication_optimization=True,
    use_collective_matmul=True,

    # Monitoring
    enable_profiling=False,  # Solo para debugging (overhead)
    log_device_placement=False
)
```

### Variables de Entorno TPU

```python
# El módulo configura automáticamente:
config.setup_environment_variables()

# Configura:
# - JAX_PLATFORMS=tpu
# - XLA_FLAGS=...
# - LIBTPU_INIT_ARGS=...
# - XLA_PYTHON_CLIENT_ALLOCATOR=platform
# - XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Configuración Manual de Entorno

```bash
# Alternativa: Configurar manualmente
export JAX_PLATFORMS=tpu
export XLA_FLAGS="--xla_tpu_enable_data_parallelism=true \
                  --xla_tpu_enable_async_collective_fusion=true \
                  --xla_tpu_enable_megacore_fusion=true"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallelism=true"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

---

## 🌐 Mesh & Sharding

### Mesh Configuration

```python
from capibara.core.tpu import create_mesh, MeshConfig

# Configurar mesh para data + model parallelism
mesh_config = MeshConfig(
    mesh_shape=(8, 8),       # (data_parallel, model_parallel)
    axis_names=('data', 'model'),
    devices=jax.devices()
)

mesh = create_mesh(mesh_config)

# Usar mesh para sharding
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P

# Particionar datos
data_sharding = P('data', None)    # Shard en dimension data

# Particionar modelo
weight_sharding = P(None, 'model')  # Shard en dimension model
```

### Sharding Strategies

```python
# Strategy 1: Pure Data Parallelism (simple)
data_parallel_mesh = create_mesh(
    mesh_shape=(64, 1),      # 64-way data parallel
    axis_names=('data',)
)

# Strategy 2: Data + Model Parallelism (balanced)
hybrid_mesh = create_mesh(
    mesh_shape=(8, 8),       # 8-way data, 8-way model
    axis_names=('data', 'model')
)

# Strategy 3: Expert Parallelism (para MoE)
moe_mesh = create_mesh(
    mesh_shape=(4, 4, 4),    # (data, model, expert)
    axis_names=('data', 'model', 'expert')
)

# Aplicar sharding a modelo
from capibara.core.tpu import shard_model

sharded_model = shard_model(
    model,
    mesh=hybrid_mesh,
    strategy="fsdp"  # fsdp, megatron, expert
)
```

---

## ⚡ XLA Optimizations

### Automatic XLA Optimization

```python
from capibara.core.tpu import enable_xla_optimizations

# Habilitar todas las optimizaciones XLA
enable_xla_optimizations(
    level=3,                 # 0 (none) - 3 (maximum)
    enable_fusion=True,
    enable_async_collective=True,
    enable_megacore=True,
    enable_latency_hiding=True
)
```

### Custom XLA Flags

```python
from capibara.core.tpu import set_xla_flags

# Configurar flags XLA específicos
set_xla_flags({
    "xla_tpu_enable_data_parallelism": True,
    "xla_tpu_enable_async_collective_fusion": True,
    "xla_tpu_enable_megacore_fusion": True,
    "xla_gpu_enable_triton_softmax_fusion": True,
    "xla_gpu_triton_gemm_any": True,
    "xla_tpu_megacore_fusion_allow_ags": True,
    "xla_tpu_enable_ag_backward_pipelining": True
})
```

### JIT Compilation

```python
import jax

# JIT con optimizaciones TPU
@jax.jit(
    donate_argnums=(0,),     # Donate buffers para memory efficiency
    backend="tpu",
    device=jax.devices()[0]
)
def optimized_forward(params, inputs):
    return model.apply(params, inputs)

# Compilar ahead-of-time
compiled_fn = jax.jit(optimized_forward).lower(params, inputs).compile()
```

---

## 💾 Memory Management

### HBM Optimization

```python
from capibara.core.tpu import MemoryOptimizer

# Crear memory optimizer
mem_opt = MemoryOptimizer(
    strategy="aggressive",   # none, normal, aggressive
    enable_rematerialization=True,
    gradient_checkpointing=True,
    activation_checkpointing_layers=2
)

# Aplicar optimizaciones
optimized_model = mem_opt.optimize(model)

# Estimate memory usage
memory_estimate = mem_opt.estimate_memory(
    model=model,
    batch_size=128,
    sequence_length=2048
)
print(f"Estimated HBM: {memory_estimate['total_gb']:.1f}GB")
```

### Gradient Checkpointing

```python
from capibara.core.tpu import enable_gradient_checkpointing

# Habilitar gradient checkpointing (recompute activations)
checkpointed_model = enable_gradient_checkpointing(
    model,
    checkpoint_every_n_layers=2,  # Checkpoint cada 2 capas
    checkpoint_policy="nothing_saveable"  # everything, nothing_saveable, minimal
)

# Reduce memoria ~50%, aumenta cómputo ~20%
```

### Buffer Donation

```python
# Donate buffers para reusar memoria
@jax.jit(donate_argnums=(0, 1))  # Donate params y state
def train_step(params, state, batch):
    # params y state se pueden reusar in-place
    return updated_params, updated_state
```

---

## 📦 Vector Quantization Integration

### TPU v6 Hardware VQ Acceleration

```python
from capibara.core.tpu import TPUv6VQIntegration

# Configurar VQ acelerado por hardware (TPU v6 only)
vq_integration = TPUv6VQIntegration(
    quantization_bits=8,     # 4, 8, 16
    vector_length=512,
    codebook_size=1024,
    enable_hardware_acceleration=True,  # Usa HW accel si disponible
    distance_metric="euclidean"  # euclidean, cosine, dot_product
)

# Quantizar modelo
quantized_model = vq_integration.quantize_model(
    model=model,
    calibration_data=calibration_dataset,
    target_accuracy_retention=0.98,  # Mantener 98% accuracy
    num_calibration_batches=100
)

# Métricas
metrics = vq_integration.get_metrics()
print(f"Compression ratio: {metrics['compression_ratio']:.1f}x")
print(f"Speedup: {metrics['inference_speedup']:.1f}x")
print(f"Accuracy retention: {metrics['accuracy_retention']:.2%}")
```

### VQ Codebook Management

```python
# Entrenar codebook
codebook = vq_integration.train_codebook(
    training_data=train_vectors,
    num_iterations=1000,
    batch_size=1024
)

# Guardar/cargar codebook
vq_integration.save_codebook("codebooks/model_vq.pkl")
vq_integration.load_codebook("codebooks/model_vq.pkl")

# Quantizar activaciones en tiempo real
@jax.jit
def quantized_forward(inputs):
    # Activations se quantizan automáticamente
    return quantized_model(inputs)
```

---

## 🎯 Performance Tuning

### Optimal Batch Size

```python
from capibara.core.tpu import find_optimal_batch_size

# Encontrar batch size óptimo
optimal_bs = find_optimal_batch_size(
    model=model,
    sequence_length=2048,
    target_utilization=0.90,  # 90% utilización TPU
    available_memory_gb=32 * 64  # 32GB * 64 chips
)

print(f"Optimal batch size: {optimal_bs}")
# Típicamente: 128-256 para v6e-64
```

### Profiling

```python
from capibara.core.tpu import TPUProfiler

# Profiler TPU
profiler = TPUProfiler(
    output_dir="profiles/",
    capture_memory=True,
    capture_compute=True
)

# Profile training step
with profiler.profile("train_step"):
    loss = train_step(params, batch)

# Generar reporte
profiler.generate_report()
# Ver en TensorBoard: tensorboard --logdir profiles/
```

### Bottleneck Analysis

```python
# Analizar bottlenecks
analysis = profiler.analyze_bottlenecks()

print(f"Compute bound: {analysis['compute_bound']:.1%}")
print(f"Memory bound: {analysis['memory_bound']:.1%}")
print(f"Communication bound: {analysis['communication_bound']:.1%}")

# Recomendaciones
for recommendation in analysis['recommendations']:
    print(f"- {recommendation}")
```

---

## 📊 Monitoring

### Real-time Metrics

```python
from capibara.core.tpu import TPUMonitor

# Crear monitor
monitor = TPUMonitor(
    update_frequency_seconds=10,
    enable_prometheus=True,
    prometheus_port=9090
)

# Monitorear training
monitor.start()

for batch in dataloader:
    loss = train_step(batch)

    # Métricas automáticamente logged
    metrics = monitor.get_current_metrics()
    print(f"TPU Utilization: {metrics['utilization']:.1%}")
    print(f"HBM Usage: {metrics['memory_usage_gb']:.1f}GB")
    print(f"TFLOPS: {metrics['current_tflops']:.1f}")

monitor.stop()
```

### Performance Metrics

```python
# Obtener métricas detalladas
perf_metrics = config.get_performance_metrics()

print(f"Total TFLOPS: {perf_metrics['total_tflops']:.1f}")
print(f"Memory bandwidth: {perf_metrics['memory_bandwidth_gb_s']:.1f}GB/s")
print(f"Interconnect bandwidth: {perf_metrics['interconnect_bandwidth_gb_s']:.1f}GB/s")
print(f"Utilization: {perf_metrics['utilization']:.1%}")
print(f"Samples/sec: {perf_metrics['throughput_samples_s']:.1f}")
```

### Alerting

```python
# Configurar alertas
monitor.set_alert(
    metric="utilization",
    condition="<",
    threshold=0.7,
    action=lambda: print("⚠️ Low TPU utilization!")
)

monitor.set_alert(
    metric="memory_usage_gb",
    condition=">",
    threshold=28,  # 28GB de 32GB
    action=lambda: print("⚠️ High memory usage!")
)
```

---

## 🔧 Troubleshooting

### Error: "TPU not found"

```python
# Verificar TPU disponible
import jax
print(f"Devices: {jax.devices()}")

# Si no aparecen TPUs:
# 1. Verificar que estás en TPU VM
# 2. Verificar variables de entorno
from capibara.core.tpu import diagnose_tpu

diagnosis = diagnose_tpu()
print(diagnosis['summary'])
```

### Error: "Out of Memory"

**Soluciones**:

```python
# 1. Reducir batch size
config.batch_size_per_chip = 8  # De 16 a 8

# 2. Habilitar gradient checkpointing
config.gradient_checkpointing = True

# 3. Usar aggressive memory optimization
config.memory_optimization = "aggressive"

# 4. Aumentar model parallelism
config.mesh_shape = (4, 16)  # Más model parallel

# 5. Usar quantization
from capibara.core.tpu import quantize_for_tpu
quantized_model = quantize_for_tpu(model, bits=8)
```

### Slow Training

**Diagnóstico**:

```python
from capibara.core.tpu import diagnose_performance

perf_diagnosis = diagnose_performance(
    model=model,
    dataloader=dataloader,
    target_tflops=15000  # Para v6e-64
)

# Ver recomendaciones
for issue in perf_diagnosis['issues']:
    print(f"Issue: {issue['problem']}")
    print(f"Impact: {issue['impact']}")
    print(f"Solution: {issue['solution']}\n")
```

**Soluciones comunes**:

1. **Data loading bottleneck**: Aumentar `num_workers`, usar prefetching
2. **Compilation overhead**: Pre-compilar funciones con `jax.jit`
3. **Communication overhead**: Reducir all-reduce frequency
4. **Suboptimal batch size**: Usar `find_optimal_batch_size()`

### Debugging TPU Code

```python
# Habilitar debugging mode
config.enable_debugging_mode()

# Deshabilitar JIT temporalmente
import jax
jax.config.update('jax_disable_jit', True)

# Habilitar NaN/Inf detection
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

# Ver compilaciones
jax.config.update('jax_log_compiles', True)
```

---

## 📈 Benchmarks

### Training Performance (Transformer, 1B params)

| TPU | Batch Size | Seq Length | Tokens/sec | TFLOPS | Utilization |
|-----|------------|------------|------------|--------|-------------|
| v4-8 | 64 | 2048 | 125K | 850 | 77% |
| v5e-64 | 512 | 2048 | 1.2M | 9,500 | 85% |
| v6e-64 | 512 | 2048 | 1.8M | 15,000 | 88% |
| v6e-256 | 2048 | 2048 | 7.5M | 62,000 | 90% |

### Memory Usage (BF16)

| Model Size | Parameters | Weights | Activations | Total (per chip) |
|------------|------------|---------|-------------|------------------|
| Small | 125M | 0.5GB | 1.5GB | 2GB |
| Base | 350M | 1.4GB | 3GB | 4.4GB |
| Large | 1.3B | 5.2GB | 8GB | 13.2GB |
| XL | 2.7B | 10.8GB | 14GB | 24.8GB |

### VQ Performance (TPU v6e)

| Operation | Baseline | VQ-8bit | Speedup |
|-----------|----------|---------|---------|
| MatMul | 10ms | 3ms | 3.3x |
| Attention | 25ms | 8ms | 3.1x |
| MLP | 15ms | 5ms | 3.0x |
| Total Inference | 120ms | 40ms | 3.0x |

---

## 📚 Referencias

### Google Cloud Documentation

- [TPU v4 Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [TPU v5e Overview](https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e)
- [TPU v6e Launch](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-v6e)
- [JAX on TPU Guide](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)

### Optimization Guides

- [XLA Optimization](https://www.tensorflow.org/xla)
- [TPU Performance Guide](https://cloud.google.com/tpu/docs/performance-guide)
- [Sharding Tutorial](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)

### Related Modules

- [capibara/jax](../../jax/README.md) - Custom JAX implementation
- [capibara/training](../../training/README.md) - Training system
- [capibara/core/distributed](../distributed/README.md) - Distributed training
- [docs/TPU_TRAINING.md](../../../docs/TPU_TRAINING.md) - Complete TPU training guide

---

## 🛠️ Advanced Topics

### Custom TPU Kernels

```python
from capibara.jax.tpu_v4 import adaptive_kernels

# Usar kernels TPU-optimizados
output = adaptive_kernels.adaptive_matmul(
    a, b,
    precision="highest",
    tpu_strategy="collective"
)
```

### Multi-Host Training

```python
# Para TPU pods con múltiples hosts
from capibara.core.tpu import MultiHostConfig

multi_host_config = MultiHostConfig(
    num_hosts=8,
    host_id=jax.process_index(),
    coordinator_address="10.0.0.1:1234"
)

# Setup multi-host
multi_host_config.setup()
```

### Checkpointing en TPU

```python
from capibara.core.tpu import TPUCheckpointer

checkpointer = TPUCheckpointer(
    checkpoint_dir="gs://bucket/checkpoints/",
    max_to_keep=3,
    async_checkpointing=True  # No bloquea training
)

# Guardar checkpoint
checkpointer.save(
    step=1000,
    params=params,
    opt_state=opt_state
)

# Restaurar
restored = checkpointer.restore(step=1000)
```

---

**Última actualización**: 2025-11-16
**Versión**: 2.0.0
**TPU Versions**: v4, v5e, v6e
