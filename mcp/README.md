# MCP - Model Control Protocol

**Resource Management and Model Routing System**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MODEL CONTROL PROTOCOL (MCP)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            ┌─────────────────┐                              │
│                            │  MCP Controller │                              │
│                            └────────┬────────┘                              │
│                                     │                                        │
│            ┌────────────────────────┼────────────────────────┐              │
│            │                        │                        │              │
│            ▼                        ▼                        ▼              │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │  Model Router   │     │    Resource     │     │    Version      │      │
│   │                 │     │    Manager      │     │    Manager      │      │
│   │ • Route requests│     │ • Memory mgmt  │     │ • Model versions│      │
│   │ • Load balance  │     │ • GPU allocation│     │ • A/B testing   │      │
│   │ • Failover      │     │ • Auto-scaling  │     │ • Rollback      │      │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│            │                        │                        │              │
│            └────────────────────────┼────────────────────────┘              │
│                                     │                                        │
│                                     ▼                                        │
│                         ┌─────────────────────┐                             │
│                         │   Model Instances   │                             │
│                         │  ┌───┐ ┌───┐ ┌───┐ │                             │
│                         │  │M1 │ │M2 │ │M3 │ │                             │
│                         │  └───┘ └───┘ └───┘ │                             │
│                         └─────────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The MCP (Model Control Protocol) module provides infrastructure for managing multiple model instances, routing requests, allocating resources, and handling model versions. It enables efficient deployment and scaling of CapibaraGPT models.

## Module Structure

```
mcp/
├── __init__.py           # Module exports
├── model_router.py       # Request routing logic
├── resource_manager.py   # Resource allocation
└── version_manager.py    # Model version control
```

## Components

### Model Router

Routes incoming requests to appropriate model instances based on load, capabilities, and configuration.

```python
from mcp import ModelRouter, RoutingConfig

# Configure router
config = RoutingConfig(
    strategy="least_loaded",     # or "round_robin", "random", "capability"
    health_check_interval=30,    # seconds
    timeout=60,                  # request timeout
    retry_count=3
)

router = ModelRouter(config)

# Register model instances
router.register_model(
    name="capibara-7b-v1",
    endpoint="http://model1:8080",
    capabilities=["text", "code"],
    max_concurrent=10
)

router.register_model(
    name="capibara-7b-v1",
    endpoint="http://model2:8080",
    capabilities=["text", "code"],
    max_concurrent=10
)

# Route request
response = await router.route(
    request={"prompt": "Hello, world!"},
    required_capabilities=["text"]
)
```

**Routing Strategies:**

```
┌────────────────────────────────────────────────────────────────┐
│                    Routing Strategies                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Round Robin:                                                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Request 1 → Model A                                     │ │
│  │  Request 2 → Model B                                     │ │
│  │  Request 3 → Model C                                     │ │
│  │  Request 4 → Model A  (cycle repeats)                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Least Loaded:                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Model A: 8/10 capacity → Skip                          │ │
│  │  Model B: 3/10 capacity → SELECT                       │ │
│  │  Model C: 7/10 capacity → Skip                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Capability-Based:                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Request needs: [code, reasoning]                       │ │
│  │  Model A: [text, code]         → Partial match          │ │
│  │  Model B: [code, reasoning]    → Full match            │ │
│  │  Model C: [text]               → No match               │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Resource Manager

Manages computational resources (memory, GPU, TPU) across model instances.

```python
from mcp import ResourceManager, ResourceConfig

# Configure resource manager
config = ResourceConfig(
    max_memory_gb=64,
    max_gpu_memory_gb=80,
    auto_scale=True,
    scale_threshold=0.8,     # Scale up at 80% utilization
    scale_down_threshold=0.3 # Scale down at 30% utilization
)

manager = ResourceManager(config)

# Allocate resources for model
allocation = manager.allocate(
    model_name="capibara-7b",
    requirements={
        "memory_gb": 16,
        "gpu_memory_gb": 24,
        "cpu_cores": 4
    }
)

# Check resource availability
available = manager.check_availability({
    "memory_gb": 32,
    "gpu_memory_gb": 40
})

# Release resources
manager.release(allocation.id)

# Get resource statistics
stats = manager.get_stats()
# {
#     "total_memory_gb": 64,
#     "used_memory_gb": 32,
#     "total_gpu_memory_gb": 80,
#     "used_gpu_memory_gb": 48,
#     "active_allocations": 3
# }
```

**Auto-Scaling:**

```
┌────────────────────────────────────────────────────────────────┐
│                     Auto-Scaling Flow                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Utilization Monitoring                                       │
│   ┌──────────────────────────────────────────────────────────┐│
│   │  Current: ████████████████████░░░░  85%                 ││
│   │  Threshold: ─────────────────────|── 80%                ││
│   │                                  ↑                       ││
│   │                            Scale Up Triggered            ││
│   └──────────────────────────────────────────────────────────┘│
│                                                                │
│   Scale Up Action:                                            │
│   1. Check available resources                                │
│   2. Launch new model instance                                │
│   3. Wait for health check                                    │
│   4. Register with router                                     │
│   5. Begin accepting traffic                                  │
│                                                                │
│   Scale Down Action:                                          │
│   1. Mark instance for drain                                  │
│   2. Stop accepting new requests                              │
│   3. Wait for in-flight requests                              │
│   4. Deregister from router                                   │
│   5. Release resources                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Version Manager

Manages model versions, deployments, and rollbacks.

```python
from mcp import VersionManager

manager = VersionManager(
    storage_path="./model_versions",
    max_versions=5  # Keep last 5 versions
)

# Register new version
manager.register_version(
    model_name="capibara",
    version="7b-v1.2",
    path="./models/capibara-7b-v1.2",
    metadata={
        "training_date": "2024-01-15",
        "dataset": "v3",
        "metrics": {"accuracy": 0.95}
    }
)

# Deploy specific version
manager.deploy(
    model_name="capibara",
    version="7b-v1.2",
    target="production"
)

# Rollback to previous version
manager.rollback(
    model_name="capibara",
    target="production"
)

# A/B testing
manager.setup_ab_test(
    model_name="capibara",
    versions=["7b-v1.1", "7b-v1.2"],
    traffic_split=[0.5, 0.5]
)

# List versions
versions = manager.list_versions("capibara")
# [
#     {"version": "7b-v1.2", "status": "production", "deployed": "2024-01-16"},
#     {"version": "7b-v1.1", "status": "canary", "deployed": "2024-01-10"},
#     {"version": "7b-v1.0", "status": "archived", "deployed": "2024-01-01"},
# ]
```

**Version States:**

| State | Description |
|-------|-------------|
| `development` | Being developed/tested |
| `staging` | Ready for production testing |
| `canary` | Receiving small % of traffic |
| `production` | Serving production traffic |
| `archived` | No longer active, kept for rollback |

## Usage Examples

### Basic Setup

```python
from mcp import MCPController, MCPConfig

# Initialize MCP
config = MCPConfig(
    router_strategy="least_loaded",
    auto_scale=True,
    health_check_interval=30
)

mcp = MCPController(config)

# Start MCP services
await mcp.start()

# Register models
mcp.register_model(
    name="capibara-7b",
    version="v1.2",
    endpoint="http://localhost:8080"
)

# Handle requests
response = await mcp.process(
    request={"prompt": "Explain quantum computing"},
    routing_hints={"prefer_version": "v1.2"}
)

# Shutdown
await mcp.shutdown()
```

### Production Deployment

```python
from mcp import MCPController, MCPConfig
from mcp.monitoring import MetricsCollector

# Production configuration
config = MCPConfig(
    router_strategy="capability",
    auto_scale=True,
    min_instances=2,
    max_instances=10,
    health_check_interval=10,
    request_timeout=120,
    circuit_breaker_threshold=5
)

mcp = MCPController(config)

# Add monitoring
metrics = MetricsCollector(
    export_interval=60,
    exporters=["prometheus", "cloudwatch"]
)
mcp.attach_metrics(metrics)

# Deploy with canary
mcp.version_manager.deploy(
    model_name="capibara",
    version="7b-v1.3",
    target="canary",
    traffic_percentage=10
)

# Monitor and promote
if metrics.canary_success_rate > 0.99:
    mcp.version_manager.promote(
        model_name="capibara",
        version="7b-v1.3",
        from_target="canary",
        to_target="production"
    )
```

## Configuration

```yaml
# config/mcp.yaml
mcp:
  router:
    strategy: "least_loaded"
    health_check_interval: 30
    timeout: 60
    retry_count: 3
    circuit_breaker:
      enabled: true
      threshold: 5
      recovery_time: 60

  resources:
    max_memory_gb: 128
    max_gpu_memory_gb: 160
    auto_scale: true
    scale_up_threshold: 0.8
    scale_down_threshold: 0.3
    min_instances: 2
    max_instances: 10

  versions:
    storage_path: "./model_versions"
    max_versions: 10
    auto_cleanup: true

  monitoring:
    enabled: true
    export_interval: 60
    exporters:
      - prometheus
      - cloudwatch
```

## API Reference

### MCPController

| Method | Description |
|--------|-------------|
| `start()` | Start MCP services |
| `shutdown()` | Graceful shutdown |
| `register_model(...)` | Register model instance |
| `deregister_model(name)` | Remove model instance |
| `process(request)` | Route and process request |
| `get_status()` | Get MCP status |

### ModelRouter

| Method | Description |
|--------|-------------|
| `route(request)` | Route request to model |
| `register_model(...)` | Register model endpoint |
| `health_check()` | Check all model health |
| `get_stats()` | Get routing statistics |

### ResourceManager

| Method | Description |
|--------|-------------|
| `allocate(requirements)` | Allocate resources |
| `release(allocation_id)` | Release resources |
| `check_availability(req)` | Check if resources available |
| `get_stats()` | Get resource statistics |

### VersionManager

| Method | Description |
|--------|-------------|
| `register_version(...)` | Register new version |
| `deploy(model, version, target)` | Deploy version |
| `rollback(model, target)` | Rollback to previous |
| `setup_ab_test(...)` | Configure A/B test |
| `list_versions(model)` | List all versions |

## See Also

- [Core Backends](../core/backends/README.md)
- [Inference Module](../inference/README.md)
- [Training Module](../training/README.md)

## Example quick

Example (pseudo-command) para iniciar un servicio MCP:

```bash
python -m mcp.server --config config/configs_toml/mcp.toml
```
