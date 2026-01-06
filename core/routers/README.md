# Routers Module

Advanced routing and load balancing system with fallback mechanisms, TPU v4-32 optimization, and adaptive routing strategies.

## 📋 Description

This module manages intelligent request routing, load balancing, and automatic fallback mechanisms with specific optimizations for TPU and real-time performance monitoring.

## 🏗️ Architecture

```
routers/
├── base.py              # Base router with TPU optimizations
├── adaptive_router.py   # Adaptive router based on load
├── bto.py              # Best Time Optimization routing
├── enhanced_router.py   # Enhanced router with advanced features
└── tts_router.py       # TTS-specific router
```

## 🚀 TPU-Optimized Base Router

```python
from capibara.core.routers import BaseRouter

# Configure base router with TPU v4-32 optimizations
base_router = BaseRouter(
    tpu_config={
        "mesh_shape": (8, 4),  # TPU v4-32
        "memory_threshold_gb": 30,
        "latency_threshold_ms": 500,
        "enable_fallback": True,
        "fallback_device": "cpu"
    },
    monitoring_enabled=True,
    automatic_scaling=True
)

# Configure endpoints and services
endpoints = [
    {"id": "tpu_primary", "device": "tpu", "capacity": 100, "priority": 1},
    {"id": "gpu_secondary", "device": "gpu", "capacity": 80, "priority": 2},
    {"id": "cpu_fallback", "device": "cpu", "capacity": 50, "priority": 3}
]

base_router.register_endpoints(endpoints)

# Route request with automatic optimization
request_data = {
    "model": "capibara_large",
    "input": "Explain quantum computing",
    "max_tokens": 1000,
    "priority": "high"
}

routing_result = base_router.route_request(
    request=request_data,
    consider_load=True,
    enable_fallback=True,
    return_metrics=True
)

print(f"Routed to: {routing_result.endpoint_id}")
print(f"Expected latency: {routing_result.estimated_latency_ms}ms")
print(f"Load balancing score: {routing_result.load_score:.3f}")
```

## 🎯 Adaptive Router

```python
from capibara.core.routers import AdaptiveRouter

# Router that adapts based on load and performance
adaptive_router = AdaptiveRouter(
    adaptation_strategy="dynamic_weighted",
    learning_rate=0.1,
    history_window=1000,
    performance_weights={
        "latency": 0.4,
        "throughput": 0.3,
        "accuracy": 0.2,
        "availability": 0.1
    }
)

# Configure adaptation metrics
adaptation_config = {
    "load_monitoring": {
        "cpu_utilization": True,
        "memory_usage": True,
        "queue_length": True,
        "response_time": True
    },
    "performance_tracking": {
        "success_rate": True,
        "error_rate": True,
        "timeout_rate": True,
        "quality_metrics": True
    },
    "adaptation_triggers": {
        "performance_degradation": 0.2,
        "load_imbalance": 0.3,
        "failure_rate": 0.05
    }
}

adaptive_router.configure_adaptation(adaptation_config)

# Adaptive processing
for request in request_stream:
    # Router learns and adapts automatically
    result = adaptive_router.route_adaptive(
        request=request,
        update_weights=True,
        record_performance=True
    )

    # Update metrics based on result
    adaptive_router.update_performance_metrics(
        endpoint_id=result.endpoint_id,
        actual_latency=result.actual_latency,
        success=result.success,
        quality_score=result.quality_score
    )
```

## ⚡ Best Time Optimization (BTO)

```python
from capibara.core.routers import BTORouter

# BTO router for temporal optimization
bto_router = BTORouter(
    optimization_objective="minimize_total_time",
    prediction_model="lstm_based",
    look_ahead_window=300,  # 5 minutes
    dynamic_rebalancing=True
)

# Configure load prediction
prediction_config = {
    "time_series_features": [
        "historical_load",
        "time_of_day",
        "day_of_week",
        "seasonal_patterns"
    ],
    "model_update_frequency": "hourly",
    "prediction_horizon": "30min",
    "confidence_threshold": 0.8
}

bto_router.configure_time_prediction(prediction_config)

# Routing with temporal optimization
future_requests = bto_router.predict_future_load(
    time_horizon="1h",
    include_seasonal=True
)

optimal_schedule = bto_router.optimize_request_scheduling(
    current_requests=pending_requests,
    predicted_load=future_requests,
    optimization_window="2h"
)

for scheduled_request in optimal_schedule:
    print(f"Request {scheduled_request.id}:")
    print(f"  Optimal time: {scheduled_request.scheduled_time}")
    print(f"  Endpoint: {scheduled_request.endpoint}")
    print(f"  Expected completion: {scheduled_request.completion_time}")
```

## 🔧 Enhanced Router

```python
from capibara.core.routers import EnhancedRouter

# Router with advanced features
enhanced_router = EnhancedRouter(
    features=[
        "circuit_breaker",
        "rate_limiting",
        "request_prioritization",
        "intelligent_caching",
        "anomaly_detection"
    ],
    multi_objective_optimization=True,
    real_time_analytics=True
)

# Configure circuit breaker
circuit_breaker_config = {
    "failure_threshold": 5,
    "timeout_threshold": 10000,  # ms
    "recovery_timeout": 30000,   # ms
    "half_open_max_calls": 3,
    "automatic_recovery": True
}

# Configure rate limiting
rate_limiting_config = {
    "requests_per_minute": 1000,
    "burst_capacity": 100,
    "per_user_limit": 50,
    "priority_multipliers": {
        "premium": 2.0,
        "standard": 1.0,
        "free": 0.5
    }
}

enhanced_router.configure_advanced_features(
    circuit_breaker=circuit_breaker_config,
    rate_limiting=rate_limiting_config
)

# Routing with advanced features
advanced_result = enhanced_router.route_with_intelligence(
    request=complex_request,
    user_tier="premium",
    enable_caching=True,
    priority_boost=True
)

if advanced_result.circuit_breaker_open:
    print("⚠️ Circuit breaker activated for endpoint")
elif advanced_result.rate_limited:
    print("🕒 Request rate limited, retry after:", advanced_result.retry_after)
elif advanced_result.cached_response:
    print("⚡ Response served from cache")
```

## 🎤 Specialized TTS Router

```python
from capibara.core.routers import TTSRouter

# Specific router for Text-to-Speech
tts_router = TTSRouter(
    voice_models={
        "neural_voice_v3": {"quality": "high", "latency": "medium", "languages": 50},
        "fast_voice_v2": {"quality": "medium", "latency": "low", "languages": 20},
        "premium_voice": {"quality": "ultra", "latency": "high", "languages": 10}
    },
    quality_vs_latency_optimization=True,
    language_specific_routing=True
)

# Configure TTS routing
tts_routing_config = {
    "voice_selection": {
        "automatic_matching": True,
        "quality_preference": "adaptive",  # Based on request
        "fallback_voice": "neural_voice_v3"
    },
    "optimization_criteria": {
        "real_time_requirement": True,
        "quality_threshold": 0.8,
        "language_coverage": True,
        "emotional_range": True
    },
    "caching_strategy": {
        "common_phrases": True,
        "user_preferences": True,
        "voice_samples": True,
        "cache_ttl": 3600  # 1 hour
    }
}

tts_router.configure_tts_routing(tts_routing_config)

# Intelligent TTS routing
tts_request = {
    "text": "Hello, how can I help you today?",
    "language": "en-US",
    "emotion": "friendly",
    "speed": "normal",
    "quality": "high",
    "real_time": True
}

tts_result = tts_router.route_tts_request(
    request=tts_request,
    user_preferences={"voice_gender": "female", "accent": "american"},
    context="customer_service"
)

print(f"Selected voice model: {tts_result.voice_model}")
print(f"Expected generation time: {tts_result.generation_time_ms}ms")
print(f"Quality score: {tts_result.quality_score:.3f}")
```

## 📊 Monitoring and Metrics

```python
# Router monitoring system
from capibara.core.routers import RouterMonitor

monitor = RouterMonitor(
    routers=[base_router, adaptive_router, bto_router, enhanced_router],
    metrics_collection_frequency=5,  # seconds
    alert_thresholds={
        "high_latency": 1000,  # ms
        "high_error_rate": 0.05,
        "load_imbalance": 0.3,
        "endpoint_failure": True
    }
)

# Real-time metrics dashboard
routing_metrics = monitor.get_comprehensive_metrics()

performance_dashboard = {
    "overall_performance": {
        "total_requests_per_sec": routing_metrics["throughput"]["total_rps"],
        "average_latency_ms": routing_metrics["latency"]["avg_ms"],
        "error_rate": routing_metrics["errors"]["rate"],
        "load_balance_score": routing_metrics["load_balance"]["score"]
    },

    "endpoint_health": {
        endpoint["id"]: {
            "status": endpoint["status"],
            "load": endpoint["current_load"],
            "response_time": endpoint["avg_response_time"],
            "error_rate": endpoint["error_rate"]
        }
        for endpoint in routing_metrics["endpoints"]
    },

    "router_intelligence": {
        "adaptation_score": routing_metrics["adaptation"]["effectiveness"],
        "prediction_accuracy": routing_metrics["prediction"]["accuracy"],
        "cache_hit_rate": routing_metrics["caching"]["hit_rate"],
        "circuit_breaker_activations": routing_metrics["reliability"]["cb_activations"]
    }
}

# Automatic alerts
active_alerts = monitor.check_alerts()
for alert in active_alerts:
    print(f"🚨 {alert.severity}: {alert.message}")
    if alert.auto_mitigation_available:
        print(f"🔧 Applying mitigation: {alert.mitigation_action}")
        monitor.apply_auto_mitigation(alert)
```

## 🔄 Advanced Load Balancing

```python
# Multi-objective load balancing configuration
load_balancing_config = {
    "algorithms": {
        "primary": "weighted_round_robin",
        "fallback": "least_connections",
        "intelligent": "ml_based_prediction"
    },

    "weights": {
        "endpoint_capacity": 0.3,
        "current_load": 0.25,
        "historical_performance": 0.2,
        "resource_availability": 0.15,
        "geographic_proximity": 0.1
    },

    "dynamic_adjustment": {
        "enabled": True,
        "adjustment_frequency": 30,  # seconds
        "sensitivity": 0.1,
        "stability_factor": 0.8
    }
}

# Apply advanced load balancing
for router in [base_router, adaptive_router, enhanced_router]:
    router.configure_load_balancing(load_balancing_config)

# Effectiveness analysis
balancing_analysis = monitor.analyze_load_balancing_effectiveness(
    time_window="1h",
    include_predictions=True
)

print("📈 Load Balancing Analysis:")
print(f"Distribution efficiency: {balancing_analysis['distribution_efficiency']:.3f}")
print(f"Resource utilization: {balancing_analysis['resource_utilization']:.3f}")
print(f"Response time variance: {balancing_analysis['response_time_variance']:.3f}")
print(f"Predicted improvements: {balancing_analysis['improvement_potential']:.3f}")
```

## 🤝 Modular Integration

```python
# Integration with other CapibaraGPT modules
from capibara.core.monitoring import TPUMonitor
from capibara.core.moe import DynamicMoE

# Router integrated with MoE system
moe_aware_router = EnhancedRouter(
    expert_system=DynamicMoE(num_experts=32),
    expert_routing=True,
    load_aware_expert_selection=True
)

# Routing with expert selection
expert_routing_result = moe_aware_router.route_to_expert(
    request=complex_request,
    required_expertise=["mathematics", "reasoning"],
    quality_requirements={"accuracy": 0.95, "latency": 200}
)

# Integrated monitoring
with TPUMonitor().context("intelligent_routing"):
    processed_result = expert_routing_result.endpoint.process_request(
        request=complex_request,
        expert_ids=expert_routing_result.selected_experts
    )

# Integrated system metrics
integration_metrics = {
    "routing_efficiency": expert_routing_result.efficiency_score,
    "expert_utilization": expert_routing_result.expert_load_balance,
    "quality_preservation": processed_result.quality_score,
    "total_processing_time": processed_result.total_time_ms
}
```

## 📚 References

- [Load Balancing Algorithms](https://en.wikipedia.org/wiki/Load_balancing_(computing))
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Consistent Hashing](https://en.wikipedia.org/wiki/Consistent_hashing)
- [Rate Limiting Strategies](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [Adaptive Load Balancing](https://research.google/pubs/pub44824/)
