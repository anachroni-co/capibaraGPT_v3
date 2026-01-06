# Monitoring Module

Sistema avanzado de monitoreo y alertas para TPU con métricas en tiempo real, detección de anomalías y gestión proactiva de rendimiento.

## 📋 Descripción

Este módulo proporciona capacidades comprehensivas de monitoreo para infraestructura TPU, incluyendo métricas de rendimiento, alertas configurables, análisis de tendencias y optimizaciones automáticas basadas en telemetría en tiempo real.

## 🏗️ Arquitectura

```
monitoring/
├── __init__.py          # Exports del sistema de monitoreo
├── tpu_alerts.py        # Sistema de alertas TPU
├── tpu_monitor.py       # Monitor básico TPU
└── tpu_decorators.py    # Decoradores de monitoreo
```

## 🔍 Componentes Principales

### 1. Sistema de Alertas TPU (`tpu_alerts.py`)

Sistema avanzado de alertas con umbrales configurables y gestión de cooldown.

```python
from capibara.core.monitoring import TPUAlertManager

# Configurar sistema de alertas
alert_manager = TPUAlertManager(
    alert_thresholds={
        "memory_utilization": 0.85,
        "compute_utilization": 0.90,
        "temperature": 75.0,  # Celsius
        "latency_ms": 500,
        "error_rate": 0.05,
        "tflops_degradation": 0.20
    },
    cooldown_periods={
        "memory_alert": 300,    # 5 minutos
        "temperature_alert": 600, # 10 minutos
        "performance_alert": 180  # 3 minutos
    },
    notification_channels=["email", "slack", "webhook"],
    alert_severity_levels=["info", "warning", "critical", "emergency"]
)

# Configurar métricas críticas
critical_metrics = alert_manager.configure_critical_metrics({
    "oom_detection": {
        "threshold": 0.95,
        "window_size": 30,  # segundos
        "severity": "critical"
    },
    "thermal_throttling": {
        "threshold": 80.0,
        "consecutive_readings": 3,
        "severity": "emergency"
    },
    "performance_degradation": {
        "baseline_deviation": 0.30,
        "measurement_window": 300,
        "severity": "warning"
    }
})

# Procesar métricas y generar alertas
current_metrics = {
    "memory_utilization": 0.89,
    "compute_utilization": 0.94,
    "temperature": 77.5,
    "latency_ms": 520,
    "tflops": 312.5,
    "error_count": 12
}

alerts_triggered = alert_manager.process_metrics(current_metrics)
for alert in alerts_triggered:
    print(f"🚨 {alert.severity.upper()}: {alert.message}")
    print(f"   Threshold: {alert.threshold}, Current: {alert.current_value}")
    print(f"   Suggested Action: {alert.suggested_action}")
```

### 2. Monitor TPU Básico (`tpu_monitor.py`)

Monitor fundamental con métricas básicas de TPU.

```python
from capibara.core.monitoring import TPUMonitor

# Inicializar monitor básico
monitor = TPUMonitor(
    monitoring_interval=5,  # segundos
    metrics_history_size=1000,
    enable_automatic_logging=True,
    log_level="INFO"
)

# Monitoreo básico de métricas
basic_metrics = monitor.get_basic_metrics()
print(f"TPU Status: {basic_metrics['status']}")
print(f"Memory Usage: {basic_metrics['memory_usage_gb']:.1f}GB")
print(f"Temperature: {basic_metrics['temperature']:.1f}°C")
print(f"Utilization: {basic_metrics['utilization']:.1%}")

# Monitoreo con contexto
with monitor.context("model_inference"):
    # Código a monitorear
    model_output = model(input_batch)
    
# Obtener métricas del contexto
context_metrics = monitor.get_context_metrics("model_inference")
print(f"Inference Time: {context_metrics['duration_ms']:.1f}ms")
print(f"Peak Memory: {context_metrics['peak_memory_gb']:.1f}GB")
```

### 3. Decoradores de Monitoreo (`tpu_decorators.py`)

Decoradores para monitoreo automático de funciones.

```python
from capibara.core.monitoring import (
    monitor_tpu_performance,
    monitor_memory_usage,
    monitor_latency,
    alert_on_anomaly
)

# Decorador de rendimiento completo
@monitor_tpu_performance(
    track_memory=True,
    track_compute=True,
    track_latency=True,
    alert_thresholds={
        "memory": 0.85,
        "latency_ms": 1000
    }
)
def inference_function(inputs):
    return model(inputs)

# Decorador específico de memoria
@monitor_memory_usage(
    alert_threshold=0.90,
    track_leaks=True,
    auto_gc=True
)
def memory_intensive_function(large_tensor):
    # Procesamiento que usa mucha memoria
    result = complex_computation(large_tensor)
    return result

# Decorador de detección de anomalías
@alert_on_anomaly(
    baseline_window=100,
    deviation_threshold=2.0,  # 2 desviaciones estándar
    metrics=["latency", "memory", "compute"]
)
def production_inference(batch):
    return model.predict(batch)

# Usar funciones decoradas
inputs = get_batch_inputs()
outputs = inference_function(inputs)

# Obtener métricas de la función decorada
function_metrics = monitor_tpu_performance.get_metrics("inference_function")
print(f"Average Latency: {function_metrics['avg_latency_ms']:.1f}ms")
print(f"Peak Memory: {function_metrics['peak_memory_gb']:.1f}GB")
```

## 📊 Métricas Detalladas

### Sistema de Métricas Comprehensivo

```python
# Configurar colector de métricas avanzado
from capibara.core.monitoring import AdvancedMetricsCollector

metrics_collector = AdvancedMetricsCollector(
    collection_frequency=1,  # Hz
    metrics_categories=[
        "hardware_utilization",
        "memory_statistics", 
        "compute_performance",
        "thermal_management",
        "power_consumption",
        "network_io",
        "model_performance"
    ],
    enable_predictive_analytics=True,
    anomaly_detection=True
)

# Métricas de hardware
hardware_metrics = metrics_collector.collect_hardware_metrics()
detailed_metrics = {
    "tpu_utilization": {
        "scalar_utilization": hardware_metrics["scalar_util"],
        "vector_utilization": hardware_metrics["vector_util"], 
        "matrix_utilization": hardware_metrics["matrix_util"],
        "memory_bandwidth_util": hardware_metrics["mem_bandwidth_util"]
    },
    
    "memory_breakdown": {
        "hbm_total_gb": hardware_metrics["hbm_total"],
        "hbm_used_gb": hardware_metrics["hbm_used"],
        "hbm_free_gb": hardware_metrics["hbm_free"],
        "fragmentation_ratio": hardware_metrics["fragmentation"],
        "allocation_efficiency": hardware_metrics["alloc_efficiency"]
    },
    
    "thermal_profile": {
        "core_temperature": hardware_metrics["core_temp"],
        "memory_temperature": hardware_metrics["mem_temp"],
        "ambient_temperature": hardware_metrics["ambient_temp"],
        "cooling_efficiency": hardware_metrics["cooling_eff"],
        "thermal_throttling": hardware_metrics["throttling"]
    },
    
    "performance_counters": {
        "instructions_per_cycle": hardware_metrics["ipc"],
        "cache_hit_ratio": hardware_metrics["cache_hits"],
        "memory_access_latency": hardware_metrics["mem_latency"],
        "compute_throughput_tflops": hardware_metrics["tflops"],
        "communication_overhead": hardware_metrics["comm_overhead"]
    }
}

print("🔍 Hardware Metrics Summary:")
for category, metrics in detailed_metrics.items():
    print(f"\n{category.upper()}:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
```

### Análisis Predictivo

```python
# Sistema de análisis predictivo
from capibara.core.monitoring import PredictiveAnalytics

predictor = PredictiveAnalytics(
    models=["arima", "lstm", "prophet"],
    prediction_horizon="1h",
    confidence_interval=0.95,
    retrain_frequency="daily"
)

# Predicciones de rendimiento
performance_forecast = predictor.predict_performance_trends(
    historical_data=metrics_collector.get_historical_data(hours=24),
    forecast_metrics=["memory_usage", "latency", "throughput", "temperature"]
)

print("📈 Performance Predictions (Next Hour):")
for metric, prediction in performance_forecast.items():
    print(f"{metric}:")
    print(f"  Predicted Value: {prediction['value']:.3f}")
    print(f"  Confidence: {prediction['confidence']:.1%}")
    print(f"  Trend: {prediction['trend']}")
    if prediction['alert_probability'] > 0.3:
        print(f"  ⚠️  Alert Probability: {prediction['alert_probability']:.1%}")

# Detección de anomalías
anomaly_detector = predictor.get_anomaly_detector()
current_state = metrics_collector.get_current_state()
anomaly_score = anomaly_detector.compute_anomaly_score(current_state)

if anomaly_score > 0.8:
    print(f"🚨 High Anomaly Score Detected: {anomaly_score:.3f}")
    anomaly_details = anomaly_detector.explain_anomaly(current_state)
    print(f"Primary Contributing Factors: {anomaly_details['factors']}")
```

## 🎯 Dashboard y Visualización

### Dashboard en Tiempo Real

```python
from capibara.core.monitoring import MonitoringDashboard

# Crear dashboard de monitoreo
dashboard = MonitoringDashboard(
    refresh_rate=2,  # segundos
    charts=[
        "tpu_utilization_timeline",
        "memory_usage_heatmap",
        "latency_distribution",
        "temperature_gauge",
        "throughput_trend",
        "alert_history"
    ],
    export_formats=["json", "prometheus", "grafana"],
    real_time_alerts=True
)

# Configurar widgets del dashboard
dashboard.add_widget("performance_summary", {
    "type": "summary_card",
    "metrics": ["avg_latency", "peak_memory", "current_tflops"],
    "update_frequency": 5
})

dashboard.add_widget("alert_status", {
    "type": "alert_panel", 
    "severity_filter": ["warning", "critical"],
    "max_alerts": 10
})

dashboard.add_widget("resource_utilization", {
    "type": "gauge_cluster",
    "gauges": ["cpu", "memory", "compute", "bandwidth"],
    "thresholds": {"warning": 0.7, "critical": 0.9}
})

# Exportar métricas para sistemas externos
dashboard.export_metrics(
    format="prometheus",
    endpoint="/metrics",
    labels={"service": "capibara", "environment": "production"}
)
```

### Integración con Grafana

```python
# Configuración para Grafana
grafana_config = {
    "datasource": {
        "name": "CapibaraTPU",
        "type": "prometheus",
        "url": "http://localhost:9090"
    },
    "dashboards": [
        {
            "name": "TPU Performance Overview",
            "panels": [
                "tpu_utilization",
                "memory_usage",
                "temperature_monitoring",
                "latency_trends"
            ]
        },
        {
            "name": "Model Performance",
            "panels": [
                "inference_latency",
                "throughput_metrics",
                "accuracy_tracking",
                "resource_efficiency"
            ]
        }
    ]
}

# Exportar configuración
dashboard.export_grafana_config(grafana_config)
```

## 🔧 Configuración Avanzada

### Configuración de Alertas Personalizadas

```python
# Configuración avanzada de alertas
custom_alert_rules = [
    {
        "name": "memory_leak_detection",
        "condition": "memory_growth_rate > 100MB/min for 10min",
        "severity": "warning",
        "actions": ["log", "notify", "trigger_gc"]
    },
    {
        "name": "performance_regression",
        "condition": "avg_latency > baseline * 1.5 for 5min", 
        "severity": "critical",
        "actions": ["alert", "auto_scale", "circuit_breaker"]
    },
    {
        "name": "thermal_emergency",
        "condition": "temperature > 85°C",
        "severity": "emergency",
        "actions": ["immediate_throttle", "emergency_notification"]
    },
    {
        "name": "batch_processing_anomaly",
        "condition": "batch_completion_time > p95 * 2",
        "severity": "warning",
        "actions": ["investigate", "adjust_batch_size"]
    }
]

# Aplicar reglas personalizadas
alert_manager.add_custom_rules(custom_alert_rules)

# Sistema de escalación
escalation_policy = {
    "warning": {
        "immediate": ["log", "metrics_update"],
        "after_5min": ["slack_notification"],
        "after_15min": ["email_team"]
    },
    "critical": {
        "immediate": ["alert_dashboard", "slack_critical"],
        "after_2min": ["email_oncall", "auto_mitigation"],
        "after_5min": ["escalate_to_manager"]
    },
    "emergency": {
        "immediate": ["all_notifications", "emergency_protocols"],
        "after_1min": ["executive_notification"],
        "continuous": ["status_page_update"]
    }
}

alert_manager.configure_escalation(escalation_policy)
```

### Optimización Automática

```python
# Sistema de optimización automática basado en métricas
from capibara.core.monitoring import AutoOptimizer

auto_optimizer = AutoOptimizer(
    optimization_targets={
        "latency": {"target": "minimize", "weight": 0.4},
        "throughput": {"target": "maximize", "weight": 0.3},
        "resource_utilization": {"target": "optimize", "weight": 0.3}
    },
    constraints={
        "memory_usage": {"max": 0.85},
        "temperature": {"max": 75.0},
        "accuracy_loss": {"max": 0.02}
    },
    optimization_frequency="hourly",
    rollback_on_degradation=True
)

# Configurar parámetros optimizables
optimizable_params = {
    "batch_size": {"range": [8, 64], "type": "discrete"},
    "learning_rate": {"range": [1e-5, 1e-3], "type": "continuous"},
    "memory_optimization_level": {"range": [0, 3], "type": "discrete"},
    "compute_precision": {"options": ["float32", "bfloat16", "int8"], "type": "categorical"}
}

# Ejecutar optimización automática
optimization_result = auto_optimizer.optimize(
    parameters=optimizable_params,
    evaluation_duration="30min",
    max_iterations=20
)

print(f"🚀 Optimization Complete!")
print(f"Latency Improvement: {optimization_result['latency_improvement']:.1%}")
print(f"Throughput Improvement: {optimization_result['throughput_improvement']:.1%}")
print(f"Optimal Parameters: {optimization_result['optimal_params']}")
```

## 🔒 Monitoreo de Seguridad

### Seguridad y Compliance

```python
# Monitoreo de seguridad integrado
from capibara.core.monitoring import SecurityMonitor

security_monitor = SecurityMonitor(
    threat_detection=True,
    compliance_checks=["gdpr", "hipaa", "sox"],
    audit_logging=True,
    anomaly_detection=True
)

# Métricas de seguridad
security_metrics = security_monitor.get_security_metrics()
compliance_status = {
    "data_encryption": security_metrics["encryption_status"],
    "access_control": security_metrics["access_violations"],
    "audit_trail": security_metrics["audit_completeness"],
    "data_retention": security_metrics["retention_compliance"],
    "privacy_protection": security_metrics["privacy_score"]
}

# Alertas de seguridad
security_alerts = security_monitor.check_security_violations()
for alert in security_alerts:
    print(f"🔒 Security Alert: {alert.message}")
    print(f"   Severity: {alert.severity}")
    print(f"   Compliance Impact: {alert.compliance_impact}")
```

## 📚 Integración y APIs

### APIs de Monitoreo

```python
# API RESTful para métricas
from capibara.core.monitoring import MonitoringAPI

api = MonitoringAPI(port=8080)

# Endpoints disponibles
@api.route("/metrics/current")
def get_current_metrics():
    return metrics_collector.get_current_state()

@api.route("/metrics/history/<int:hours>")
def get_historical_metrics(hours):
    return metrics_collector.get_historical_data(hours=hours)

@api.route("/alerts/active")
def get_active_alerts():
    return alert_manager.get_active_alerts()

@api.route("/health")
def health_check():
    return {
        "status": "healthy",
        "tpu_status": monitor.get_tpu_status(),
        "last_update": monitor.get_last_update_time()
    }

# Webhooks para integraciones
api.add_webhook("/webhook/slack", slack_notification_handler)
api.add_webhook("/webhook/pagerduty", pagerduty_integration)
api.add_webhook("/webhook/custom", custom_alert_handler)

# Iniciar API
api.start()
```

## 📈 Benchmarking y Testing

### Suite de Benchmarks

```python
# Sistema de benchmarking integrado
from capibara.core.monitoring import BenchmarkSuite

benchmark_suite = BenchmarkSuite(
    benchmark_categories=[
        "inference_performance",
        "training_throughput", 
        "memory_efficiency",
        "thermal_stability",
        "scaling_behavior"
    ],
    baseline_comparison=True,
    regression_detection=True
)

# Ejecutar benchmarks
benchmark_results = benchmark_suite.run_comprehensive_benchmark(
    model=production_model,
    test_data=benchmark_dataset,
    duration="1h"
)

# Análisis de resultados
performance_report = benchmark_suite.generate_performance_report(
    results=benchmark_results,
    include_recommendations=True,
    compare_to_baseline=True
)

print("📊 Benchmark Results:")
print(f"Overall Performance Score: {performance_report['score']:.2f}/100")
print(f"Regression Detected: {'Yes' if performance_report['regression'] else 'No'}")
print(f"Recommendations: {len(performance_report['recommendations'])} items")
```

## 🤝 Referencias y Documentación

- [TPU Monitoring Best Practices](https://cloud.google.com/tpu/docs/monitoring)
- [Prometheus Metrics Design](https://prometheus.io/docs/practices/naming/)
- [Grafana Dashboard Creation](https://grafana.com/docs/grafana/latest/dashboards/)
- [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)
- [ML System Monitoring Patterns](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/)