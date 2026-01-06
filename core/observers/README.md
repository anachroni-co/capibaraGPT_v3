# Observer Pattern for Dynamic Expert Activation

Este módulo implementa el patrón Observer para activar dinámicamente expertos basado en patrones de peticiones y entradas. Permite un sistema de enrutamiento inteligente que puede adaptarse a diferentes tipos de consultas y activar los expertos más apropiados automáticamente.

## 🎯 Características Principales

### 🔍 Observadores Inteligentes
- **RequestPatternObserver**: Detecta patrones específicos en el texto de las peticiones
- **ComplexityObserver**: Analiza la complejidad de las consultas y activa expertos según sea necesario
- **DomainSpecificObserver**: Especializado en detectar dominios específicos (matemáticas, programación, etc.)
- **PerformanceObserver**: Monitorea el rendimiento del sistema y toma decisiones basadas en la carga
- **AdaptiveObserver**: Aprende de patrones de activación y se adapta con el tiempo

### 🚀 Gestión Dinámica de Expertos
- **ExpertActivationManager**: Gestiona el ciclo de vida de los expertos
- **ExpertPool**: Pool dinámico de expertos que pueden ser activados bajo demanda
- **ActivationStrategy**: Diferentes estrategias para la activación de expertos

### 🔄 Integración con Router
- **ObserverAwareRouter**: Router que integra el patrón Observer con el sistema de enrutamiento tradicional
- **RoutingMode**: Diferentes modos de enrutamiento (tradicional, observer-first, híbrido)
- **DynamicRoutingDecision**: Decisiones de enrutamiento enriquecidas con información de observadores

## 📋 Casos de Uso

### 1. Activación Automática por Patrones
```python
from capibara.core.observers import create_observer_aware_router, RoutingMode

# Crear router con patrón observer
router = create_observer_aware_router(
    routing_mode=RoutingMode.OBSERVER_ENHANCED
)

# Procesar petición - los expertos se activan automáticamente
result = await router.route_request(
    input_data="¿Qué pasaría si el servidor principal falla durante el pico de tráfico?"
)

print(f"Expertos activados: {result.experts_activated}")
# Output: ['CSA'] - Expert de análisis contrafactual activado automáticamente
```

### 2. Detección de Complejidad
```python
# Petición compleja que activa múltiples expertos
complex_query = """
Diseña un sistema de machine learning para predecir fallos del servidor.
¿Qué pasaría si los datos de entrenamiento están sesgados?
Incluye el código en Python y calcula la probabilidad de error.
"""

result = await router.route_request(input_data=complex_query)
print(f"Expertos activados: {result.experts_activated}")
# Output: ['CSA', 'CodeExpert', 'MathExpert'] - Múltiples expertos para consulta compleja
```

### 3. Aprendizaje Adaptativo
```python
from capibara.core.observers import ActivationStrategy

# Router con aprendizaje adaptativo
adaptive_router = create_observer_aware_router(
    routing_mode=RoutingMode.OBSERVER_ENHANCED,
    activation_strategy=ActivationStrategy.ADAPTIVE
)

# Procesar peticiones y proporcionar feedback
result = await adaptive_router.route_request(input_data="Calculate derivative of x²")
adaptive_router.provide_feedback("req_001", {"MathExpert": True})  # Feedback positivo

# El sistema aprende y mejora las activaciones futuras
```

### 4. Observadores Personalizados
```python
from capibara.core.observers import RequestObserver, create_expert_activation_event

class CustomDomainObserver(RequestObserver):
    async def observe(self, event):
        if "blockchain" in event.request_text.lower():
            return [create_expert_activation_event(
                expert_name="BlockchainExpert",
                reason="Blockchain domain detected",
                confidence=0.8
            )]
        return []

# Añadir observador personalizado
router.add_observer(CustomDomainObserver("BlockchainObserver"))
```

## 🏗️ Arquitectura

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    ObserverAwareRouter                      │
├─────────────────────────────────────────────────────────────┤
│                RouterObserverIntegration                    │
├─────────────────────────────────────────────────────────────┤
│              ExpertActivationManager                        │
├─────────────────────────────────────────────────────────────┤
│    ObserverManager           │         ExpertPool           │
│  ┌─────────────────────────┐ │ ┌─────────────────────────┐   │
│  │ RequestPatternObserver  │ │ │    CSAExpert           │   │
│  │ ComplexityObserver      │ │ │    MathExpert          │   │
│  │ DomainSpecificObserver  │ │ │    CodeExpert          │   │
│  │ PerformanceObserver     │ │ │    SpanishExpert       │   │
│  │ AdaptiveObserver        │ │ │    CustomExperts       │   │
│  └─────────────────────────┘ │ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Activación

1. **Recepción de Petición**: El router recibe una petición del usuario
2. **Notificación a Observadores**: Se notifica a todos los observadores activos
3. **Análisis de Patrones**: Cada observador analiza la petición según su especialidad
4. **Generación de Eventos**: Los observadores generan eventos de activación de expertos
5. **Aplicación de Estrategia**: Se aplica la estrategia de activación configurada
6. **Activación de Expertos**: Se activan los expertos apropiados
7. **Procesamiento**: Los expertos procesan la petición
8. **Combinación de Resultados**: Se combinan los resultados de múltiples expertos

## 📊 Estrategias de Activación

### ActivationStrategy.IMMEDIATE
Activa expertos inmediatamente cuando los observadores lo sugieren.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.IMMEDIATE
)
```

### ActivationStrategy.THRESHOLD_BASED
Requiere un nivel mínimo de confianza para activar expertos.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.THRESHOLD_BASED
)
# Configura el umbral
router.observer_integration.strategy_config["confidence_threshold"] = 0.7
```

### ActivationStrategy.CONSENSUS_REQUIRED
Requiere que múltiples observadores estén de acuerdo antes de activar un experto.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.CONSENSUS_REQUIRED
)
# Configura votos requeridos
router.observer_integration.strategy_config["consensus_required_votes"] = 2
```

### ActivationStrategy.LOAD_BALANCED
Considera la carga actual del sistema al tomar decisiones de activación.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.LOAD_BALANCED
)
```

### ActivationStrategy.ADAPTIVE
Aprende de activaciones pasadas y se adapta automáticamente.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.ADAPTIVE
)
```

## 🎛️ Modos de Enrutamiento

### RoutingMode.TRADITIONAL
Usa solo el enrutamiento tradicional, sin observadores.

### RoutingMode.OBSERVER_FIRST
Los observadores tienen prioridad sobre el enrutamiento tradicional.

### RoutingMode.OBSERVER_ENHANCED
Combina enrutamiento tradicional con activación de observadores.

### RoutingMode.HYBRID
Cambia dinámicamente entre modos según la complejidad de la petición.

## 📈 Monitoreo y Métricas

### Estadísticas del Router
```python
stats = router.get_statistics()
print(f"Total peticiones: {stats['integration_metrics']['total_requests']}")
print(f"Activaciones de observadores: {stats['integration_metrics']['observer_activations']}")
print(f"Activaciones de expertos: {stats['integration_metrics']['expert_activations']}")
```

### Rendimiento de Observadores
```python
observer_manager = router.observer_integration.activation_manager.observer_manager
for observer in observer_manager.observers:
    performance = observer.get_performance_summary()
    print(f"{observer.name}: {performance['success_rate']:.2%} éxito")
```

### Métricas de Expertos
```python
expert_pool = router.observer_integration.activation_manager.expert_pool
pool_stats = expert_pool.get_pool_statistics()
print(f"Utilización del pool: {pool_stats['current_utilization']:.2%}")
```

## 🧪 Ejemplos y Demos

### Demo Comprensivo
```python
from capibara.core.observers.examples import ObserverPatternDemo

demo = ObserverPatternDemo()
await demo.run_comprehensive_demo()
```

### Demo Interactivo
```python
from capibara.core.observers.examples import InteractiveObserverDemo

interactive_demo = InteractiveObserverDemo()
await interactive_demo.run_interactive_demo()
```

### Test Rápido
```python
from capibara.core.observers.examples import quick_test

await quick_test()
```

### Benchmark de Rendimiento
```python
from capibara.core.observers.examples import benchmark_observer_performance

await benchmark_observer_performance()
```

## 🔧 Configuración Avanzada

### Configuración de Observadores
```python
# Crear observador de patrones personalizado
pattern_observer = RequestPatternObserver("CustomPatterns", priority=1)
pattern_observer.expert_patterns["MyExpert"] = [
    r"(?i)\b(custom|pattern|detection)\b"
]

router.add_observer(pattern_observer)
```

### Registro de Expertos Personalizados
```python
class MyCustomExpert:
    async def process(self, context):
        return {"result": "Custom processing completed"}

router.register_expert("MyExpert", MyCustomExpert, max_concurrent=2)
```

### Configuración de Estrategias
```python
# Configurar parámetros de estrategia
router.observer_integration.strategy_config.update({
    "confidence_threshold": 0.8,
    "consensus_required_votes": 3,
    "max_concurrent_activations": 5,
    "load_balance_threshold": 0.7
})
```

## 🐛 Debugging y Logs

### Habilitar Logging Detallado
```python
import logging
logging.getLogger("capibara.core.observers").setLevel(logging.DEBUG)
```

### Inspeccionar Eventos de Activación
```python
# Acceder al historial de activaciones
activation_history = router.observer_integration.activation_manager.activation_history
for activation in activation_history[-5:]:  # Últimas 5 activaciones
    print(f"Request: {activation['request_id']}")
    print(f"Experts: {activation['approved_activations']}")
```

## 🚀 Mejores Prácticas

1. **Configuración de Prioridades**: Asigna prioridades apropiadas a los observadores
2. **Umbrales de Confianza**: Ajusta umbrales según tus necesidades de precisión
3. **Monitoreo de Rendimiento**: Supervisa regularmente las métricas del sistema
4. **Feedback de Aprendizaje**: Proporciona feedback para mejorar el aprendizaje adaptativo
5. **Gestión de Recursos**: Configura límites de concurrencia apropiados para los expertos

## 📚 Referencias

- [Patrón Observer](https://refactoring.guru/design-patterns/observer)
- [Sistemas de Expertos](https://en.wikipedia.org/wiki/Expert_system)
- [Enrutamiento Dinámico](https://en.wikipedia.org/wiki/Dynamic_routing)

## 🤝 Contribución

Para contribuir al desarrollo del patrón Observer:

1. Implementa nuevos observadores especializados
2. Mejora las estrategias de activación existentes
3. Añade métricas y monitoreo adicional
4. Crea ejemplos y casos de uso
5. Optimiza el rendimiento del sistema

## 📄 Licencia

Este módulo forma parte de CapibaraGPT y está sujeto a la misma licencia del proyecto principal.