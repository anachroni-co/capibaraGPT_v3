"""
Integration Examples for CapibaraGPT-v2 Adapter System

Ejemplos prácticos de integración y uso del system de adapters,
demostrando casos de uso reales y mejores prácticas.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional

# Imports from adapter system
from .adapter_registry import adapter_registry, AdapterType
from .kernel_abstraction_adapter import KernelAbstractionAdapter, KernelOperation, KernelExecutionContext
from .performance_adapter import PerformanceAdapter, OptimizationGoal
from .hardware_compatibility_adapter import HardwareCompatibilityAdapter, OptimizationLevel
from .quantization_adapter import QuantizationAdapter, QuantizationType, QuantizationQuality
from .language_processing_adapter import (
    LanguageProcessingAdapter, 
    CulturalContext, 
    MultilingualContext,
    ProcessingMode
)
from .adapter_metrics import (
    metrics_collector,
    start_metrics_collection,
    monitor_adapter_performance,
    get_metrics_overview
)

logger = logging.getLogger(__name__)

class AdapterSystemDemo:
    """Complete demonstration of the adapter system."""
    
    def __init__(self):
        self.adapters = {}
        self.initialized = False
    
    def initialize_system(self) -> bool:
        """Initializes the entire adapter system."""
        print("🚀 Inicializando Sistema de Adapters de CapibaraGPT-v2...")
        
        try:
            # 1. Inicializar adapters principales
            self.adapters['kernel'] = KernelAbstractionAdapter()
            self.adapters['performance'] = PerformanceAdapter(OptimizationGoal.BALANCED)
            self.adapters['hardware'] = HardwareCompatibilityAdapter(OptimizationLevel.BALANCED)
            self.adapters['quantization'] = QuantizationAdapter()
            self.adapters['language'] = LanguageProcessingAdapter()
            
            # 2. Inicializar cada adapter
            for name, adapter in self.adapters.items():
                print(f"  📦 Inicializando {name} adapter...")
                success = adapter.initialize()
                if success:
                    print(f"    ✅ {name} adapter inicializado correctamente")
                else:
                    print(f"    ❌ Error inicializando {name} adapter")
                    return False
            
            # 3. Iniciar system of metrics
            print("  📊 Iniciando system of metrics...")
            start_metrics_collection()
            
            # 4. Configurar monitoreo automático
            self._setup_monitoring()
            
            self.initialized = True
            print("✅ Sistema de adapters inicializado completamente\n")
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando system: {e}")
            return False
    
    def _setup_monitoring(self):
        """Configura el monitoreo automático."""
        def alert_handler(alert):
            level_emoji = {"warning": "⚠️", "error": "❌", "critical": "🚨"}
            emoji = level_emoji.get(alert.alert_level.value, "❓")
            print(f"{emoji} ALERTA: {alert.message}")
        
        metrics_collector.add_alert_callback(alert_handler)
    
    def demo_kernel_abstraction(self):
        """Demuestra el uso del Kernel Abstraction Adapter."""
        print("🔧 === DEMO: KERNEL ABSTRACTION ADAPTER ===")
        
        if not self.initialized:
            print("❌ Sistema no inicializado")
            return
        
        kernel_adapter = self.adapters['kernel']
        
        # 1. Mostrar backends disponibles
        backends = kernel_adapter.get_available_backends()
        print(f"📋 Backends disponibles: {list(backends.keys())}")
        
        # 2. Demo de Flash Attention
        print("\n🧠 Ejecutando Flash Attention...")
        
        # Create test data
        batch_size, seq_len, hidden_dim = 2, 128, 64
        query = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        key = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        value = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        
        # Configurar contexto de ejecución
        context = KernelExecutionContext(
            operation=KernelOperation.FLASH_ATTENTION,
            input_shape=(batch_size, seq_len, hidden_dim),
            dtype="float32",
            precision_requirements="high"
        )
        
        try:
            start_time = time.time()
            result = kernel_adapter.flash_attention(query, key, value, context=context)
            execution_time = (time.time() - start_time) * 1000
            
            print(f"  ✅ Flash Attention completado en {execution_time:.2f}ms")
            print(f"  📊 Shape resultado: {getattr(result, 'shape', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Error en Flash Attention: {e}")
        
        # 3. Demo de Matrix Multiply
        print("\n🔢 Ejecutando Matrix Multiply...")
        
        a = np.random.randn(256, 512).astype(np.float32)
        b = np.random.randn(512, 256).astype(np.float32)
        
        try:
            start_time = time.time()
            result = kernel_adapter.matrix_multiply(a, b)
            execution_time = (time.time() - start_time) * 1000
            
            print(f"  ✅ Matrix Multiply completado en {execution_time:.2f}ms")
            print(f"  📊 Shape resultado: {getattr(result, 'shape', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Error en Matrix Multiply: {e}")
        
        # 4. Mostrar statistics de operaciones
        stats = kernel_adapter.get_operation_stats()
        print(f"\n📈 Estadísticas de operaciones:")
        print(f"  🔄 Operaciones cacheadas: {stats['cached_operations']}")
        print(f"  🎯 Backends totales: {stats['total_backends']}")
        
        print("✅ Demo Kernel Abstraction completado\n")
    
    def demo_performance_optimization(self):
        """Demuestra el uso del Performance Adapter."""
        print("⚡ === DEMO: PERFORMANCE ADAPTER ===")
        
        if not self.initialized:
            print("❌ Sistema no inicializado")
            return
        
        performance_adapter = self.adapters['performance']
        
        # 1. Habilitar adaptación automática
        print("🔄 Habilitando adaptación automática...")
        performance_adapter.enable_auto_adaptation()
        
        # 2. Simular carga de trabajo
        print("🏋️ Simulando carga de trabajo...")
        
        @monitor_adapter_performance("DemoWorkload", "intensive_operation")
        def intensive_operation(size: int):
            # Simular operación intensiva
            data = np.random.randn(size, size)
            result = np.dot(data, data.T)
            time.sleep(0.1)  # Simular procesamiento
            return result
        
        # Ejecutar operaciones con diferentes cargas
        for i, size in enumerate([100, 200, 300, 500], 1):
            print(f"  🔄 Operación {i}: matriz {size}x{size}")
            try:
                result = intensive_operation(size)
                print(f"    ✅ Completado: shape {result.shape}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # 3. Obtener reporte de rendimiento
        print("\n📊 Generando reporte de rendimiento...")
        report = performance_adapter.get_performance_report()
        
        print(f"📈 Métricas actuales:")
        for metric, value in report['current_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        
        print(f"\n📊 Estadísticas de adaptación:")
        stats = report['adaptation_stats']
        print(f"  Total adaptaciones: {stats['total_adaptations']}")
        print(f"  Adaptaciones exitosas: {stats['successful_adaptations']}")
        
        # 4. Cambiar objetivo de optimización
        print("\n🎯 Cambiando objetivo a MINIMIZE_LATENCY...")
        performance_adapter.set_optimization_goal(OptimizationGoal.MINIMIZE_LATENCY)
        
        print("✅ Demo Performance Adapter completado\n")
    
    def demo_hardware_compatibility(self):
        """Demuestra el uso del Hardware Compatibility Adapter."""
        print("🖥️ === DEMO: HARDWARE COMPATIBILITY ADAPTER ===")
        
        if not self.initialized:
            print("❌ Sistema no inicializado")
            return
        
        hardware_adapter = self.adapters['hardware']
        
        # 1. Detectar hardware del system
        print("🔍 Detectando hardware del system...")
        
        try:
            hardware_profile = hardware_adapter.force_hardware_detection()
            
            print(f"💻 Sistema detectado: {hardware_profile['system_name']}")
            print(f"🏗️ Arquitectura: {hardware_profile['system_architecture']}")
            print(f"💾 Memoria total: {hardware_profile['total_memory_gb']:.1f} GB")
            print(f"⚡ Compute total: {hardware_profile['total_compute_tflops']:.1f} TFLOPS")
            
            print(f"\n🔧 Componentes detectados ({len(hardware_profile['capabilities'])}):")
            for cap in hardware_profile['capabilities'][:5]:  # Mostrar primeros 5
                print(f"  • {cap['name']}: {cap['hardware_type']}")
                if cap.get('memory_gb', 0) > 0:
                    print(f"    💾 {cap['memory_gb']:.1f} GB")
                if cap.get('peak_performance_tflops', 0) > 0:
                    print(f"    ⚡ {cap['peak_performance_tflops']:.1f} TFLOPS")
            
        except Exception as e:
            print(f"❌ Error detectando hardware: {e}")
            return
        
        # 2. Aplicar optimizaciones
        print("\n🔧 Aplicando optimizaciones de hardware...")
        
        try:
            optimizations = hardware_adapter.execute("optimize")
            applied = optimizations.get('applied_optimizations', [])
            
            if applied:
                print(f"✅ {len(applied)} optimizaciones aplicadas:")
                for opt in applied:
                    print(f"  • {opt['type']}: {opt['parameter']} = {opt['value']}")
                    print(f"    📈 Mejora esperada: {opt['expected_improvement']:.1f}%")
            else:
                print("ℹ️ No se aplicaron optimizaciones adicionales")
                
        except Exception as e:
            print(f"❌ Error aplicando optimizaciones: {e}")
        
        # 3. Obtener resumen del system
        print("\n📊 Resumen del system:")
        summary = hardware_adapter.get_hardware_summary()
        
        print(f"🔧 Componentes totales: {summary['total_components']}")
        print(f"💾 Memoria disponible: {summary['total_memory_gb']:.1f} GB")
        print(f"⚡ Potencia de cómputo: {summary['total_compute_tflops']:.1f} TFLOPS")
        print(f"🎯 Oportunidades de optimización: {summary['optimization_opportunities']}")
        
        print(f"\n🏷️ Tipos de hardware detectados:")
        for hw_type, count in summary['hardware_types'].items():
            print(f"  • {hw_type}: {count}")
        
        print("✅ Demo Hardware Compatibility completado\n")
    
    def demo_quantization_methods(self):
        """Demuestra el uso del Quantization Adapter."""
        print("🗜️ === DEMO: QUANTIZATION ADAPTER ===")
        
        if not self.initialized:
            print("❌ Sistema no inicializado")
            return
        
        quantization_adapter = self.adapters['quantization']
        
        # 1. Crear datos de prueba
        print("📊 Preparando datos de prueba...")
        test_data = np.random.randn(1000, 512).astype(np.float32)
        original_size = test_data.nbytes / (1024 * 1024)  # MB
        print(f"  📏 Tamaño original: {original_size:.2f} MB")
        
        # 2. Benchmark de methods disponibles
        print("\n🏃 Ejecutando benchmark de methods de cuantización...")
        
        try:
            benchmark_results = quantization_adapter.benchmark(test_data)
            
            print("📈 Resultados del benchmark:")
            for method, metrics in benchmark_results['benchmark_results'].items():
                if metrics.get('success', False):
                    print(f"\n  🔧 {method.upper()}:")
                    print(f"    📦 Compresión: {metrics['compression_ratio']:.1f}x")
                    print(f"    🎯 Precisión: {metrics['accuracy_retention']:.1%}")
                    print(f"    ⏱️ Tiempo: {metrics['execution_time_ms']:.1f}ms")
                    print(f"    💾 Ahorro: {metrics['memory_savings_mb']:.1f}MB")
                else:
                    print(f"  ❌ {method}: {metrics.get('error', 'Unknown error')}")
                    
        except Exception as e:
            print(f"❌ Error en benchmark: {e}")
            return
        
        # 3. Demostrar selección automática
        print("\n🤖 Demostrando selección automática de method...")
        
        for quality in [QuantizationQuality.HIGH_QUALITY, QuantizationQuality.BALANCED, QuantizationQuality.MAXIMUM_COMPRESSION]:
            print(f"\n  🎯 Calidad: {quality.value}")
            
            try:
                result = quantization_adapter.quantize(test_data, method=None, quality=quality)
                
                method_used = result.metadata.get('method', 'unknown')
                print(f"    🔧 Método seleccionado: {method_used}")
                print(f"    📦 Compresión: {result.compression_ratio:.1f}x")
                print(f"    🎯 Precisión: {result.accuracy_retention:.1%}")
                print(f"    ⏱️ Tiempo: {result.quantization_time_ms:.1f}ms")
                
                # Test de dequantización
                dequantized = quantization_adapter.dequantize(result.quantized_data)
                print(f"    ✅ Dequantización: {getattr(dequantized, 'shape', 'OK')}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # 4. Mostrar methods disponibles
        print("\n📋 Métodos disponibles:")
        methods_info = quantization_adapter.execute("get_methods")
        
        for method, info in methods_info['available_methods'].items():
            print(f"\n  🔧 {method.upper()}:")
            print(f"    ✅ Disponible: {info['available']}")
            print(f"    🎓 Calibrado: {info['calibrated']}")
            
            compressions = info['estimated_compression_ratios']
            print(f"    📦 Compresión estimada:")
            print(f"      Datos pequeños: {compressions['small_data']:.1f}x")
            print(f"      Datos medianos: {compressions['medium_data']:.1f}x")
            print(f"      Datos grandes: {compressions['large_data']:.1f}x")
        
        print("✅ Demo Quantization Adapter completado\n")
    
    def demo_language_processing(self):
        """Demuestra el uso del Language Processing Adapter."""
        print("🌐 === DEMO: LANGUAGE PROCESSING ADAPTER ===")
        
        if not self.initialized:
            print("❌ Sistema no inicializado")
            return
        
        language_adapter = self.adapters['language']
        
        # 1. Detección avanzada of language
        print("🔍 Demostrando detección avanzada of languages...")
        
        test_texts = [
            "Hello, how are you today?",
            "Hola, ¿cómo estás hoy?",
            "Hello, como estas? 你好吗?",  # Code-switching
            "مرحبا، كيف حالك اليوم؟",
            "Bonjour, comment allez-vous?"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n  📝 Texto {i}: '{text[:50]}...'")
            
            try:
                detection = language_adapter.detect_language(text)
                result = detection['detection_result']
                
                print(f"    🌍 Idioma principal: {result['primary_language']}")
                print(f"    🎯 Confianza: {result['confidence']:.2f}")
                print(f"    🔀 Multilingüe: {result['is_multilingual']}")
                print(f"    🔄 Code-switching: {result['code_switching']}")
                
                if result['is_multilingual']:
                    print(f"    🌐 Idiomas detectados: {result['languages_detected']}")
                
                print(f"    📊 Complejidad: {result['complexity_score']:.2f}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # 2. Adaptación cultural
        print("\n🏛️ Demostrando adaptación cultural...")
        
        cultural_examples = [
            {
                'text': "Please complete this task immediately",
                'source': CulturalContext.WESTERN_INDIVIDUALISTIC,
                'target': CulturalContext.EASTERN_COLLECTIVE
            },
            {
                'text': "I need this done by 3 PM exactly",
                'source': CulturalContext.WESTERN_INDIVIDUALISTIC,
                'target': CulturalContext.AFRICAN_COMMUNAL
            }
        ]
        
        for example in cultural_examples:
            print(f"\n  📝 Texto: '{example['text']}'")
            print(f"  🌍 De: {example['source'].value}")
            print(f"  🎯 A: {example['target'].value}")
            
            try:
                adaptation = language_adapter.adapt_culturally(
                    example['text'],
                    example['source'],
                    example['target']
                )
                
                result = adaptation['adaptation_result']
                print(f"  ✨ Adaptado: '{result['adapted_content']}'")
                print(f"  🔄 Cambios: {len(result['changes_made'])}")
                
                for change in result['changes_made'][:2]:  # Mostrar primeros 2
                    print(f"    • {change}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # 3. Procesamiento multilingüe completo
        print("\n🌐 Demostrando procesamiento multilingüe completo...")
        
        multilingual_text = "Hello everyone! Hola a todos! 大家好！"
        
        context = MultilingualContext(
            primary_language="en",
            secondary_languages=["es", "zh"],
            processing_mode=ProcessingMode.MULTILINGUAL,
            cultural_adaptation_level=0.8
        )
        
        try:
            analysis = language_adapter.process_multilingual(multilingual_text, context)
            
            print(f"  📝 Texto: '{multilingual_text}'")
            print(f"  🌍 Análisis of languages:")
            
            lang_detection = analysis['language_detection']
            print(f"    Idioma principal: {lang_detection['primary_language']}")
            print(f"    Multilingüe: {lang_detection['is_multilingual']}")
            
            if analysis['code_switching_analysis']:
                cs_analysis = analysis['code_switching_analysis']
                print(f"    Code-switching detectado: {cs_analysis['detected']}")
                if cs_analysis['detected']:
                    print(f"    Idiomas en code-switching: {cs_analysis['languages']}")
            
            print(f"  🎯 Recomendaciones de procesamiento:")
            for rec in analysis['recommendations'][:3]:  # Mostrar primeras 3
                print(f"    • {rec}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 4. Mostrar perfiles of languages disponibles
        print("\n📚 Perfiles of languages disponibles:")
        profiles = language_adapter.execute("get_profiles")
        
        print(f"  🌐 Total idiomas: {profiles['total_languages']}")
        print(f"  🏛️ Familias lingüísticas: {len(profiles['supported_families'])}")
        
        print(f"  📋 Algunos idiomas soportados:")
        for lang_code, profile in list(profiles['language_profiles'].items())[:5]:
            print(f"    • {profile['language_name']} ({lang_code})")
            print(f"      Familia: {profile['family']}")
            print(f"      Contexto cultural: {profile['cultural_context']}")
        
        print("✅ Demo Language Processing completado\n")
    
    def demo_metrics_system(self):
        """Demonstrates the automatic metrics system."""
        print("📊 === DEMO: SISTEMA DE MÉTRICAS AUTOMÁTICAS ===")
        
        # 1. Obtener overview del system
        print("🔍 Obteniendo overview del system...")
        overview = get_metrics_overview()
        
        print(f"📊 Estado del system:")
        print(f"  🔧 Adapters activos: {overview['total_adapters']}")
        print(f"  📈 Score promedio: {overview['system_performance']['average_system_score']:.2f}")
        print(f"  ⚠️ Alertas totales: {overview['total_alerts']}")
        print(f"  🚨 Alertas sin reconocer: {overview['unacknowledged_alerts']}")
        print(f"  🔄 Operaciones totales: {overview['system_performance']['total_operations']}")
        
        # 2. Estado por adapter
        print(f"\n📋 Estado por adapter:")
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
        
        for name, info in overview['adapters_summary'].items():
            emoji = status_emoji.get(info['status'], "❓")
            print(f"  {emoji} {name}:")
            print(f"    📊 Score: {info['performance_score']:.2f}")
            print(f"    ✅ Éxito: {info['success_rate']:.1%}")
            print(f"    ⏱️ Tiempo promedio: {info['avg_execution_time']:.1f}ms")
            print(f"    🔄 Operaciones: {info['total_operations']}")
        
        # 3. Obtener alertas recientes
        print(f"\n🚨 Alertas recientes:")
        alerts = metrics_collector.get_alerts(limit=5)
        
        if alerts:
            for alert in alerts:
                level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
                emoji = level_emoji.get(alert['alert_level'], "❓")
                
                print(f"  {emoji} {alert['adapter_name']} - {alert['metric_type']}")
                print(f"    📅 {alert['datetime'][:19]}")
                print(f"    📊 Valor: {alert['current_value']:.3f}")
                print(f"    ✅ Reconocida: {'Sí' if alert['acknowledged'] else 'No'}")
        else:
            print("  ✅ No hay alertas recientes")
        
        # 4. Métricas detalladas de un adapter específico
        if self.adapters:
            adapter_name = list(self.adapters.keys())[0]
            print(f"\n🔍 Métricas detalladas - {adapter_name}:")
            
            # Obtener métricas específicas
            adapter_metrics = metrics_collector.get_adapter_metrics(f"{adapter_name.title()}Adapter")
            
            if adapter_metrics:
                print(f"  ⏰ Uptime: {adapter_metrics['uptime_seconds']:.0f}s")
                print(f"  📊 Score de rendimiento: {adapter_metrics['performance_score']:.2f}")
                print(f"  🔄 Operaciones totales: {adapter_metrics['total_operations']}")
                
                print(f"  📈 Métricas actuales:")
                for metric, value in adapter_metrics['current_metrics'].items():
                    print(f"    {metric}: {value:.3f}")
            else:
                print("  ℹ️ Métricas no disponibles aún")
        
        print("✅ Demo Sistema de Métricas completado\n")
    
    def run_complete_demo(self):
        """Executes the complete system demonstration."""
        print("🎬 === DEMOSTRACIÓN COMPLETA DEL SISTEMA DE ADAPTERS ===\n")
        
        # Inicializar system
        if not self.initialize_system():
            print("❌ No se pudo inicializar el system")
            return
        
        # Ejecutar todas las demos
        demos = [
            ("Kernel Abstraction", self.demo_kernel_abstraction),
            ("Performance Optimization", self.demo_performance_optimization),
            ("Hardware Compatibility", self.demo_hardware_compatibility),
            ("Quantization Methods", self.demo_quantization_methods),
            ("Language Processing", self.demo_language_processing),
            ("Metrics System", self.demo_metrics_system)
        ]
        
        for demo_name, demo_func in demos:
            try:
                demo_func()
                time.sleep(1)  # Pausa entre demos
            except Exception as e:
                print(f"❌ Error en demo {demo_name}: {e}\n")
        
        # Resumen final
        print("🎯 === RESUMEN FINAL ===")
        
        final_overview = get_metrics_overview()
        print(f"✅ Sistema completamente operativo")
        print(f"📊 {final_overview['total_adapters']} adapters funcionando")
        print(f"🎯 Score promedio del system: {final_overview['system_performance']['average_system_score']:.2f}")
        print(f"🔄 {final_overview['system_performance']['total_operations']} operaciones ejecutadas")
        
        print("\n🚀 El system de adapters está listo para uso en producción!")
        print("📚 Consulta el README.md para más información y ejemplos avanzados.")

# Ejemplos de uso específicos

def example_kernel_integration():
    """Example of integration with existing kernels."""
    print("🔧 Ejemplo: Integración con kernels existentes")
    
    # Importar el adapter of kernels
    from .kernel_abstraction_adapter import kernel_adapter
    
    # Inicializar si no está inicializado
    if not kernel_adapter.get_status().value == "ready":
        kernel_adapter.initialize()
    
    # Usar flash attention con fallback automático
    query = np.random.randn(2, 10, 64).astype(np.float32)
    key = np.random.randn(2, 10, 64).astype(np.float32)
    value = np.random.randn(2, 10, 64).astype(np.float32)
    
    try:
        result = kernel_adapter.flash_attention(query, key, value)
        print(f"✅ Flash attention ejecutado: {result.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")

def example_performance_monitoring():
    """Performance monitoring example."""
    print("📊 Ejemplo: Monitoreo de rendimiento")
    
    # Decorator for automatic monitoring
    @monitor_adapter_performance("ExampleWorkload", "data_processing")
    def process_data(data_size: int):
        # Simular procesamiento de datos
        data = np.random.randn(data_size, data_size)
        result = np.linalg.svd(data, compute_uv=False)
        return result
    
    # Ejecutar operaciones monitoreadas
    for size in [100, 200, 300]:
        try:
            result = process_data(size)
            print(f"✅ Procesado matriz {size}x{size}: {len(result)} valores singulares")
        except Exception as e:
            print(f"❌ Error procesando {size}x{size}: {e}")

def example_quantization_pipeline():
    """Quantization pipeline example."""
    print("🗜️ Ejemplo: Pipeline de cuantización")
    
    from .quantization_adapter import quantization_adapter
    
    # Inicializar adapter
    if not quantization_adapter.get_status().value == "ready":
        quantization_adapter.initialize()
    
    # Simular pesos de modelo
    model_weights = np.random.randn(1000, 768).astype(np.float32)
    print(f"📊 Pesos originales: {model_weights.nbytes / (1024*1024):.1f} MB")
    
    # Cuantización automática
    try:
        result = quantization_adapter.quantize(
            model_weights,
            quality=QuantizationQuality.BALANCED
        )
        
        print(f"🔧 Método usado: {result.metadata.get('method', 'unknown')}")
        print(f"📦 Compresión: {result.compression_ratio:.1f}x")
        print(f"🎯 Precisión retenida: {result.accuracy_retention:.1%}")
        print(f"💾 Ahorro: {result.memory_savings_mb:.1f} MB")
        
        # Dequantizar para verificar
        dequantized = quantization_adapter.dequantize(result.quantized_data)
        print(f"✅ Dequantización exitosa: {dequantized.shape}")
        
    except Exception as e:
        print(f"❌ Error en cuantización: {e}")

# Main function to execute examples
def main():
    """Main function to execute the demonstration."""
    print("🎬 Iniciando demostración del Sistema de Adapters de CapibaraGPT-v2\n")
    
    # Crear y ejecutar demo completa
    demo = AdapterSystemDemo()
    demo.run_complete_demo()
    
    print("\n" + "="*80)
    print("📚 EJEMPLOS ADICIONALES")
    print("="*80)
    
    # Ejecutar ejemplos específicos
    examples = [
        ("Integración con Kernels", example_kernel_integration),
        ("Monitoreo de Rendimiento", example_performance_monitoring),
        ("Pipeline de Cuantización", example_quantization_pipeline)
    ]
    
    for example_name, example_func in examples:
        print(f"\n--- {example_name} ---")
        try:
            example_func()
        except Exception as e:
            print(f"❌ Error en ejemplo: {e}")
    
    print(f"\n🎉 Demostración completada!")
    print(f"📖 Para más información, consulta el README.md")

if __name__ == "__main__":
    main()