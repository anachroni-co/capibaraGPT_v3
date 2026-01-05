#!/usr/bin/env python3
"""
Integration Service for Programming-Specific RAG with Resource Monitoring

Este servicio implementa la integración completa entre:
1. El detector de consultas de programación (solo activa RAG para programación)
2. El monitor de recursos que envía datos cada 2 segundos a la VM services
3. El sistema de fallback a colas de trabajo cuando recursos > 90%
"""

import time
import asyncio
import threading
from typing import Dict, Any, Optional
from resource_publisher import ResourceMonitor
from programming_rag_detector import is_programming_query


class IntegratedResourceProgrammingRAG:
    """
    Sistema integrado que combina:
    - Detección de consultas de programación
    - Monitoreo de recursos para fallback
    - Enviar información a VM services
    """
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor(
            services_vm_host="34.175.255.139",
            services_vm_port=5000
        )
        
        # Variables para control de recursos
        self.resource_data = None
        self.last_resource_update = 0
        self.high_resource_mode = False
        
        # Bandera para saber si el monitoreo de recursos está corriendo
        self.monitoring_active = False
        
        print("🌐 Integrated Resource Programming RAG System Initialized")
        print("   Programming-specific RAG detection: ACTIVE")
        print("   Resource monitoring: READY")
        print("   High-resource fallback: CONFIGURED")
    
    def should_activate_rag(self, query: str) -> bool:
        """
        Determina si se debe activar RAG para una consulta

        Args:
            query: Consulta del usuario

        Returns:
            True si es una consulta de programación Y recursos no están altos
        """
        is_programming = is_programming_query(query)

        # Obtener estado de recursos actual
        current_resource_data = self.get_current_resource_status()

        # Verificar si hay alta demanda de recursos
        high_resource_usage = current_resource_data.get('overall_status', {}).get('resource_threshold_exceeded', False)

        if is_programming and not high_resource_usage:
            print(f"✅ RAG ACTIVATED: Programming query with sufficient resources")
            return True
        elif is_programming and high_resource_usage:
            print(f"⚠️  RAG DEFERRED: Programming query but resources are high - queue fallback")
            return False
        elif not is_programming:
            print(f"❌ RAG SKIPPED: Not a programming query")
            return False

        return False
    
    def get_current_resource_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual de recursos"""
        if self.resource_data:
            # Solo actualizar cada 5 segundos como máximo
            if time.time() - self.last_resource_update > 5:
                new_data = self.resource_monitor.get_resource_usage()
                if new_data:
                    self.resource_data = new_data
                    self.last_resource_update = time.time()
        else:
            # Obtener datos por primera vez
            self.resource_data = self.resource_monitor.get_resource_usage()
            self.last_resource_update = time.time()
        
        return self.resource_data or {}
    
    def start_resource_monitoring(self, interval: float = 2.0):
        """Inicia el monitoreo continuo de recursos"""
        if not self.monitoring_active:
            self.resource_monitor.start_monitoring(interval_seconds=interval)
            self.monitoring_active = True
            print(f"📊 Resource monitoring started with {interval}s interval")
    
    def stop_resource_monitoring(self):
        """Detiene el monitoreo de recursos"""
        if self.monitoring_active:
            self.resource_monitor.stop_monitoring()
            self.monitoring_active = False
            print("📊 Resource monitoring stopped")


def main():
    print("🌐 Capibara6 Integrated Programming RAG & Resource System")
    print("=" * 70)
    print("Sistema que combina:")
    print("- Detector de consultas de programación (solo activa RAG para código)")
    print("- Monitoreo de recursos cada 2 segundos") 
    print("- Comunicación con VM services para fallback a colas")
    print("- Activación condicional de RAG basado en uso de recursos")
    print("=" * 70)
    
    # Crear sistema integrado
    integrated_system = IntegratedResourceProgrammingRAG()
    
    # Iniciar monitoreo de recursos
    integrated_system.start_resource_monitoring(interval=2.0)
    
    print(f"\n🔄 SISTEMA ACTIVO")
    print("Presiona Ctrl+C para detener...")
    
    # Simular pruebas de consultas
    test_queries = [
        "How to sort an array in Python?",
        "What is the weather like today?",
        "Write a JavaScript function to reverse a string",
        "Can you explain quantum physics?",
        "Help with debugging this Python code"
    ]
    
    print(f"\n🧪 PRUEBA DE DETECCIÓN:")
    for query in test_queries:
        activate_rag = integrated_system.should_activate_rag(query)
        print(f"   Query: '{query}' → RAG: {activate_rag}")
    
    print(f"\n📈 El sistema está monitoreando recursos...")
    print(f"   - Consultas de programación activarán RAG si recursos disponibles")
    print(f"   - Consultas no de programación nunca activarán RAG")
    print(f"   - Si recursos > 90%, se usará fallback a colas")
    
    try:
        # Mantener el sistema corriendo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\n🛑 Deteniendo sistema integrado...")
        integrated_system.stop_resource_monitoring()
        print(f"👋 Sistema detenido correctamente")


if __name__ == "__main__":
    main()