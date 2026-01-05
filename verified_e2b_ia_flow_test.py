#!/usr/bin/env python3
"""
Prueba Verificada del Flujo E2B-IA basado en el estado actual del sistema
"""

import requests
import json
import time
from typing import Dict, Any

class VerifiedE2BIAFlowTest:
    """
    Clase para verificar el flujo real de integración E2B-IA con base en el sistema actual
    """
    
    def __init__(self):
        self.test_results = {}
    
    def test_current_infrastructure(self) -> Dict[str, Any]:
        """
        Verificar la infraestructura actual disponible
        """
        print("🔍 VERIFICACIÓN DE INFRAESTRUCTURA ACTUAL")
        print("-" * 50)
        
        # Testear servicios individuales
        tests = {
            "Gateway Server (8080)": {
                "url": "http://localhost:8080/api/health",
                "test": lambda: self._check_gateway()
            },
            "Models-Europe vLLM (8082)": {
                "url": "http://10.204.0.9:8082/health", 
                "test": lambda: self._check_vllm_direct()
            },
            "Flask API (5000)": {
                "url": "http://localhost:5000/api/health",
                "test": lambda: self._check_flask_api()
            },
            "Nginx Proxy": {
                "url": "http://localhost:80/api/health",
                "test": lambda: self._check_nginx_proxy()
            }
        }
        
        results = {}
        for service_name, service_info in tests.items():
            try:
                result = service_info["test"]()
                results[service_name] = result
                status = "✅" if result["available"] else "❌"
                print(f"{status} {service_name}: {result['status']}")
            except Exception as e:
                results[service_name] = {"available": False, "error": str(e)}
                print(f"❌ {service_name}: Error - {str(e)}")
        
        return results
    
    def _check_gateway(self) -> Dict[str, Any]:
        """Verificar estado del gateway server"""
        try:
            response = requests.get("http://localhost:8080/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "available": True,
                    "status": f"Healthy - vLLM: {data.get('services', {}).get('vllm', 'unknown')}",
                    "details": data
                }
            else:
                return {"available": False, "status": f"HTTP {response.status_code}"}
        except:
            return {"available": False, "status": "No accesible"}
    
    def _check_vllm_direct(self) -> Dict[str, Any]:
        """Verificar directamente vLLM en models-europe"""
        try:
            response = requests.get("http://10.204.0.9:8082/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "available": True, 
                    "status": f"Healthy - {data.get('models_loaded', 0)}/{data.get('models_available', 0)} modelos cargados",
                    "details": data
                }
            else:
                return {"available": False, "status": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"available": False, "status": f"Error de conexión: {str(e)}"}
    
    def _check_flask_api(self) -> Dict[str, Any]:
        """Verificar Flask API en puerto 5000"""
        try:
            response = requests.get("http://localhost:5000/api/health", timeout=5)
            if response.status_code == 200:
                return {"available": True, "status": "Healthy", "details": response.json()}
            else:
                return {"available": False, "status": f"HTTP {response.status_code}"}
        except:
            return {"available": False, "status": "No accesible"}
    
    def _check_nginx_proxy(self) -> Dict[str, Any]:
        """Verificar si Nginx puede proxy a servicios internos"""
        try:
            # Intentar acceder al health check a través de nginx (simulando frontend)
            response = requests.get("http://localhost/api/health", headers={"Host": "www.capibara6.com"}, timeout=5)
            if response.status_code == 200:
                return {"available": True, "status": "Proxy funcionando"}
            else:
                return {"available": False, "status": f"Proxy HTTP {response.status_code}"}
        except:
            return {"available": False, "status": "Proxy no accesible"}
    
    def test_e2b_integration_availability(self) -> Dict[str, Any]:
        """
        Verificar si la integración E2B está disponible
        """
        print(f"\n🔧 VERIFICACIÓN DE INTEGRACIÓN E2B")
        print("-" * 50)
        
        # Verificar en el backend que se supone maneja E2B
        try:
            # El backend integrado corre en el puerto 5001
            response = requests.get("http://localhost:5001/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                e2b_available = health_data.get('e2b_available', False)
                print(f"✅ Servidor integrado disponible - E2B integration: {e2b_available}")
                
                result = {
                    "available": True,
                    "e2b_available": e2b_available,
                    "details": health_data
                }
            else:
                print(f"❌ Servidor integrado no disponible (HTTP {response.status_code})")
                result = {"available": False, "e2b_available": False}
        except Exception as e:
            print(f"⚠️  Servidor integrado no corriendo en puerto 5001: {str(e)}")
            print("   (Este es el comportamiento esperado si el servicio no está iniciado)")
            result = {"available": False, "e2b_available": False, "error": str(e)}
        
        return result
    
    def test_actual_flow_verification(self) -> Dict[str, Any]:
        """
        Verificar el flujo real basado en los servicios disponibles
        """
        print(f"\n🔄 VERIFICACIÓN DEL FLUJO REAL")
        print("-" * 50)
        
        # Verificar qué servicios están disponibles
        infra_results = self.test_current_infrastructure()
        e2b_result = self.test_e2b_integration_availability()
        
        flow_analysis = {
            "frontend_access": infra_results.get("Nginx Proxy", {}).get("available", False),
            "gateway_available": infra_results.get("Gateway Server (8080)", {}).get("available", False),
            "models_available": infra_results.get("Models-Europe vLLM (8082)", {}).get("available", False),
            "e2b_available": e2b_result.get("e2b_available", False),
            "basic_api_available": infra_results.get("Flask API (5000)", {}).get("available", False)
        }
        
        print(f"\n📋 ANÁLISIS DEL FLUJO:")
        print(f"   Frontend ↔ Nginx: {'✅ Disponible' if flow_analysis['frontend_access'] else '❌ No disponible'}")
        print(f"   Nginx ↔ Gateway: {'✅ Disponible' if flow_analysis['gateway_available'] else '❌ No disponible'}")
        print(f"   Gateway ↔ Models: {'✅ Disponible' if flow_analysis['models_available'] else '❌ No disponible'}")
        print(f"   E2B Integration: {'✅ Disponible' if flow_analysis['e2b_available'] else '❌ No disponible'}")
        print(f"   Base API (Flask): {'✅ Disponible' if flow_analysis['basic_api_available'] else '❌ No disponible'}")
        
        # Determinar estado del flujo
        if flow_analysis['frontend_access'] and flow_analysis['gateway_available']:
            if flow_analysis['models_available']:
                print(f"\n🟢 FLUJO BÁSICO DE IA FUNCIONAL:")
                print(f"   Frontend → Nginx → Gateway → Models-Europe → Resultado")
            else:
                print(f"\n🟡 FLUJO DE IA PARCIALMENTE FUNCIONAL:")
                print(f"   Frontend → Nginx → Gateway → (Modelos no disponibles)")
        
        if flow_analysis['e2b_available']:
            print(f"   E2B Integration: Disponible para tareas de sandbox")
        else:
            print(f"   E2B Integration: No disponible (requiere iniciar servidor integrado)")
        
        return flow_analysis
    
    def generate_test_code_and_verify_flow(self):
        """
        Simular el flujo completo de generación de código y ejecución
        """
        print(f"\n🧪 SIMULACIÓN DE FLUJO COMPLETO E2B-IA")
        print("-" * 50)
        
        print("1️⃣  Simulando petición de usuario: 'Genera visualización con plotly'")
        print("2️⃣  IA debería generar código (en models-europe si disponible)")
        
        # Código de ejemplo que podría generar la IA
        example_ia_code = '''
import plotly.graph_objects as go
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
fig.update_layout(title='Gráfica generada por IA', xaxis_title='X', yaxis_title='Y')

# Show plot
fig.show()
'''
        print(f"3️⃣  IA generó código de {len(example_ia_code)} caracteres")
        
        if self.test_current_infrastructure().get("Models-Europe vLLM (8082)", {}).get("available", False):
            print("   ✅ Models-Europe disponible para generar código")
        else:
            print("   ⚠️  Models-Europe no disponible - simulando generación")
        
        print("4️⃣  Sistema decide usar E2B para ejecutar código de forma segura")
        
        if self.test_e2b_integration_availability().get("e2b_available", False):
            print("   ✅ Backend E2B disponible para crear sandbox")
            print("5️⃣  Backend llama a API E2B para crear sandbox")
            print("6️⃣  Código se inyecta en sandbox remota")
            print("7️⃣  Sandbox ejecuta código y genera resultados")
            print("8️⃣  Resultados retornan al backend")
            print("9️⃣  Backend procesa resultados y los envía al frontend")
            print("🔟  Frontend visualiza los resultados")
        else:
            print("   ❌ Backend E2B no disponible - flujo E2B no operativo")
            print("   (Pero flujo de IA normal puede funcionar si models-europe está disponible)")
    
    def run_verification(self) -> Dict[str, Any]:
        """
        Ejecutar la verificación completa
        """
        print("🧪 VERIFICACIÓN DEL FLUJO E2B-IA - SISTEMA ACTUAL")
        print("=" * 60)
        
        flow_analysis = self.test_actual_flow_verification()
        self.generate_test_code_and_verify_flow()
        
        # Resumen
        print(f"\n📋 RESUMEN VERIFICACIÓN:")
        print("=" * 60)
        
        has_basic_ia = (flow_analysis.get('frontend_access', False) and 
                       flow_analysis.get('gateway_available', False))
        
        has_models = flow_analysis.get('models_available', False)
        has_e2b = flow_analysis.get('e2b_available', False)
        
        print(f"   🤖 Flujo Básico de IA: {'✅ OPERATIVO' if has_basic_ia else '❌ PARCIAL'}")
        print(f"   🧠 Modelos Disponibles: {'✅ SÍ' if has_models else '❌ NO'}")
        print(f"   🛡️  Flujo E2B-Sandbox: {'✅ OPERATIVO' if has_e2b else '❌ NO'}")
        print(f"   🌐 Frontend-Backend: {'✅ CONECTADO' if flow_analysis.get('frontend_access', False) else '❌ DESCONECTADO'}")
        
        overall_status = "🟢 FUNCIONAL" if (has_basic_ia or has_models) else "🔴 NO FUNCIONAL"
        print(f"\n   🎯 ESTADO GENERAL: {overall_status}")
        
        return {
            "overall_status": overall_status,
            "flow_analysis": flow_analysis,
            "has_basic_ia": has_basic_ia,
            "has_models": has_models,
            "has_e2b": has_e2b
        }

def main():
    print("🔄 Verificación del Flujo de Integración E2B-IA")
    print("Sistema actual: Frontend → Services → Models-Europe/IA/E2B → Results")
    print()
    
    tester = VerifiedE2BIAFlowTest()
    results = tester.run_verification()
    
    print(f"\n✅ VERIFICACIÓN COMPLETADA")
    
    if results["has_basic_ia"]:
        print("   El sistema puede procesar solicitudes de IA básicas")
        if results["has_models"]:
            print("   Con acceso a modelos de IA en models-europe")
        if results["has_e2b"]:
            print("   Con capacidad para ejecutar código en sandbox E2B")
    else:
        print("   Se necesitan servicios adicionales para completar el flujo")

if __name__ == "__main__":
    main()