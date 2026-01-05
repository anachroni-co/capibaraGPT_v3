#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para el conector MCP de capibara6
"""

import asyncio
import json
import requests
import time
from mcp_connector import Capibara6MCPConnector

class MCPTester:
    """Tester para el conector MCP de capibara6"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.connector = Capibara6MCPConnector()
    
    async def test_direct_connector(self):
        """Probar el conector directamente (sin servidor)"""
        print("🧪 Probando conector MCP directamente...")
        
        # Test de inicialización
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        response = await self.connector.handle_request(init_request)
        print("✅ Inicialización:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        # Test de herramientas
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = await self.connector.handle_request(tools_request)
        print("\n✅ Herramientas disponibles:")
        tools = response.get("result", {}).get("tools", [])
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Test de recursos
        resources_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        }
        
        response = await self.connector.handle_request(resources_request)
        print("\n✅ Recursos disponibles:")
        resources = response.get("result", {}).get("resources", [])
        for resource in resources:
            print(f"  - {resource['name']}: {resource['description']}")
        
        # Test de prompts
        prompts_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/list",
            "params": {}
        }
        
        response = await self.connector.handle_request(prompts_request)
        print("\n✅ Prompts disponibles:")
        prompts = response.get("result", {}).get("prompts", [])
        for prompt in prompts:
            print(f"  - {prompt['name']}: {prompt['description']}")
        
        # Test de ejecución de herramienta
        tool_call_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "analyze_document",
                "arguments": {
                    "document": "Este es un documento de prueba para el análisis con capibara6.",
                    "analysis_type": "compliance",
                    "language": "es"
                }
            }
        }
        
        response = await self.connector.handle_request(tool_call_request)
        print("\n✅ Ejecución de herramienta:")
        content = response.get("result", {}).get("content", [])
        if content:
            print(content[0].get("text", "Sin contenido"))
    
    def test_server_endpoints(self):
        """Probar endpoints del servidor MCP"""
        print("\n🌐 Probando endpoints del servidor MCP...")
        
        # Test de estado
        try:
            response = requests.get(f"{self.base_url}/api/mcp/status", timeout=10)
            if response.status_code == 200:
                print("✅ Estado del servidor:")
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            else:
                print(f"❌ Error en estado: {response.status_code}")
        except Exception as e:
            print(f"❌ Error conectando al servidor: {e}")
            return
        
        # Test de inicialización
        try:
            response = requests.post(f"{self.base_url}/api/mcp/initialize", 
                                   json={}, timeout=10)
            if response.status_code == 200:
                print("\n✅ Inicialización MCP:")
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            else:
                print(f"❌ Error en inicialización: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en inicialización: {e}")
        
        # Test de herramientas
        try:
            response = requests.get(f"{self.base_url}/api/mcp/tools/list", timeout=10)
            if response.status_code == 200:
                print("\n✅ Herramientas del servidor:")
                tools = response.json().get("result", {}).get("tools", [])
                for tool in tools:
                    print(f"  - {tool['name']}: {tool['description']}")
            else:
                print(f"❌ Error listando herramientas: {response.status_code}")
        except Exception as e:
            print(f"❌ Error listando herramientas: {e}")
        
        # Test de recursos
        try:
            response = requests.get(f"{self.base_url}/api/mcp/resources/list", timeout=10)
            if response.status_code == 200:
                print("\n✅ Recursos del servidor:")
                resources = response.json().get("result", {}).get("resources", [])
                for resource in resources:
                    print(f"  - {resource['name']}: {resource['description']}")
            else:
                print(f"❌ Error listando recursos: {response.status_code}")
        except Exception as e:
            print(f"❌ Error listando recursos: {e}")
        
        # Test de ejecución de herramienta
        try:
            tool_data = {
                "name": "analyze_document",
                "arguments": {
                    "document": "Documento de prueba para análisis con capibara6 MCP.",
                    "analysis_type": "technical",
                    "language": "es"
                }
            }
            
            response = requests.post(f"{self.base_url}/api/mcp/tools/call", 
                                   json=tool_data, timeout=30)
            if response.status_code == 200:
                print("\n✅ Ejecución de herramienta:")
                result = response.json()
                if "result" in result and "content" in result["result"]:
                    content = result["result"]["content"][0]["text"]
                    print(content[:500] + "..." if len(content) > 500 else content)
                else:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"❌ Error ejecutando herramienta: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ Error ejecutando herramienta: {e}")
        
        # Test de lectura de recurso
        try:
            resource_data = {
                "uri": "capibara6://model/info"
            }
            
            response = requests.post(f"{self.base_url}/api/mcp/resources/read", 
                                   json=resource_data, timeout=10)
            if response.status_code == 200:
                print("\n✅ Lectura de recurso:")
                result = response.json()
                if "result" in result and "contents" in result["result"]:
                    content = result["result"]["contents"][0]["text"]
                    print(json.dumps(json.loads(content), indent=2, ensure_ascii=False))
                else:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"❌ Error leyendo recurso: {response.status_code}")
        except Exception as e:
            print(f"❌ Error leyendo recurso: {e}")
        
        # Test completo
        try:
            response = requests.post(f"{self.base_url}/api/mcp/test", 
                                   json={"test_type": "full"}, timeout=30)
            if response.status_code == 200:
                print("\n✅ Test completo:")
                result = response.json()
                print(f"Estado: {result.get('status')}")
                print(f"Tipo de test: {result.get('test_type')}")
            else:
                print(f"❌ Error en test completo: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en test completo: {e}")
    
    def test_performance(self):
        """Probar rendimiento del conector"""
        print("\n⚡ Probando rendimiento...")
        
        # Test de latencia
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/api/mcp/status", timeout=5)
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            
            if response.status_code == 200:
                print(f"✅ Latencia de estado: {latency:.2f}ms")
            else:
                print(f"❌ Error en test de latencia: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en test de latencia: {e}")
        
        # Test de throughput (múltiples solicitudes)
        print("🔄 Probando throughput...")
        start_time = time.time()
        successful_requests = 0
        
        for i in range(10):
            try:
                response = requests.get(f"{self.base_url}/api/mcp/status", timeout=5)
                if response.status_code == 200:
                    successful_requests += 1
            except:
                pass
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = successful_requests / total_time
        
        print(f"✅ Throughput: {throughput:.2f} requests/sec ({successful_requests}/10 exitosas)")

async def main():
    """Función principal de testing"""
    print("🦫 capibara6 MCP Connector - Test Suite")
    print("=" * 50)
    
    tester = MCPTester()
    
    # Test directo del conector
    await tester.test_direct_connector()
    
    # Test de endpoints del servidor
    tester.test_server_endpoints()
    
    # Test de rendimiento
    tester.test_performance()
    
    print("\n" + "=" * 50)
    print("✅ Tests completados")

if __name__ == "__main__":
    asyncio.run(main())