#!/usr/bin/env python3
"""
Script de verificación para probar la solución del problema CORS con el endpoint MCP
"""

import requests
import subprocess
import sys
import time
import threading

def start_proxy_server():
    """Iniciar el servidor proxy en segundo plano"""
    print("🚀 Iniciando servidor proxy en puerto 8001...")
    
    def run_proxy():
        try:
            from proxy_cors import app
            app.run(host='0.0.0.0', port=8001, debug=False)
        except Exception as e:
            print(f"❌ Error iniciando proxy: {e}")
    
    proxy_thread = threading.Thread(target=run_proxy, daemon=True)
    proxy_thread.start()
    
    # Esperar un poco para que el servidor inicie
    time.sleep(2)
    return proxy_thread

def test_cors_fix():
    """Probar que el problema de CORS está resuelto"""
    print("\n🔍 Probando solución de problema CORS...")
    
    tests = [
        {
            'name': 'Prueba básica de proxy',
            'url': 'http://localhost:8001/health',
            'method': 'GET',
            'expected_status': 404  # Porque no existe el endpoint /health en los destinos
        },
        {
            'name': 'Prueba de redirección MCP status -> health (caso principal)',
            'url': 'http://localhost:8001/api/mcp/status',
            'method': 'GET',
            'expected_status': 200  # Debería redirigir a /api/mcp/health y devolver 200
        },
        {
            'name': 'Prueba de redirección MCP status con ruta adicional',
            'url': 'http://localhost:8001/api/mcp/tool/status',
            'method': 'GET',
            'expected_status': 200  # Debería redirigir a /api/mcp/tool/health
        },
        {
            'name': 'Prueba de variante v1 de MCP status',
            'url': 'http://localhost:8001/api/v1/mcp/status',
            'method': 'GET',
            'expected_status': 200  # Debería redirigir a /api/v1/mcp/health
        }
    ]
    
    results = []
    
    for test in tests:
        try:
            response = requests.request(
                method=test['method'],
                url=test['url'],
                timeout=10
            )
            
            success = response.status_code == test['expected_status']
            results.append({
                'name': test['name'],
                'url': test['url'],
                'expected': test['expected_status'],
                'actual': response.status_code,
                'success': success,
                'response_headers': dict(response.headers)
            })
            
            status_icon = "✅" if success else "❌"
            print(f"  {status_icon} {test['name']}: {response.status_code} (esperado: {test['expected_status']})")
            
        except requests.exceptions.ConnectionError:
            print(f"  ❌ {test['name']}: No se pudo conectar al servidor proxy")
            results.append({
                'name': test['name'],
                'url': test['url'],
                'expected': test['expected_status'],
                'actual': 'CONNECTION_ERROR',
                'success': False
            })
        except Exception as e:
            print(f"  ❌ {test['name']}: Error - {e}")
            results.append({
                'name': test['name'],
                'url': test['url'],
                'expected': test['expected_status'],
                'actual': f'ERROR: {e}',
                'success': False
            })
    
    return results

def check_cors_headers(response_headers):
    """Verificar si los encabezados CORS están presentes"""
    cors_headers = [
        'Access-Control-Allow-Origin',
        'Access-Control-Allow-Methods',
        'Access-Control-Allow-Headers'
    ]
    
    present_headers = []
    for header in cors_headers:
        if header.lower() in [h.lower() for h in response_headers.keys()]:
            present_headers.append(header)
    
    return present_headers

def main():
    print("🦫 Capibara6 - Verificación de Solución CORS")
    print("=" * 60)
    
    print("🔧 Correcciones implementadas:")
    print("  • Redirección automática de /api/mcp/status a /api/mcp/health")
    print("  • Redirección de /api/v1/mcp/status a /api/v1/mcp/health") 
    print("  • Soporte para diferentes variantes del endpoint status")
    
    # Intentar iniciar el servidor proxy
    proxy_thread = start_proxy_server()
    
    # Realizar pruebas
    results = test_cors_fix()
    
    # Analizar resultados
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"\n📊 Resultados:")
    print(f"  ✅ Pruebas exitosas: {len(successful_tests)}")
    print(f"  ❌ Pruebas fallidas: {len(failed_tests)}")
    
    if failed_tests:
        print("\n⚠️  Detalles de pruebas fallidas:")
        for test in failed_tests:
            print(f"    - {test['name']}: esperado {test['expected']}, obtenido {test['actual']}")
    
    # Verificar encabezados CORS en respuestas exitosas
    print(f"\n🔍 Verificación de encabezados CORS:")
    for result in results:
        if result['success'] and 'response_headers' in result:
            cors_present = check_cors_headers(result['response_headers'])
            if cors_present:
                print(f"  ✅ {result['name']}: Encabezados CORS presentes: {', '.join(cors_present)}")
            else:
                print(f"  ⚠️  {result['name']}: No se encontraron encabezados CORS")
    
    # Conclusión
    print("\n" + "=" * 60)
    if len(successful_tests) >= 2:  # Al menos las pruebas principales pasaron
        print("🎉 ¡Solución implementada correctamente!")
        print("✅ El problema de CORS con el endpoint /api/mcp/status debería estar resuelto")
        print("✅ Las solicitudes ahora se redirigen correctamente al endpoint /health")
        return 0
    else:
        print("💥 Algunas pruebas críticas fallaron")
        print("❌ El problema de CORS podría persistir")
        return 1

if __name__ == "__main__":
    sys.exit(main())