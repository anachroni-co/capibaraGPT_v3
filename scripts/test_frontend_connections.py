#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar conexiones del frontend con todos los servicios
Prueba cada endpoint y genera un reporte de estado
"""

import json
import sys
import time
import requests
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin

# Timeout para requests
REQUEST_TIMEOUT = 10

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Imprime un encabezado formateado"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str):
    """Imprime mensaje informativo"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


def load_vm_config() -> Optional[Dict]:
    """Carga la configuración de VMs desde archivo JSON"""
    try:
        with open('vm_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print_warning("vm_config.json no encontrado. Intentando obtener información de VMs...")
        return None
    except json.JSONDecodeError as e:
        print_error(f"Error parseando vm_config.json: {e}")
        return None


def get_vm_ips_from_config(config: Dict) -> Dict[str, str]:
    """Extrae las IPs de la configuración"""
    ips = {
        "bounty2_external": "",
        "bounty2_internal": "",
        "rag3_external": "",
        "rag3_internal": "",
        "gptoss_external": "",
        "gptoss_internal": ""
    }
    
    if config and "vms" in config:
        vms = config["vms"]
        
        if "bounty2" in vms:
            ips["bounty2_external"] = vms["bounty2"].get("ip_external", "")
            ips["bounty2_internal"] = vms["bounty2"].get("ip_internal", "")
        
        if "rag3" in vms:
            ips["rag3_external"] = vms["rag3"].get("ip_external", "")
            ips["rag3_internal"] = vms["rag3"].get("ip_internal", "")
        
        if "gpt-oss-20b" in vms:
            ips["gptoss_external"] = vms["gpt-oss-20b"].get("ip_external", "")
            ips["gptoss_internal"] = vms["gpt-oss-20b"].get("ip_internal", "")
    
    return ips


def get_ips_from_config_js() -> Dict[str, str]:
    """Lee las IPs desde web/config.js"""
    ips = {
        "bounty2_external": "",
        "gptoss_external": "",
        "rag3_external": ""
    }
    
    try:
        with open('web/config.js', 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Buscar BOUNTY2_EXTERNAL
            import re
            match = re.search(r"BOUNTY2_EXTERNAL:\s*['\"]([^'\"]+)['\"]", content)
            if match:
                ips["bounty2_external"] = match.group(1)
            
            match = re.search(r"GPTOSS_EXTERNAL:\s*['\"]([^'\"]+)['\"]", content)
            if match:
                ips["gptoss_external"] = match.group(1)
            
            match = re.search(r"RAG3_EXTERNAL:\s*['\"]([^'\"]+)['\"]", content)
            if match:
                ips["rag3_external"] = match.group(1)
    except FileNotFoundError:
        print_warning("web/config.js no encontrado")
    except Exception as e:
        print_warning(f"Error leyendo web/config.js: {e}")
    
    return ips


def test_endpoint(url: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Tuple[bool, str, Optional[int]]:
    """
    Prueba un endpoint HTTP
    
    Returns:
        (success, message, status_code)
    """
    try:
        if method == "GET":
            response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT, headers=headers)
        elif method == "HEAD":
            response = requests.head(url, timeout=REQUEST_TIMEOUT, headers=headers)
        else:
            return False, f"Método {method} no soportado", None
        
        if response.status_code < 400:
            return True, f"OK (Status: {response.status_code})", response.status_code
        else:
            return False, f"Error HTTP {response.status_code}", response.status_code
    
    except requests.exceptions.Timeout:
        return False, "Timeout (servicio no responde)", None
    except requests.exceptions.ConnectionError:
        return False, "Connection Error (servicio no accesible)", None
    except requests.exceptions.RequestException as e:
        return False, f"Error: {str(e)}", None


def test_service(name: str, base_url: str, endpoints: List[Dict]) -> Dict:
    """
    Prueba múltiples endpoints de un servicio
    
    Args:
        name: Nombre del servicio
        base_url: URL base del servicio
        endpoints: Lista de endpoints a probar [{"path": "/health", "method": "GET", "description": "Health check"}]
    
    Returns:
        Dict con resultados
    """
    print_info(f"Probando servicio: {name}")
    print(f"   URL Base: {base_url}")
    
    results = {
        "service": name,
        "base_url": base_url,
        "endpoints": [],
        "success_count": 0,
        "total_count": len(endpoints)
    }
    
    for endpoint in endpoints:
        path = endpoint.get("path", "")
        method = endpoint.get("method", "GET")
        description = endpoint.get("description", "")
        data = endpoint.get("data")
        headers = endpoint.get("headers", {})
        
        url = urljoin(base_url, path)
        
        print(f"   → {method} {path} {f'({description})' if description else ''}...", end=" ")
        
        success, message, status_code = test_endpoint(url, method, data, headers)
        
        endpoint_result = {
            "path": path,
            "method": method,
            "url": url,
            "success": success,
            "message": message,
            "status_code": status_code
        }
        
        results["endpoints"].append(endpoint_result)
        
        if success:
            print_success(message)
            results["success_count"] += 1
        else:
            print_error(message)
        
        time.sleep(0.5)  # Pequeña pausa entre requests
    
    return results


def main():
    print_header("Verificación de Conexiones Frontend - Capibara6")
    
    # Cargar configuración
    config = load_vm_config()
    
    # Obtener IPs
    if config:
        vm_ips = get_vm_ips_from_config(config)
        print_info("IPs obtenidas desde vm_config.json")
    else:
        vm_ips = get_ips_from_config_js()
        print_info("IPs obtenidas desde web/config.js")
        print_warning("Se recomienda ejecutar: python3 scripts/get_vm_info.py para obtener IPs completas")
    
    # Usar IPs externas para pruebas desde local
    bounty2_ip = vm_ips.get("bounty2_external", "")
    rag3_ip = vm_ips.get("rag3_external", "")
    gptoss_ip = vm_ips.get("gptoss_external", "")
    
    if not bounty2_ip:
        print_error("No se encontró IP de Bounty2. Por favor actualiza web/config.js o ejecuta scripts/get_vm_info.py")
        sys.exit(1)
    
    if not gptoss_ip:
        print_error("No se encontró IP de gpt-oss-20b. Por favor actualiza web/config.js o ejecuta scripts/get_vm_info.py")
        sys.exit(1)
    
    print(f"\n{Colors.BOLD}IPs configuradas:{Colors.END}")
    print(f"  Bounty2: {bounty2_ip}")
    print(f"  rag3: {rag3_ip if rag3_ip else 'NO CONFIGURADA'}")
    print(f"  gpt-oss-20b: {gptoss_ip}")
    
    # Definir endpoints a probar según la arquitectura
    services = []
    
    # 1. Bounty2 - Ollama
    services.append({
        "name": "Ollama (Bounty2)",
        "base_url": f"http://{bounty2_ip}:11434",
        "endpoints": [
            {"path": "/api/tags", "method": "GET", "description": "Listar modelos"},
            {"path": "/api/version", "method": "GET", "description": "Versión de Ollama"},
        ]
    })
    
    # 2. Bounty2 - Backend Flask
    services.append({
        "name": "Backend Flask (Bounty2)",
        "base_url": f"http://{bounty2_ip}:5001",
        "endpoints": [
            {"path": "/api/health", "method": "GET", "description": "Health check"},
            {"path": "/api/models", "method": "GET", "description": "Listar modelos disponibles"},
            {"path": "/api/chat", "method": "POST", "description": "Chat endpoint", 
             "data": {"message": "test", "model": "gpt-oss-20b"}},
        ]
    })
    
    # 3. rag3 - RAG API
    if rag3_ip:
        services.append({
            "name": "RAG API (rag3)",
            "base_url": f"http://{rag3_ip}:8000",
            "endpoints": [
                {"path": "/health", "method": "GET", "description": "Health check"},
                {"path": "/api/search/semantic", "method": "POST", "description": "Búsqueda semántica",
                 "data": {"query": "test", "n_results": 3}},
            ]
        })
    else:
        print_warning("rag3 no configurado, omitiendo pruebas")
    
    # 4. gpt-oss-20b - TTS
    services.append({
        "name": "TTS (gpt-oss-20b)",
        "base_url": f"http://{gptoss_ip}:5002",
        "endpoints": [
            {"path": "/api/tts/voices", "method": "GET", "description": "Listar voces disponibles"},
            {"path": "/api/tts/speak", "method": "POST", "description": "Síntesis de voz",
             "data": {"text": "test", "voice": "default"}},
        ]
    })
    
    # 5. gpt-oss-20b - MCP
    services.append({
        "name": "MCP Server (gpt-oss-20b)",
        "base_url": f"http://{gptoss_ip}:5003",
        "endpoints": [
            {"path": "/api/mcp/status", "method": "GET", "description": "Estado del MCP"},
            {"path": "/api/mcp/analyze", "method": "POST", "description": "Análisis MCP",
             "data": {"query": "test"}},
        ]
    })
    
    # 6. gpt-oss-20b - MCP Alternativo
    services.append({
        "name": "MCP Server Alt (gpt-oss-20b)",
        "base_url": f"http://{gptoss_ip}:5010",
        "endpoints": [
            {"path": "/api/mcp/status", "method": "GET", "description": "Estado del MCP"},
        ]
    })
    
    # 7. gpt-oss-20b - N8n
    services.append({
        "name": "N8n (gpt-oss-20b)",
        "base_url": f"http://{gptoss_ip}:5678",
        "endpoints": [
            {"path": "/healthz", "method": "GET", "description": "Health check"},
        ]
    })
    
    # 8. gpt-oss-20b - Bridge
    services.append({
        "name": "Bridge (gpt-oss-20b)",
        "base_url": f"http://{gptoss_ip}:5000",
        "endpoints": [
            {"path": "/api/health", "method": "GET", "description": "Health check"},
        ]
    })
    
    # Ejecutar pruebas
    all_results = []
    
    for service_config in services:
        result = test_service(
            service_config["name"],
            service_config["base_url"],
            service_config["endpoints"]
        )
        all_results.append(result)
        print()  # Línea en blanco entre servicios
    
    # Resumen
    print_header("Resumen de Pruebas")
    
    total_endpoints = sum(r["total_count"] for r in all_results)
    successful_endpoints = sum(r["success_count"] for r in all_results)
    failed_endpoints = total_endpoints - successful_endpoints
    
    print(f"{Colors.BOLD}Total de endpoints probados: {total_endpoints}{Colors.END}")
    print_success(f"Exitosos: {successful_endpoints}")
    if failed_endpoints > 0:
        print_error(f"Fallidos: {failed_endpoints}")
    else:
        print_success("¡Todos los endpoints están funcionando!")
    
    print()
    
    # Detalles por servicio
    print(f"{Colors.BOLD}Detalles por servicio:{Colors.END}\n")
    for result in all_results:
        service_name = result["service"]
        success_count = result["success_count"]
        total_count = result["total_count"]
        
        if success_count == total_count:
            print_success(f"{service_name}: {success_count}/{total_count} endpoints OK")
        elif success_count > 0:
            print_warning(f"{service_name}: {success_count}/{total_count} endpoints OK")
        else:
            print_error(f"{service_name}: 0/{total_count} endpoints OK")
        
        # Mostrar endpoints fallidos
        for endpoint in result["endpoints"]:
            if not endpoint["success"]:
                print(f"   ❌ {endpoint['method']} {endpoint['path']}: {endpoint['message']}")
    
    # Guardar resultados en archivo JSON
    output_file = "frontend_connection_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ips": {
                "bounty2": bounty2_ip,
                "rag3": rag3_ip,
                "gptoss": gptoss_ip
            },
            "summary": {
                "total_endpoints": total_endpoints,
                "successful_endpoints": successful_endpoints,
                "failed_endpoints": failed_endpoints
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{Colors.BLUE}📄 Resultados guardados en: {output_file}{Colors.END}")
    
    # Recomendaciones
    if failed_endpoints > 0:
        print()
        print_header("Recomendaciones")
        print("1. Verifica que los servicios están corriendo en cada VM:")
        print("   bash scripts/check_services_on_vm.sh")
        print()
        print("2. Verifica las reglas de firewall:")
        print("   gcloud compute firewall-rules list --project=mamba-001")
        print()
        print("3. Verifica que las IPs en web/config.js son correctas")
        print()
        print("4. Si los servicios están en IPs internas, asegúrate de usar IPs externas")
        print("   para acceso desde tu portátil")
    
    return 0 if failed_endpoints == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pruebas canceladas por el usuario")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

