#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de verificación completa de servicios en las VMs de Capibara6
Verifica conectividad, servicios activos y genera configuración para el frontend
"""

import subprocess
import json
import sys
import socket
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configuración de VMs
VMS_CONFIG = {
    "bounty2": {
        "zone": "europe-west4-a",
        "project": "mamba-001",
        "services": {
            "ollama": {"port": 11434, "path": "/api/tags"},
            "backend": {"port": 5001, "path": "/api/health"},
            "backend_alt": {"port": 5000, "path": "/api/health"},
        },
        "description": "Ollama con modelos (gpt-oss-20B, mixtral, phi-mini3)"
    },
    "rag3": {
        "zone": "europe-west2-c",
        "project": "mamba-001",
        "services": {
            "rag_api": {"port": 8000, "path": "/health"},
            "postgres": {"port": 5432, "path": None},
        },
        "description": "Sistema de base de datos RAG"
    },
    "gpt-oss-20b": {
        "zone": "europe-southwest1-b",
        "project": "mamba-001",
        "services": {
            "bridge": {"port": 5000, "path": "/api/health"},
            "tts": {"port": 5002, "path": "/api/tts/voices"},
            "mcp": {"port": 5003, "path": "/api/mcp/status"},
            "mcp_alt": {"port": 5010, "path": "/api/mcp/status"},
            "n8n": {"port": 5678, "path": "/healthz"},
            "model": {"port": 8080, "path": "/health"},
        },
        "description": "Servicios TTS, MCP, N8n y Bridge"
    }
}

class Colors:
    """Colores para output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header(text: str):
    """Imprime un encabezado formateado"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")

def print_success(text: str):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅{Colors.NC} {text}")

def print_error(text: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌{Colors.NC} {text}")

def print_warning(text: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️{Colors.NC} {text}")

def print_info(text: str):
    """Imprime mensaje informativo"""
    print(f"{Colors.CYAN}ℹ️{Colors.NC} {text}")

def get_vm_ip(vm_name: str, zone: str, project: str) -> Optional[str]:
    """Obtiene la IP pública de una VM"""
    try:
        # Intentar obtener IP pública
        result = subprocess.run(
            [
                "gcloud", "compute", "instances", "describe", vm_name,
                "--zone", zone,
                "--project", project,
                "--format", "value(networkInterfaces[0].accessConfigs[0].natIP)"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        
        # Si no hay IP pública, intentar IP interna
        result = subprocess.run(
            [
                "gcloud", "compute", "instances", "describe", vm_name,
                "--zone", zone,
                "--project", project,
                "--format", "value(networkInterfaces[0].networkIP)"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
            
    except Exception as e:
        print_error(f"Error obteniendo IP de {vm_name}: {e}")
    
    return None

def check_port(ip: str, port: int, timeout: int = 3) -> bool:
    """Verifica si un puerto está abierto"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_http_service(ip: str, port: int, path: str = "/", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """Verifica si un servicio HTTP está respondiendo"""
    try:
        url = f"http://{ip}:{port}{path}"
        response = requests.get(url, timeout=timeout)
        return True, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except Exception as e:
        return False, str(e)

def verify_vm_services(vm_name: str, ip: str, services: Dict) -> Dict:
    """Verifica todos los servicios de una VM"""
    results = {
        "vm": vm_name,
        "ip": ip,
        "services": {},
        "all_ok": True
    }
    
    print_info(f"Verificando servicios en {vm_name} ({ip})...")
    
    for service_name, service_config in services.items():
        port = service_config["port"]
        path = service_config.get("path", "/")
        
        # Verificar puerto
        port_open = check_port(ip, port)
        
        if port_open:
            # Si hay path, verificar HTTP
            if path:
                http_ok, http_msg = check_http_service(ip, port, path)
                if http_ok:
                    print_success(f"{service_name} ({ip}:{port}{path}) - ACTIVO")
                    results["services"][service_name] = {
                        "status": "active",
                        "port": port,
                        "http_status": http_msg
                    }
                else:
                    print_warning(f"{service_name} ({ip}:{port}) - Puerto abierto pero HTTP no responde: {http_msg}")
                    results["services"][service_name] = {
                        "status": "port_open",
                        "port": port,
                        "http_error": http_msg
                    }
                    results["all_ok"] = False
            else:
                print_success(f"{service_name} ({ip}:{port}) - Puerto abierto")
                results["services"][service_name] = {
                    "status": "port_open",
                    "port": port
                }
        else:
            print_error(f"{service_name} ({ip}:{port}) - INACTIVO")
            results["services"][service_name] = {
                "status": "inactive",
                "port": port
            }
            results["all_ok"] = False
    
    return results

def check_vm_connectivity(vm1_ip: str, vm2_ip: str) -> bool:
    """Verifica conectividad entre dos VMs"""
    try:
        # Usar ping para verificar conectividad básica
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", vm2_ip],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

def generate_frontend_config(vm_ips: Dict[str, str]) -> str:
    """Genera configuración para el frontend"""
    config = f"""// Configuración automática para desarrollo local
// Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// IPs de las VMs de Google Cloud

const VM_CONFIG = {{
    // VM bounty2 - Ollama con modelos
    BOUNTY2_IP: '{vm_ips.get("bounty2", "N/A")}',
    OLLAMA_URL: 'http://{vm_ips.get("bounty2", "N/A")}:11434',
    BACKEND_API_URL: 'http://{vm_ips.get("bounty2", "N/A")}:5001',
    
    // VM rag3 - Base de datos RAG
    RAG3_IP: '{vm_ips.get("rag3", "N/A")}',
    RAG_API_URL: 'http://{vm_ips.get("rag3", "N/A")}:8000',
    
    // VM gpt-oss-20b - Servicios TTS, MCP, N8n, Bridge
    GPT_OSS_IP: '{vm_ips.get("gpt-oss-20b", "N/A")}',
    TTS_URL: 'http://{vm_ips.get("gpt-oss-20b", "N/A")}:5002',
    MCP_URL: 'http://{vm_ips.get("gpt-oss-20b", "N/A")}:5003',
    N8N_URL: 'http://{vm_ips.get("gpt-oss-20b", "N/A")}:5678',
    BRIDGE_URL: 'http://{vm_ips.get("gpt-oss-20b", "N/A")}:5000',
}};

// Configuración del chatbot para desarrollo local
const CHATBOT_CONFIG = {{
    BACKEND_URL: window.location.hostname === 'localhost'
        ? VM_CONFIG.BACKEND_API_URL  // Usar VM bounty2 cuando está en localhost
        : 'https://www.capibara6.com',
    
    ENDPOINTS: {{
        CHAT: '/api/chat',
        CHAT_STREAM: '/api/chat/stream',
        SAVE_CONVERSATION: '/api/save-conversation',
        SAVE_LEAD: '/api/save-lead',
        HEALTH: '/api/health',
        MCP_STATUS: '/api/mcp/status',
        MCP_TOOLS_CALL: '/api/mcp/tools/call',
        MCP_ANALYZE: '/api/mcp/analyze',
        TTS_SPEAK: '/api/tts/speak',
        TTS_VOICES: '/api/tts/voices',
        MODELS: '/api/models',
    }},
    
    MODEL_CONFIG: {{
        max_tokens: 200,
        temperature: 0.7,
        model_name: 'gpt-oss-20b',
        timeout: 120000
    }}
}};

console.log('🔧 Configuración de VMs cargada');
console.log('🔗 Backend URL:', CHATBOT_CONFIG.BACKEND_URL);
"""
    return config

def main():
    """Función principal"""
    print_header("🔍 Verificación de Servicios - Capibara6 VMs")
    
    # Obtener IPs de todas las VMs
    print_header("📋 Obteniendo IPs de las VMs")
    vm_ips = {}
    
    for vm_name, config in VMS_CONFIG.items():
        ip = get_vm_ip(vm_name, config["zone"], config["project"])
        if ip:
            vm_ips[vm_name] = ip
            print_success(f"{vm_name} ({config['zone']}): {Colors.YELLOW}{ip}{Colors.NC}")
            print_info(f"  Descripción: {config['description']}")
        else:
            print_error(f"No se pudo obtener IP para {vm_name}")
            vm_ips[vm_name] = "N/A"
    
    # Verificar servicios en cada VM
    print_header("🔌 Verificando Servicios")
    all_results = {}
    
    for vm_name, config in VMS_CONFIG.items():
        ip = vm_ips.get(vm_name)
        if ip and ip != "N/A":
            results = verify_vm_services(vm_name, ip, config["services"])
            all_results[vm_name] = results
        else:
            print_warning(f"Saltando verificación de {vm_name} - IP no disponible")
    
    # Verificar conectividad entre VMs
    print_header("🌐 Verificando Conectividad entre VMs")
    
    if vm_ips.get("bounty2") and vm_ips.get("rag3"):
        if check_vm_connectivity(vm_ips["bounty2"], vm_ips["rag3"]):
            print_success("bounty2 → rag3: Conectividad OK")
        else:
            print_error("bounty2 → rag3: Sin conectividad")
    
    if vm_ips.get("bounty2") and vm_ips.get("gpt-oss-20b"):
        if check_vm_connectivity(vm_ips["bounty2"], vm_ips["gpt-oss-20b"]):
            print_success("bounty2 → gpt-oss-20b: Conectividad OK")
        else:
            print_error("bounty2 → gpt-oss-20b: Sin conectividad")
    
    if vm_ips.get("rag3") and vm_ips.get("gpt-oss-20b"):
        if check_vm_connectivity(vm_ips["rag3"], vm_ips["gpt-oss-20b"]):
            print_success("rag3 → gpt-oss-20b: Conectividad OK")
        else:
            print_error("rag3 → gpt-oss-20b: Sin conectividad")
    
    # Generar resumen
    print_header("📝 Resumen")
    
    print(f"{Colors.YELLOW}IPs de las VMs:{Colors.NC}")
    for vm_name, ip in vm_ips.items():
        status = "✅" if ip != "N/A" else "❌"
        print(f"  {status} {vm_name}: {ip}")
    
    print(f"\n{Colors.YELLOW}Configuración recomendada para frontend local:{Colors.NC}")
    if vm_ips.get("bounty2") and vm_ips["bounty2"] != "N/A":
        print(f"  • Backend (Ollama): http://{vm_ips['bounty2']}:11434")
        print(f"  • Backend API: http://{vm_ips['bounty2']}:5001")
    if vm_ips.get("rag3") and vm_ips["rag3"] != "N/A":
        print(f"  • RAG API: http://{vm_ips['rag3']}:8000")
    if vm_ips.get("gpt-oss-20b") and vm_ips["gpt-oss-20b"] != "N/A":
        print(f"  • TTS: http://{vm_ips['gpt-oss-20b']}:5002")
        print(f"  • MCP: http://{vm_ips['gpt-oss-20b']}:5003")
        print(f"  • N8n: http://{vm_ips['gpt-oss-20b']}:5678")
        print(f"  • Bridge: http://{vm_ips['gpt-oss-20b']}:5000")
    
    # Generar archivo de configuración
    config_content = generate_frontend_config(vm_ips)
    config_file = "web/config-vm-auto.js"
    
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)
        print_success(f"Configuración guardada en {config_file}")
    except Exception as e:
        print_error(f"No se pudo guardar configuración: {e}")
    
    # Guardar resultados en JSON
    results_file = "vm_verification_results.json"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "vm_ips": vm_ips,
                "results": all_results
            }, f, indent=2)
        print_success(f"Resultados guardados en {results_file}")
    except Exception as e:
        print_error(f"No se pudo guardar resultados: {e}")
    
    print_header("✅ Verificación Completada")
    
    # Determinar estado general
    all_services_ok = all(
        result.get("all_ok", False)
        for result in all_results.values()
    )
    
    if all_services_ok:
        print_success("Todos los servicios están activos y funcionando correctamente")
        return 0
    else:
        print_warning("Algunos servicios no están disponibles. Revisa los resultados arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

