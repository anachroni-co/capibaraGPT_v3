#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simple para verificar servicios desde el portátil local
"""

import socket
import requests
import sys
from typing import Tuple

# IPs conocidas según documentación
VMS = {
    "bounty2": {
        "ip": "34.12.166.76",
        "services": [
            {"name": "Ollama", "port": 11434, "path": "/api/tags"},
            {"name": "Backend Capibara6", "port": 5001, "path": "/api/health"},
            {"name": "Backend alternativo", "port": 5000, "path": "/api/health"},
        ]
    },
    "gpt-oss-20b": {
        "ip": "34.175.136.104",
        "services": [
            {"name": "Bridge/Main Server", "port": 5000, "path": "/api/health"},
            {"name": "TTS Server", "port": 5002, "path": "/api/tts/voices"},
            {"name": "MCP Server", "port": 5003, "path": "/api/mcp/status"},
            {"name": "MCP Server (alt)", "port": 5010, "path": "/api/mcp/status"},
            {"name": "N8n", "port": 5678, "path": "/healthz"},
            {"name": "Modelo", "port": 8080, "path": "/health"},
        ]
    },
    "rag3": {
        "ip": None,  # Por determinar
        "services": [
            {"name": "RAG API", "port": 8000, "path": "/health"},
        ]
    }
}

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

def check_http(ip: str, port: int, path: str = "/", timeout: int = 3) -> Tuple[bool, str]:
    """Verifica si un servicio HTTP responde"""
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

def main():
    print("\n" + "="*60)
    print("🔍 Verificación de Servicios desde Local")
    print("="*60 + "\n")
    
    for vm_name, vm_config in VMS.items():
        ip = vm_config["ip"]
        if not ip:
            print(f"⚠️  {vm_name}: IP no configurada")
            continue
        
        print(f"📡 Verificando {vm_name} ({ip})...")
        
        for service in vm_config["services"]:
            name = service["name"]
            port = service["port"]
            path = service.get("path", "/")
            
            # Verificar puerto
            port_open = check_port(ip, port)
            
            if port_open:
                # Verificar HTTP si hay path
                if path:
                    http_ok, http_msg = check_http(ip, port, path)
                    if http_ok:
                        print(f"  ✅ {name} ({ip}:{port}{path}) - ACTIVO - {http_msg}")
                    else:
                        print(f"  ⚠️  {name} ({ip}:{port}) - Puerto abierto pero HTTP: {http_msg}")
                else:
                    print(f"  ✅ {name} ({ip}:{port}) - Puerto abierto")
            else:
                print(f"  ❌ {name} ({ip}:{port}) - INACTIVO")
        
        print()
    
    print("="*60)
    print("📝 Resumen")
    print("="*60 + "\n")
    
    print("IPs configuradas:")
    for vm_name, vm_config in VMS.items():
        ip = vm_config["ip"] or "No configurada"
        print(f"  • {vm_name}: {ip}")
    
    print("\n✅ Verificación completada\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Verificación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

