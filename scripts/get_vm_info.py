#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para obtener información de las VMs de GCloud y verificar conexiones
"""

import subprocess
import json
import sys
from typing import Dict, Optional, List

PROJECT = "mamba-001"
VMS = {
    "bounty2": {
        "zone": "europe-west4-a",
        "description": "Ollama con modelos gpt-oss-20B, mixtral, phi-mini3"
    },
    "rag3": {
        "zone": "europe-west2-c",
        "description": "Sistema de base de datos RAG"
    },
    "gpt-oss-20b": {
        "zone": "europe-southwest1-b",
        "description": "TTS, MCP, N8n y Bridge"
    }
}


def run_gcloud_command(command: List[str]) -> Optional[str]:
    """Ejecuta un comando de gcloud y retorna la salida"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error ejecutando comando: {' '.join(command)}")
        print(f"   Error: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout ejecutando: {' '.join(command)}")
        return None
    except FileNotFoundError:
        print("❌ gcloud CLI no encontrado. Por favor instala Google Cloud SDK")
        return None


def get_vm_info(vm_name: str, zone: str) -> Dict:
    """Obtiene información de una VM específica"""
    print(f"🔍 Obteniendo información de {vm_name}...")
    
    info = {
        "name": vm_name,
        "zone": zone,
        "ip_external": None,
        "ip_internal": None,
        "network": None,
        "status": None
    }
    
    # Obtener IP externa
    cmd = [
        "gcloud", "compute", "instances", "describe", vm_name,
        "--zone", zone,
        "--project", PROJECT,
        "--format", "value(networkInterfaces[0].accessConfigs[0].natIP)"
    ]
    ip_ext = run_gcloud_command(cmd)
    if ip_ext:
        info["ip_external"] = ip_ext
    
    # Obtener IP interna
    cmd = [
        "gcloud", "compute", "instances", "describe", vm_name,
        "--zone", zone,
        "--project", PROJECT,
        "--format", "value(networkInterfaces[0].networkIP)"
    ]
    ip_int = run_gcloud_command(cmd)
    if ip_int:
        info["ip_internal"] = ip_int
    
    # Obtener red VPC
    cmd = [
        "gcloud", "compute", "instances", "describe", vm_name,
        "--zone", zone,
        "--project", PROJECT,
        "--format", "value(networkInterfaces[0].network)"
    ]
    network = run_gcloud_command(cmd)
    if network:
        # Extraer solo el nombre de la red
        network_name = network.split('/')[-1] if '/' in network else network
        info["network"] = network_name
    
    # Obtener estado
    cmd = [
        "gcloud", "compute", "instances", "describe", vm_name,
        "--zone", zone,
        "--project", PROJECT,
        "--format", "value(status)"
    ]
    status = run_gcloud_command(cmd)
    if status:
        info["status"] = status
    
    return info


def check_firewall_rules(vm_networks: Dict[str, str]) -> Dict:
    """Verifica reglas de firewall entre VMs"""
    print("\n🔒 Verificando reglas de firewall...")
    
    # Obtener todas las reglas de firewall
    cmd = [
        "gcloud", "compute", "firewall-rules", "list",
        "--project", PROJECT,
        "--format", "json"
    ]
    
    result = run_gcloud_command(cmd)
    if not result:
        return {"error": "No se pudieron obtener reglas de firewall"}
    
    try:
        rules = json.loads(result)
        return {
            "total_rules": len(rules),
            "rules": rules
        }
    except json.JSONDecodeError:
        return {"error": "Error parseando reglas de firewall"}


def main():
    print("=" * 60)
    print("🔍 Verificación de VMs de Capibara6")
    print("=" * 60)
    print()
    
    vm_info = {}
    networks = set()
    
    # Obtener información de cada VM
    for vm_name, config in VMS.items():
        info = get_vm_info(vm_name, config["zone"])
        info["description"] = config["description"]
        vm_info[vm_name] = info
        
        if info.get("network"):
            networks.add(info["network"])
        
        print(f"  ✅ {vm_name}:")
        print(f"     - IP Externa: {info.get('ip_external', 'N/A')}")
        print(f"     - IP Interna: {info.get('ip_internal', 'N/A')}")
        print(f"     - Red VPC: {info.get('network', 'N/A')}")
        print(f"     - Estado: {info.get('status', 'N/A')}")
        print()
    
    # Verificar si están en la misma red
    print("=" * 60)
    print("🌐 Análisis de Red")
    print("=" * 60)
    
    if len(networks) == 1:
        print(f"✅ Todas las VMs están en la misma red VPC: {list(networks)[0]}")
        print("   Esto permite comunicación de alta velocidad entre VMs")
    else:
        print("⚠️  Las VMs están en diferentes redes VPC:")
        for vm_name, info in vm_info.items():
            print(f"   - {vm_name}: {info.get('network', 'N/A')}")
        print("\n   💡 Recomendación: Mover todas las VMs a la misma red VPC")
        print("      para mejor rendimiento y menor latencia")
    
    print()
    
    # Guardar configuración
    config = {
        "vms": vm_info,
        "network": {
            "same_vpc": len(networks) == 1,
            "vpc_networks": list(networks)
        }
    }
    
    # Agregar endpoints de servicios
    config["service_endpoints"] = {
        "bounty2": {
            "ollama": f"http://{vm_info['bounty2'].get('ip_internal') or vm_info['bounty2'].get('ip_external', 'N/A')}:11434",
            "backend": f"http://{vm_info['bounty2'].get('ip_internal') or vm_info['bounty2'].get('ip_external', 'N/A')}:5001"
        },
        "rag3": {
            "rag_api": f"http://{vm_info['rag3'].get('ip_internal') or vm_info['rag3'].get('ip_external', 'N/A')}:8000"
        },
        "gpt-oss-20b": {
            "tts": f"http://{vm_info['gpt-oss-20b'].get('ip_internal') or vm_info['gpt-oss-20b'].get('ip_external', 'N/A')}:5002",
            "mcp": f"http://{vm_info['gpt-oss-20b'].get('ip_internal') or vm_info['gpt-oss-20b'].get('ip_external', 'N/A')}:5003",
            "mcp_alt": f"http://{vm_info['gpt-oss-20b'].get('ip_internal') or vm_info['gpt-oss-20b'].get('ip_external', 'N/A')}:5010",
            "n8n": f"http://{vm_info['gpt-oss-20b'].get('ip_internal') or vm_info['gpt-oss-20b'].get('ip_external', 'N/A')}:5678",
            "bridge": f"http://{vm_info['gpt-oss-20b'].get('ip_internal') or vm_info['gpt-oss-20b'].get('ip_external', 'N/A')}:5000"
        }
    }
    
    # Guardar en archivo JSON
    output_file = "vm_config.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("📝 Configuración guardada")
    print("=" * 60)
    print(f"✅ Archivo generado: {output_file}")
    print()
    print("🎯 Próximos pasos:")
    print("   1. Revisar vm_config.json para verificar las IPs")
    print("   2. Verificar que los servicios estén corriendo en cada VM")
    print("   3. Configurar firewall rules si es necesario")
    print("   4. Actualizar archivos de configuración del backend y frontend")
    print()
    
    return config


if __name__ == "__main__":
    try:
        config = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

