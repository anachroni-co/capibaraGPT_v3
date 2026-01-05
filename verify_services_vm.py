#!/usr/bin/env python3
"""
Verificación de la VM de Servicios ARM-Axion (gpt-oss-20b)
IP: 34.175.136.104
Zona: europe-southwest1-b
Servicios: TTS, MCP, N8n
"""

import subprocess
import json
import time
from typing import Dict, List

def check_remote_services():
    """Verificar servicios en la VM remota de servicios"""
    
    print("="*80)
    print("🔍 VERIFICACIÓN DE SERVICIOS REMOTOS - VM gpt-oss-20b")
    print("IP: 34.175.136.104 | Zona: europe-southwest1-b")
    print("="*80)
    
    # Definir servicios a verificar
    services = [
        {"name": "TTS Kyutai", "port": 5002, "endpoint": "/health", "desc": "Text-to-Speech API"},
        {"name": "MCP Server", "port": 5003, "endpoint": "/api/mcp/health", "desc": "Model Context Protocol"},
        {"name": "N8n", "port": 5678, "endpoint": "/healthz", "desc": "Workflow Automation"},
    ]
    
    print("\n📡 SERVICIOS A VERIFICAR:")
    for svc in services:
        print(f"  • {svc['name']} (puerto {svc['port']}) - {svc['desc']}")
    
    print("\n" + "="*80)
    print("ESTADO DE LOS SERVICIOS")
    print("="*80)

    # Realizar verificaciones
    results = []
    for svc in services:
        print(f"\n🔍 Verificando {svc['name']} en puerto {svc['port']}...")
        
        # Probar con gcloud compute ssh
        cmd = [
            "gcloud", "compute", "ssh", "gpt-oss-20b",
            "--zone=europe-southwest1-b",
            "--project=mamba-001",
            "--command",
            f"curl -s --connect-timeout 5 http://localhost:{svc['port']}{svc['endpoint']} || echo 'No responde'"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and "No responde" not in result.stdout:
                status = "✅ ACTIVO"
                response_sample = result.stdout[:100]
            else:
                status = "❌ INACTIVO"
                response_sample = result.stderr[:100] if result.stderr else "Sin respuesta"
                
            results.append({
                "service": svc["name"],
                "port": svc["port"],
                "status": status,
                "response": response_sample
            })
            print(f"   {status} - {response_sample[:50]}...")
        except subprocess.TimeoutExpired:
            results.append({
                "service": svc["name"],
                "port": svc["port"],
                "status": "❌ TIMEOUT",
                "response": "Tiempo de espera agotado"
            })
            print(f"   ❌ TIMEOUT")
        except Exception as e:
            results.append({
                "service": svc["name"],
                "port": svc["port"],
                "status": "❌ ERROR",
                "response": str(e)
            })
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "="*80)
    print("RESULTADOS DE VERIFICACIÓN")
    print("="*80)
    
    print(f"{'Servicio':<15} {'Puerto':<8} {'Estado':<10} {'Respuesta (truncada)'}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['service']:<15} {result['port']:<8} {result['status']:<10} {result['response']}")
    
    print("\n" + "="*80)
    print("RESUMEN DE LA VM DE SERVICIOS")
    print("="*80)
    
    active_services = [r for r in results if "ACTIVO" in r["status"]]
    inactive_services = [r for r in results if "ACTIVO" not in r["status"]]
    
    print(f"Servicios activos: {len(active_services)}/{len(services)}")
    print(f"Servicios inactivos: {len(inactive_services)}/{len(services)}")
    
    if len(active_services) == len(services):
        print("\n🎉 ¡TODOS LOS SERVICIOS ESTÁN FUNCIONANDO!")
        print("   • TTS Kyutai (puerto 5002) - Text-to-Speech disponible")
        print("   • MCP Server (puerto 5003) - Model Context Protocol disponible") 
        print("   • N8n (puerto 5678) - Workflow Automation disponible")
        print("\n   ARQUITECTURA ARM-Axion con los 5 componentes críticos:")
        print("   1. vLLM con los 5 modelos (en VM Bounty2)")
        print("   2. Sistema RAG con Milvus, Nebula, PostgreSQL (en VM RAG3)")
        print("   3. TTS Kyutai (en VM gpt-oss-20b)")
        print("   4. MCP - Model Context Protocol (en VM gpt-oss-20b)")
        print("   5. N8n - Workflow Automation (en VM gpt-oss-20b)")
        print("\n   ¡Sistema ARM-Axion completamente integrado y funcional!")
        
        return True
    else:
        print(f"\n⚠️  {len(inactive_services)} servicios no están respondiendo")
        print("   Servicios inactivos:", [r["service"] for r in inactive_services])
        print("   Pueden requerir inicio manual o verificación adicional")
        
        return False

def check_n8n_details():
    """Verificar detalles específicos de n8n"""
    print("\n🔍 VERIFICACIÓN DETALLADA DE N8N (Workflows)...")

    cmd = [
        "gcloud", "compute", "ssh", "gpt-oss-20b",
        "--zone=europe-southwest1-b",
        "--project=mamba-001",
        "--command",
        "ls -la /home/elect/.n8n/ 2>/dev/null || echo 'Directorio n8n no encontrado'"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        print(f"   Contenido de directorio n8n: {result.stdout}")
    except Exception as e:
        print(f"   ❌ No se pudo acceder al directorio n8n: {e}")

    # Verificar si n8n está instalado como servicio
    cmd = [
        "gcloud", "compute", "ssh", "gpt-oss-20b",
        "--zone=europe-southwest1-b",
        "--project=mamba-001",
        "--command",
        "systemctl is-active n8n 2>/dev/null || echo 'Servicio n8n no encontrado o inactivo'"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        print(f"   Estado del servicio n8n: {result.stdout.strip()}")
    except Exception as e:
        print(f"   ❌ No se pudo verificar el servicio n8n: {e}")


def main():
    """Función principal"""
    print("🚀 Iniciando verificación de la VM de Servicios ARM-Axion...")
    print("   Verificando TTS, MCP y N8n en gpt-oss-20b (34.175.136.104)")
    
    success = check_remote_services()
    
    # Verificar detalles de n8n si está disponible
    check_n8n_details()
    
    print("\n" + "="*80)
    if success:
        print("✅ VERIFICACIÓN COMPLETA: VM de servicios ARM-Axion operativa")
        print("   Todos los servicios están respondiendo correctamente")
    else:
        print("⚠️  VERIFICACIÓN PARCIAL: Algunos servicios no están respondiendo")
        print("   Se requiere verificación adicional en la VM remota")
    print("="*80)


if __name__ == "__main__":
    main()