#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplificado para verificar conexiones del frontend con servicios
"""

import json
import re
import requests
import sys
from typing import Dict, List

# IPs desde config.js
def get_ips_from_config():
    """Lee IPs desde web/config.js"""
    try:
        with open('web/config.js', 'r') as f:
            content = f.read()
        
        ips = {}
        match = re.search(r"BOUNTY2_EXTERNAL:\s*['\"]([^'\"]+)['\"]", content)
        if match:
            ips['bounty2'] = match.group(1)
        
        match = re.search(r"GPTOSS_EXTERNAL:\s*['\"]([^'\"]+)['\"]", content)
        if match:
            ips['gptoss'] = match.group(1)
        
        match = re.search(r"RAG3_EXTERNAL:\s*['\"]([^'\"]+)['\"]", content)
        if match:
            ips['rag3'] = match.group(1)
        
        return ips
    except Exception as e:
        print(f"Error leyendo config.js: {e}")
        return {}

def test_url(url, method='GET', data=None):
    """Prueba una URL"""
    try:
        if method == 'GET':
            r = requests.get(url, timeout=5)
        elif method == 'POST':
            r = requests.post(url, json=data, timeout=5)
        else:
            return False, f"Metodo {method} no soportado"
        
        if r.status_code < 400:
            return True, f"OK ({r.status_code})"
        else:
            return False, f"Error {r.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("VERIFICACION DE CONEXIONES FRONTEND - CAPIBARA6")
    print("=" * 70)
    print()
    
    # Obtener IPs
    ips = get_ips_from_config()
    
    if not ips.get('bounty2'):
        print("ERROR: No se encontro IP de Bounty2 en web/config.js")
        sys.exit(1)
    
    if not ips.get('gptoss'):
        print("ERROR: No se encontro IP de gpt-oss-20b en web/config.js")
        sys.exit(1)
    
    print(f"IPs configuradas:")
    print(f"  Bounty2: {ips.get('bounty2', 'NO CONFIGURADA')}")
    print(f"  rag3: {ips.get('rag3', 'NO CONFIGURADA')}")
    print(f"  gpt-oss-20b: {ips.get('gptoss', 'NO CONFIGURADA')}")
    print()
    
    # Servicios a probar
    services = []
    
    # Bounty2 - Ollama
    services.append({
        'name': 'Ollama (Bounty2)',
        'url': f"http://{ips['bounty2']}:11434/api/tags",
        'method': 'GET'
    })
    
    # Bounty2 - Backend
    services.append({
        'name': 'Backend Flask (Bounty2)',
        'url': f"http://{ips['bounty2']}:5001/api/health",
        'method': 'GET'
    })
    
    # rag3 - RAG API
    if ips.get('rag3'):
        services.append({
            'name': 'RAG API (rag3)',
            'url': f"http://{ips['rag3']}:8000/health",
            'method': 'GET'
        })
    
    # gpt-oss-20b - TTS
    services.append({
        'name': 'TTS (gpt-oss-20b)',
        'url': f"http://{ips['gptoss']}:5002/api/tts/voices",
        'method': 'GET'
    })
    
    # gpt-oss-20b - MCP
    services.append({
        'name': 'MCP Server (gpt-oss-20b)',
        'url': f"http://{ips['gptoss']}:5003/api/mcp/status",
        'method': 'GET'
    })
    
    # gpt-oss-20b - MCP Alt
    services.append({
        'name': 'MCP Server Alt (gpt-oss-20b)',
        'url': f"http://{ips['gptoss']}:5010/api/mcp/status",
        'method': 'GET'
    })
    
    # gpt-oss-20b - N8n
    services.append({
        'name': 'N8n (gpt-oss-20b)',
        'url': f"http://{ips['gptoss']}:5678/healthz",
        'method': 'GET'
    })
    
    # gpt-oss-20b - Bridge
    services.append({
        'name': 'Bridge (gpt-oss-20b)',
        'url': f"http://{ips['gptoss']}:5000/api/health",
        'method': 'GET'
    })
    
    # Probar servicios
    print("Probando servicios...")
    print("-" * 70)
    
    results = []
    success_count = 0
    
    for service in services:
        print(f"\n{service['name']}")
        print(f"  URL: {service['url']}")
        print(f"  Estado: ", end="", flush=True)
        
        success, message = test_url(service['url'], service['method'])
        results.append({
            'name': service['name'],
            'url': service['url'],
            'success': success,
            'message': message
        })
        
        if success:
            print(f"OK - {message}")
            success_count += 1
        else:
            print(f"ERROR - {message}")
    
    # Resumen
    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total de servicios: {len(services)}")
    print(f"Exitosos: {success_count}")
    print(f"Fallidos: {len(services) - success_count}")
    print()
    
    if success_count == len(services):
        print("TODOS LOS SERVICIOS ESTAN FUNCIONANDO CORRECTAMENTE")
    else:
        print("SERVICIOS CON PROBLEMAS:")
        for r in results:
            if not r['success']:
                print(f"  - {r['name']}: {r['message']}")
    
    # Guardar resultados
    with open('connection_test_results.json', 'w') as f:
        json.dump({
            'ips': ips,
            'results': results,
            'summary': {
                'total': len(services),
                'success': success_count,
                'failed': len(services) - success_count
            }
        }, f, indent=2)
    
    print(f"\nResultados guardados en: connection_test_results.json")
    
    return 0 if success_count == len(services) else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

