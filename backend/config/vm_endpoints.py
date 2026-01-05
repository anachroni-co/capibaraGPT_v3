#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración de endpoints de VMs para Capibara6
Este archivo contiene las URLs de conexión a los servicios en las diferentes VMs
"""

import os
import json
from typing import Dict, Optional

# Intentar cargar configuración desde archivo JSON si existe
VM_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../../vm_config.json")


def load_vm_config() -> Optional[Dict]:
    """Carga la configuración de VMs desde archivo JSON"""
    if os.path.exists(VM_CONFIG_FILE):
        try:
            with open(VM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error cargando vm_config.json: {e}")
    return None


# Configuración por defecto (se puede sobrescribir con variables de entorno o vm_config.json)
DEFAULT_VM_ENDPOINTS = {
    "bounty2": {
        "ip_external": os.getenv("BOUNTY2_IP_EXTERNAL", ""),
        "ip_internal": os.getenv("BOUNTY2_IP_INTERNAL", ""),
        "ollama": {
            "endpoint": os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
            "port": 11434,
            "models": ["gpt-oss-20B", "mixtral", "phi-mini3"]
        },
        "backend": {
            "endpoint": os.getenv("BOUNTY2_BACKEND_ENDPOINT", "http://localhost:5001"),
            "port": 5001
        }
    },
    "rag3": {
        "ip_external": os.getenv("RAG3_IP_EXTERNAL", ""),
        "ip_internal": os.getenv("RAG3_IP_INTERNAL", ""),
        "rag_api": {
            "endpoint": os.getenv("RAG_API_ENDPOINT", "http://localhost:8000"),
            "port": 8000
        }
    },
    "gpt-oss-20b": {
        "ip_external": os.getenv("GPTOSS_IP_EXTERNAL", ""),
        "ip_internal": os.getenv("GPTOSS_IP_INTERNAL", ""),
        "tts": {
            "endpoint": os.getenv("TTS_ENDPOINT", "http://localhost:5002"),
            "port": 5002
        },
        "mcp": {
            "endpoint": os.getenv("MCP_ENDPOINT", "http://localhost:5003"),
            "port": 5003
        },
        "mcp_alt": {
            "endpoint": os.getenv("MCP_ALT_ENDPOINT", "http://localhost:5010"),
            "port": 5010
        },
        "n8n": {
            "endpoint": os.getenv("N8N_ENDPOINT", "http://localhost:5678"),
            "port": 5678
        },
        "bridge": {
            "endpoint": os.getenv("BRIDGE_ENDPOINT", "http://localhost:5000"),
            "port": 5000
        }
    }
}


class VMEndpoints:
    """Gestor de endpoints de VMs"""
    
    def __init__(self):
        self.config = load_vm_config() or {}
        self.endpoints = self._build_endpoints()
    
    def _build_endpoints(self) -> Dict:
        """Construye los endpoints usando configuración de archivo o valores por defecto"""
        endpoints = DEFAULT_VM_ENDPOINTS.copy()
        
        # Si hay configuración desde archivo, usarla
        if self.config.get("service_endpoints"):
            file_endpoints = self.config["service_endpoints"]
            
            # Actualizar endpoints de bounty2
            if "bounty2" in file_endpoints:
                endpoints["bounty2"].update(file_endpoints["bounty2"])
            
            # Actualizar endpoints de rag3
            if "rag3" in file_endpoints:
                endpoints["rag3"].update(file_endpoints["rag3"])
            
            # Actualizar endpoints de gpt-oss-20b
            if "gpt-oss-20b" in file_endpoints:
                endpoints["gpt-oss-20b"].update(file_endpoints["gpt-oss-20b"])
        
        return endpoints
    
    def get_ollama_endpoint(self, use_internal: bool = True) -> str:
        """Obtiene el endpoint de Ollama"""
        vm_info = self.endpoints.get("bounty2", {})
        ollama = vm_info.get("ollama", {})
        
        if use_internal and vm_info.get("ip_internal"):
            return f"http://{vm_info['ip_internal']}:{ollama.get('port', 11434)}"
        elif vm_info.get("ip_external"):
            return f"http://{vm_info['ip_external']}:{ollama.get('port', 11434)}"
        
        return ollama.get("endpoint", "http://localhost:11434")
    
    def get_rag_endpoint(self, use_internal: bool = True) -> str:
        """Obtiene el endpoint del RAG API"""
        vm_info = self.endpoints.get("rag3", {})
        rag_api = vm_info.get("rag_api", {})
        
        if use_internal and vm_info.get("ip_internal"):
            return f"http://{vm_info['ip_internal']}:{rag_api.get('port', 8000)}"
        elif vm_info.get("ip_external"):
            return f"http://{vm_info['ip_external']}:{rag_api.get('port', 8000)}"
        
        return rag_api.get("endpoint", "http://localhost:8000")
    
    def get_tts_endpoint(self, use_internal: bool = True) -> str:
        """Obtiene el endpoint de TTS"""
        vm_info = self.endpoints.get("gpt-oss-20b", {})
        tts = vm_info.get("tts", {})
        
        if use_internal and vm_info.get("ip_internal"):
            return f"http://{vm_info['ip_internal']}:{tts.get('port', 5002)}"
        elif vm_info.get("ip_external"):
            return f"http://{vm_info['ip_external']}:{tts.get('port', 5002)}"
        
        return tts.get("endpoint", "http://localhost:5002")
    
    def get_mcp_endpoint(self, use_internal: bool = True, use_alt: bool = False) -> str:
        """Obtiene el endpoint de MCP"""
        vm_info = self.endpoints.get("gpt-oss-20b", {})
        mcp_key = "mcp_alt" if use_alt else "mcp"
        mcp = vm_info.get(mcp_key, {})
        
        if use_internal and vm_info.get("ip_internal"):
            return f"http://{vm_info['ip_internal']}:{mcp.get('port', 5003)}"
        elif vm_info.get("ip_external"):
            return f"http://{vm_info['ip_external']}:{mcp.get('port', 5003)}"
        
        return mcp.get("endpoint", "http://localhost:5003")
    
    def get_n8n_endpoint(self, use_internal: bool = True) -> str:
        """Obtiene el endpoint de N8n"""
        vm_info = self.endpoints.get("gpt-oss-20b", {})
        n8n = vm_info.get("n8n", {})
        
        if use_internal and vm_info.get("ip_internal"):
            return f"http://{vm_info['ip_internal']}:{n8n.get('port', 5678)}"
        elif vm_info.get("ip_external"):
            return f"http://{vm_info['ip_external']}:{n8n.get('port', 5678)}"
        
        return n8n.get("endpoint", "http://localhost:5678")
    
    def get_bridge_endpoint(self, use_internal: bool = True) -> str:
        """Obtiene el endpoint del Bridge"""
        vm_info = self.endpoints.get("gpt-oss-20b", {})
        bridge = vm_info.get("bridge", {})
        
        if use_internal and vm_info.get("ip_internal"):
            return f"http://{vm_info['ip_internal']}:{bridge.get('port', 5000)}"
        elif vm_info.get("ip_external"):
            return f"http://{vm_info['ip_external']}:{bridge.get('port', 5000)}"
        
        return bridge.get("endpoint", "http://localhost:5000")
    
    def are_vms_in_same_vpc(self) -> bool:
        """Verifica si las VMs están en la misma VPC"""
        if self.config.get("network", {}).get("same_vpc"):
            return True
        
        # Verificar manualmente
        networks = set()
        for vm_name in ["bounty2", "rag3", "gpt-oss-20b"]:
            vm_data = self.config.get("vms", {}).get(vm_name, {})
            if vm_data.get("network"):
                networks.add(vm_data["network"])
        
        return len(networks) == 1
    
    def get_all_endpoints(self, use_internal: bool = True) -> Dict[str, str]:
        """Obtiene todos los endpoints en un diccionario"""
        return {
            "ollama": self.get_ollama_endpoint(use_internal),
            "rag_api": self.get_rag_endpoint(use_internal),
            "tts": self.get_tts_endpoint(use_internal),
            "mcp": self.get_mcp_endpoint(use_internal),
            "mcp_alt": self.get_mcp_endpoint(use_internal, use_alt=True),
            "n8n": self.get_n8n_endpoint(use_internal),
            "bridge": self.get_bridge_endpoint(use_internal)
        }


# Instancia global
vm_endpoints = VMEndpoints()


if __name__ == "__main__":
    print("🔗 Endpoints de VMs de Capibara6")
    print("=" * 60)
    print()
    
    print("📡 Endpoints (usando IPs internas si están disponibles):")
    endpoints = vm_endpoints.get_all_endpoints(use_internal=True)
    for service, endpoint in endpoints.items():
        print(f"  {service:12s}: {endpoint}")
    
    print()
    print("🌐 Estado de Red:")
    if vm_endpoints.are_vms_in_same_vpc():
        print("  ✅ Todas las VMs están en la misma VPC")
    else:
        print("  ⚠️  Las VMs NO están en la misma VPC")
    
    print()
    print("💡 Para actualizar la configuración:")
    print("   1. Ejecuta: python3 scripts/get_vm_info.py")
    print("   2. O configura variables de entorno:")
    print("      export OLLAMA_ENDPOINT=http://IP:11434")
    print("      export RAG_API_ENDPOINT=http://IP:8000")
    print("      etc.")

