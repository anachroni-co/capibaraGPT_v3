#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración para Google Cloud y E2B.
"""

import os
from typing import Dict, Any

class CloudConfig:
    """Configuración para servicios en la nube."""
    
    def __init__(self):
        # E2B Configuration
        self.e2b_api_key = os.getenv("E2B_API_KEY", "e2b_01ea80c0f5c76ebcac24d99e9136e2975787b918")
        
        # Google Cloud Configuration
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID", "mamba-001")
        self.gcp_zone = os.getenv("GCP_ZONE", "europe-southwest1-b")
        self.gcp_vm_name = os.getenv("GCP_VM_NAME", "gpt-oss-20b")
        
        # Model Configuration
        self.model_20b_endpoint = os.getenv("MODEL_20B_ENDPOINT", f"http://{self.gcp_vm_name}.{self.gcp_zone}.c.{self.gcp_project_id}.internal:8000")
        self.model_120b_endpoint = os.getenv("MODEL_120B_ENDPOINT", "http://localhost:8001")
        
        # E2B Sandbox Configuration
        self.e2b_sandbox_config = {
            'timeout': int(os.getenv("E2B_TIMEOUT", "30")),
            'memory_limit_mb': int(os.getenv("E2B_MEMORY_LIMIT_MB", "512")),
            'cpu_limit_percent': int(os.getenv("E2B_CPU_LIMIT_PERCENT", "50")),
            'max_concurrent_sandboxes': int(os.getenv("E2B_MAX_CONCURRENT", "5"))
        }
        
        self._validate_config()
        self._log_config()
    
    def _validate_config(self):
        """Valida la configuración."""
        if not self.e2b_api_key:
            raise ValueError("E2B_API_KEY es requerida")
        
        if not self.e2b_api_key.startswith("e2b_"):
            raise ValueError("E2B_API_KEY debe comenzar con 'e2b_'")
    
    def _log_config(self):
        """Registra la configuración (sin exponer credenciales)."""
        print("\n--- Cloud Configuration ---")
        print(f"E2B API Key: {self.e2b_api_key[:10]}...")
        print(f"GCP Project: {self.gcp_project_id}")
        print(f"GCP Zone: {self.gcp_zone}")
        print(f"GCP VM: {self.gcp_vm_name}")
        print(f"Model 20B Endpoint: {self.model_20b_endpoint}")
        print(f"E2B Timeout: {self.e2b_sandbox_config['timeout']}s")
        print(f"E2B Memory Limit: {self.e2b_sandbox_config['memory_limit_mb']}MB")
        print("---------------------------\n")
    
    def get_e2b_config(self) -> Dict[str, Any]:
        """Retorna configuración para E2B."""
        return {
            'api_key': self.e2b_api_key,
            'timeout': self.e2b_sandbox_config['timeout'],
            'memory_limit_mb': self.e2b_sandbox_config['memory_limit_mb'],
            'cpu_limit_percent': self.e2b_sandbox_config['cpu_limit_percent']
        }
    
    def get_gcp_config(self) -> Dict[str, Any]:
        """Retorna configuración para Google Cloud."""
        return {
            'project_id': self.gcp_project_id,
            'zone': self.gcp_zone,
            'vm_name': self.gcp_vm_name,
            'model_20b_endpoint': self.model_20b_endpoint,
            'model_120b_endpoint': self.model_120b_endpoint
        }
    
    def get_ssh_command(self) -> str:
        """Retorna comando SSH para conectarse a la VM."""
        return f"gcloud compute ssh --zone \"{self.gcp_zone}\" \"{self.gcp_vm_name}\" --project \"{self.gcp_project_id}\""


# Instancia global de configuración
cloud_config = CloudConfig()
