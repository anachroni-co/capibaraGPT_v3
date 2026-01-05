#!/usr/bin/env python3
"""
Script para verificar y actualizar configuración CORS en el backend
"""

import os
import sys

BACKEND_FILES = [
    'backend/server.py',
    'backend/capibara6_integrated_server.py',
    'backend/server_gptoss.py'
]

CORS_CONFIG = """
from flask_cors import CORS

# Configurar CORS para permitir localhost:8000
CORS(app, origins=[
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://www.capibara6.com",
    "https://capibara6.com"
])
"""

print("🔍 Verificando configuración CORS en archivos del backend...")
print("")

for file_path in BACKEND_FILES:
    if os.path.exists(file_path):
        print(f"📄 {file_path}:")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'CORS(app' in content:
                if 'localhost:8000' in content or 'origins=' in content or 'origins=[' in content:
                    print("  ✅ CORS configurado")
                else:
                    print("  ⚠️  CORS básico (sin restricciones de origen)")
            else:
                print("  ❌ CORS no encontrado")
    else:
        print(f"📄 {file_path}: No existe")

print("")
print("💡 Si CORS no está configurado correctamente, añade:")
print(CORS_CONFIG)

