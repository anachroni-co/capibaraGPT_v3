#!/usr/bin/env python3
"""
Script para descargar el modelo Cohere c4ai-command-r-plus-4bit
"""
import os
from huggingface_hub import snapshot_download

# Descargar el modelo Cohere c4ai-command-r-plus-4bit
model_name = "CohereForAI/c4ai-command-r-plus-4bit"
local_dir = "/home/elect/models/c4ai-command-r-plus-4bit"

print(f"Descargando el modelo {model_name}...")
print(f"Directorio local: {local_dir}")

# Crear el directorio si no existe
os.makedirs(local_dir, exist_ok=True)

# Descargar el modelo
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Evitar enlaces simbólicos
    resume_download=True,  # Reanudar descargas interrumpidas
)

print("¡Descarga completada!")