#!/usr/bin/env python3
"""
Script mejorado para descargar el modelo Cohere c4ai-command-r-plus-4bit
con monitoreo de progreso
"""
import os
import time
from huggingface_hub import snapshot_download
from pathlib import Path

def monitor_download_progress(local_dir):
    """Monitorear el progreso de la descarga"""
    start_time = time.time()
    
    while True:
        time.sleep(10)  # Esperar 10 segundos entre verificaciones
        
        # Contar archivos descargados
        if os.path.exists(local_dir):
            files = list(Path(local_dir).rglob("*"))
            size = sum(f.stat().st_size for f in files if f.is_file())
            
            elapsed = time.time() - start_time
            print(f"Tiempo transcurrido: {elapsed/60:.1f} minutos")
            print(f"Archivos descargados: {len([f for f in files if f.is_file()])}")
            print(f"Tamaño total: {size / (1024**3):.2f} GB")
            
            # Buscar archivos específicos del modelo
            model_files = [f for f in files if 'model' in f.name.lower() or f.name.endswith('.bin') or f.name.endswith('.safetensors')]
            if model_files:
                print(f"Archivos de modelo encontrados: {len(model_files)}")
                for f in model_files[:3]:  # Mostrar los primeros 3
                    print(f"  - {f.name}: {f.stat().st_size / (1024**3):.2f} GB")
                break

def download_model():
    """Descargar el modelo Cohere c4ai-command-r-plus-4bit"""
    model_name = "CohereForAI/c4ai-command-r-plus-4bit"
    local_dir = "/home/elect/models/c4ai-command-r-plus-4bit"
    
    print(f"Descargando el modelo {model_name}...")
    print(f"Directorio local: {local_dir}")
    
    # Crear el directorio si no existe
    os.makedirs(local_dir, exist_ok=True)
    
    # Iniciar monitoreo en segundo plano
    import threading
    monitor_thread = threading.Thread(target=monitor_download_progress, args=(local_dir,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Descargar el modelo
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=2,  # Limitar workers para no sobrecargar
        )
        print("¡Descarga completada!")
    except Exception as e:
        print(f"Error durante la descarga: {e}")
    
    # Esperar un poco para que se complete el monitoreo
    time.sleep(5)

if __name__ == "__main__":
    download_model()