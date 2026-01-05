#!/usr/bin/env python3
"""
Script de validación para el setup de fine-tuning GPT-OSS-20B
Verifica que todas las dependencias y configuraciones estén correctas
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Error: Se requiere Python 3.8+, tienes {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Verificar dependencias principales"""
    print("\n📦 Verificando dependencias...")
    
    required_packages = [
        'jax',
        'flax', 
        'optax',
        't5x',
        'seqio',
        'tensorflow',
        'tensorstore',
        'gin-config'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO INSTALADO")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Faltan dependencias: {', '.join(missing)}")
        print("Instala con: pip install " + " ".join(missing))
        return False
    
    return True

def check_tpu_connection():
    """Verificar conexión a TPU"""
    print("\n🔗 Verificando conexión a TPU...")
    
    try:
        import jax
        devices = jax.devices()
        tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
        
        if not tpu_devices:
            print("⚠️ No se detectaron dispositivos TPU")
            print("   Asegúrate de estar ejecutando en la VM TPU")
            return False
        
        print(f"✅ Detectados {len(tpu_devices)} dispositivos TPU")
        print(f"   Dispositivos: {[str(d) for d in tpu_devices[:3]]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error verificando TPU: {e}")
        return False

def check_gcs_access():
    """Verificar acceso a Google Cloud Storage"""
    print("\n☁️ Verificando acceso a GCS...")
    
    bucket = os.getenv('BUCKET', 'gs://your-gcs-bucket')
    if bucket == 'gs://your-gcs-bucket':
        print("⚠️ Variable BUCKET no configurada")
        print("   Exporta: export BUCKET='gs://tu-bucket'")
        return False
    
    try:
        result = subprocess.run(['gsutil', 'ls', bucket], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Acceso a GCS: {bucket}")
            return True
        else:
            print(f"❌ Error accediendo a GCS: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ gsutil no encontrado. Instala Google Cloud SDK")
        return False
    except Exception as e:
        print(f"❌ Error verificando GCS: {e}")
        return False

def check_gin_config():
    """Verificar archivo de configuración .gin"""
    print("\n⚙️ Verificando configuración .gin...")
    
    gin_file = Path("configs/gpt_oss_20b_finetune.gin")
    if not gin_file.exists():
        print("❌ No se encuentra gpt_oss_20b_finetune.gin")
        print("   Asegúrate de estar en el directorio fine-tuning/")
        return False
    
    with open(gin_file, 'r') as f:
        content = f.read()
    
    # Verificar placeholders
    placeholders = ['<BUCKET>', '<VOCAB_SIZE>', '<D_MODEL>', '<N_LAYERS>', '<N_HEADS>']
    found_placeholders = [p for p in placeholders if p in content]
    
    if found_placeholders:
        print(f"⚠️ Placeholders sin reemplazar: {found_placeholders}")
        print("   Edita el archivo .gin con valores reales")
        return False
    
    print("✅ Archivo .gin configurado correctamente")
    return True

def check_dataset_config():
    """Verificar configuración de datasets"""
    print("\n📊 Verificando configuración de datasets...")
    
    seqio_file = Path("datasets/seqio_tasks.py")
    if not seqio_file.exists():
        print("⚠️ No se encuentra seqio_tasks.py")
        print("   Crea la configuración de datasets")
        return False
    
    print("✅ Configuración de datasets encontrada")
    return True

def main():
    """Función principal de validación"""
    print("🔍 Validando setup de fine-tuning GPT-OSS-20B")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_tpu_connection,
        check_gcs_access,
        check_gin_config,
        check_dataset_config
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"❌ Error en verificación: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Resultado: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("🎉 ¡Setup validado correctamente! Listo para entrenar.")
        return 0
    else:
        print("❌ Setup incompleto. Corrige los errores antes de continuar.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
