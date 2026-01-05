#!/usr/bin/env python3
"""
Script simple para verificar que los parches de Triton están funcionando
"""

import sys
import os
sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')

def test_triton_import():
    """Probar que triton_key se puede importar correctamente"""
    print("🔍 Probando importación de triton_key...")
    
    try:
        # Probar el import en el lugar donde se esperaba originalmente
        from torch._inductor.runtime.triton_compat import triton_key
        result = triton_key()
        print(f"✅ Importación exitosa de triton_key: {result}")
        return True
    except Exception as e:
        print(f"❌ Error al importar triton_key: {e}")
        return False

def test_system_info():
    """Probar la función get_system que usaba triton_key"""
    print("\n🔍 Probando función get_system...")
    
    try:
        from torch._inductor.codecache import CacheBase
        system_info = CacheBase.get_system()
        print(f"✅ get_system funcionando: {type(system_info)}")
        print(f"   Información del sistema: {list(system_info.keys()) if isinstance(system_info, dict) else 'No es un dict'}")
        return True
    except Exception as e:
        print(f"❌ Error en get_system: {e}")
        return False

def test_torch_inductor():
    """Probar que torch._inductor puede cargar sin errores de triton_key"""
    print("\n🔍 Probando carga de torch._inductor...")
    
    try:
        import torch._inductor.codecache
        print("✅ torch._inductor.codecache cargado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error al cargar torch._inductor.codecache: {e}")
        return False

def main():
    print("🧪 PRUEBA DE PATCHES DE TRITON")
    print("="*50)
    
    # Establecer variables de entorno que ayudan a evitar problemas de compilación
    os.environ['TORCH_COMPILE_BACKEND'] = 'eager'
    os.environ['TORCHDYNAMO_DISABLED'] = '1'
    os.environ['VLLM_USE_V1'] = '0'
    os.environ['VLLM_ENABLE_V1_ENGINE'] = '0'
    os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
    os.environ['TORCHINDUCTOR_DISABLED'] = '1'
    
    tests = [
        ("Importación de triton_key", test_triton_import),
        ("Función get_system", test_system_info),
        ("Carga de torch._inductor", test_torch_inductor),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*50)
    print("📋 RESULTADOS:")
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print(f"\n{'✅ ¡TODAS LAS PRUEBAS PASARON!' if all_passed else '❌ ALGUNAS PRUEBAS FALLARON'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)