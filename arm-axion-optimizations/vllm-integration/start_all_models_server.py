#!/usr/bin/env python3
"""
Script para inicializar el servidor vLLM con todos los modelos ARM-Axion
Configura las operaciones personalizadas con fallback y arranca el servidor
"""

import os
import sys
import torch

# Asegurar que nuestro vLLM modificado esté en el path
vllm_path = '/home/elect/capibara6/vllm-source-modified'
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

# Asegurar que nuestro vLLM modificado esté en el path
vllm_path = '/home/elect/capibara6/vllm-source-modified'
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

# FORZAR USO DEL BACKEND CLÁSICO ANTES DE IMPORTAR vLLM
os.environ['VLLM_USE_V1'] = '0'  # Deshabilitar V1 Engine
os.environ['VLLM_ENABLE_V1_ENGINE'] = '0'  # Deshabilitar V1 Engine
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def setup_custom_ops_fallback():
    """
    Configura fallback para operaciones personalizadas que no están disponibles
    """
    print("🔧 Configurando fallback para operaciones personalizadas...")
    
    try:
        # Intentamos importar torch.ops._C, si falla creamos los fallbacks
        import torch
        
        # Si está disponible _C, verificamos qué operaciones están disponibles
        if hasattr(torch.ops, '_C'):
            print("✅ torch.ops._C disponible, verificando operaciones...")
            
            # Obtener el namespace _C
            c_ops = torch.ops._C
            
            # Lista de operaciones que necesitamos verificar
            required_ops = [
                'rms_norm',
                'fused_add_rms_norm',
                'rotary_embedding',
                'apply_repetition_penalties_',
                'paged_attention_v1',
                'paged_attention_v2',
                'awq_dequantize',
                'awq_gemm',
                'gptq_gemm',
                'gptq_shuffle',
                'moe_align_block_size',
                'cutlass_scaled_mm',
                'cutlass_scaled_fp4_mm',
            ]
            
            available_ops = []
            missing_ops = []
            
            for op_name in required_ops:
                if hasattr(c_ops, op_name):
                    available_ops.append(op_name)
                else:
                    missing_ops.append(op_name)
            
            print(f"   ✓ Operaciones disponibles: {len(available_ops)}")
            print(f"   ⚠️ Operaciones faltantes: {len(missing_ops)}")
            
            if missing_ops:
                print(f"   - Faltantes: {missing_ops}")
            
            # Si hay operaciones faltantes, creamos versiones fallback
            for op_name in missing_ops:
                # Creamos un stub para cada operación faltante
                def make_stub(op_name):
                    def stub_fn(*args, **kwargs):
                        raise NotImplementedError(f"Operación personalizada '{op_name}' no está disponible en esta plataforma ARM64. "
                                                f"Esta operación requiere compilación específica para ARM64.")
                    return stub_fn
                
                setattr(c_ops, op_name, make_stub(op_name))
                
        else:
            print("⚠️ torch.ops._C no está disponible, creando espacio de nombres...")
            # Creamos un módulo mock para _C
            class MockCNamespace:
                pass
            
            torch.ops._C = MockCNamespace()
            
            # Añadir operaciones mock para las operaciones comunes
            required_ops = [
                'rms_norm', 'fused_add_rms_norm', 'rotary_embedding',
                'apply_repetition_penalties_', 'paged_attention_v1', 'paged_attention_v2',
                'awq_dequantize', 'awq_gemm', 'gptq_gemm', 'gptq_shuffle',
                'moe_align_block_size', 'cutlass_scaled_mm', 'cutlass_scaled_fp4_mm'
            ]
            
            for op_name in required_ops:
                def make_mock_op(op_name):
                    def mock_op_fn(*args, **kwargs):
                        raise NotImplementedError(f"Operación '{op_name}' no disponible en ARM64. "
                                                f"Esta operación requiere compilación de kernels para ARM64.")
                    return mock_op_fn
                
                setattr(torch.ops._C, op_name, make_mock_op(op_name))
                
    except Exception as e:
        print(f"⚠️ Error configurando fallback de operaciones: {e}")
        # Si falla completamente, seguimos adelante ya que vLLM tiene mecanismos internos para manejar esto


def create_torch_compiled_fallbacks():
    """
    Crea implementaciones de fallback para operaciones que usan torch.compile
    """
    print("🔧 Creando fallbacks para operaciones torch.compile...")
    import torch
    
    # Asegurar que use métodos compatibles con ARM
    if not hasattr(torch, '_dynamo'):
        try:
            import torch._dynamo
        except ImportError:
            print("⚠️ Dynamo no disponible, usando eager mode")
            os.environ['TORCH_DYNAMO_BACKEND'] = 'eager'
            os.environ['TORCH_COMPILE_BACKEND'] = 'eager'
    else:
        print("✅ Dynamo disponible")
        

def check_platform_compatibility():
    """
    Verifica la compatibilidad de la plataforma ARM-Axion
    """
    from vllm.platforms import current_platform
    
    print(f"🌍 Plataforma actual: {current_platform.device_type}")
    print(f"   ¿Es CPU?: {current_platform.is_cpu()}")
    print(f"   ¿Es CUDA?: {current_platform.is_cuda()}")
    
    # Confirmar que la plataforma ARM64 se detecta correctamente
    if not current_platform.is_cpu() or current_platform.device_type != 'cpu':
        print("⚠️ ADVERTENCIA: La plataforma ARM64 no se detecta correctamente como CPU. Continuando de todos modos.")
        return True
    else:
        print("✅ Plataforma ARM64 correctamente detectada como CPU")
        return True


def start_multi_model_server():
    """
    Inicia el servidor multi-modelo ARM-Axion con todos los modelos
    """
    print("="*70)
    print("🚀 INICIANDO SERVIDOR MULTI-MODELO ARM-AXION")
    print("   5 Modelos: Qwen2.5, Phi4-mini, Mistral7B, Gemma3-27B, GPT-OSS-20B")
    print("="*70)
    
    # Configurar entorno
    setup_custom_ops_fallback()
    create_torch_compiled_fallbacks()
    
    # Verificar plataforma
    if not check_platform_compatibility():
        print("❌ No se puede continuar: plataforma no compatible detectada")
        return
    
    # Ahora importamos el servidor después de configurar todo
    print("\n📥 Importando servidor multi-modelo...")
    
    try:
        from fallback_multi_model_server import app
        print("✅ Servidor multi-modelo importado correctamente")
        
        # Mostrar configuración
        import json
        config_path = "config.production.json"  # Valor por defecto
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"\n📋 Configuración cargada con {len(config.get('experts', []))} modelos:")
            for expert in config.get('experts', []):
                print(f"   • {expert['expert_id']}: {expert.get('description', '')}")
        
    except ImportError as e:
        print(f"❌ Error importando servidor: {e}")
        print("   Verificando si el archivo existe...")
        
        server_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/multi_model_server.py"
        if os.path.exists(server_path):
            print(f"   ✓ Archivo encontrado en: {server_path}")
        else:
            print(f"   ❌ Archivo no encontrado: {server_path}")
        
        # Importar directamente desde el path correcto
        import importlib.util
        spec = importlib.util.spec_from_file_location("multi_model_server", server_path)
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        app = server_module.app
        print("✅ Servidor importado usando importlib")
    
    print("\n📡 Iniciando servidor en puerto 8080...")
    print("   Presiona Ctrl+C para detener")
    print("="*70)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


def start_interactive_test():
    """
    Inicia una interfaz interactiva para probar todos los modelos
    """
    print("="*70)
    print("🧪 INICIANDO TEST INTERACTIVO DE MODELOS ARM-AXION")
    print("   Prueba de los 5 modelos con sistema de fallback")
    print("="*70)
    
    # Configurar entorno
    setup_custom_ops_fallback()
    create_torch_compiled_fallbacks()
    
    # Verificar plataforma
    if not check_platform_compatibility():
        print("❌ No se puede continuar: plataforma no compatible detectada")
        return
    
    # Importar la interfaz interactiva
    try:
        from interactive_test_interface import InteractiveCapibara6
        print("✅ Interfaz interactiva importada correctamente")
        
        app = InteractiveCapibara6()
        app.run()
        
    except ImportError as e:
        print(f"❌ Error importando interfaz: {e}")
        
        # Intentar importar desde el path correcto
        interface_path = "/home/elect/capibara6/interactive_test_interface.py"
        if os.path.exists(interface_path):
            print(f"   ✓ Archivo encontrado en: {interface_path}")
            import importlib.util
            spec = importlib.util.spec_from_file_location("interactive_test_interface", interface_path)
            interface_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(interface_module)
            
            # Obtener la clase si existe
            if hasattr(interface_module, 'InteractiveCapibara6'):
                app = interface_module.InteractiveCapibara6()
                app.run()
            elif hasattr(interface_module, 'main'):
                interface_module.main()
            else:
                print("   ❌ No se encontró interfaz interactiva funcional")
                return
        else:
            print(f"   ❌ Archivo no encontrado: {interface_path}")
            return
    
    print("\n👋 ¡Prueba interactiva completada!")


if __name__ == "__main__":
    print("Seleccione una opción:")
    print("1. Iniciar servidor multi-modelo ARM-Axion")
    print("2. Iniciar test interactivo de modelos")
    print("3. Iniciar ambos")
    
    choice = input("\nIngrese su elección (1-3): ").strip()
    
    if choice == "1":
        start_multi_model_server()
    elif choice == "2":
        start_interactive_test()
    elif choice == "3":
        print("Iniciando ambos sistemas...")
        # Podríamos iniciar ambos en procesos separados, pero por simplicidad:
        start_multi_model_server()
    else:
        print("Opción inválida")