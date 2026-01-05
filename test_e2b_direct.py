#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test E2B Integration - Prueba directa del sandbox E2B
"""

import os
import asyncio
from e2b_code_interpreter import AsyncSandbox
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar la API key de E2B
E2B_API_KEY = "e2b_4bebb1dfce65d4db486ed23cd352d88e72f105df"
os.environ['E2B_API_KEY'] = E2B_API_KEY

async def test_e2b_connection():
    """Prueba la conexión al sandbox E2B"""
    print("🧪 Iniciando prueba de conexión E2B...")
    
    try:
        # Crear una instancia del sandbox E2B con la plantilla correcta
        print("🔗 Conectando al sandbox E2B...")
        sandbox = await AsyncSandbox.create(api_key=E2B_API_KEY)
        print("✅ Conexión E2B establecida")
        
        # Ejecutar un comando simple para verificar que funciona
        print("\n📝 Ejecutando código de prueba...")
        execution = await sandbox.run_code("print('¡Hola desde el sandbox E2B!')")
        
        print(f"📊 Resultado de ejecución: {execution}")
        
        if execution and execution.logs and execution.logs.stdout:
            print(f"✅ Salida del código: {execution.logs.stdout}")
        else:
            print("⚠️  No se obtuvo salida del código")
        
        # Probar con código más complejo
        print("\n🧮 Ejecutando cálculos complejos...")
        complex_code = """
import numpy as np
import pandas as pd

# Crear un array de ejemplo
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array shape: {arr.shape}")
print(f"Array sum: {arr.sum()}")

# Crear un dataframe
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"DataFrame mean: {df.mean()}")
"""
        
        execution2 = await sandbox.run_code(complex_code)
        if execution2 and execution2.logs and execution2.logs.stdout:
            print(f"✅ Salida del código complejo: {execution2.logs.stdout}")
        else:
            print("⚠️  No se obtuvo salida del código complejo")
        
        # Cerrar el sandbox
        sandbox.kill()
        print("✅ Sandbox E2B cerrado correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba E2B: {e}")
        return False

async def main():
    """Función principal"""
    print("🚀 Iniciando pruebas completas de E2B...")
    print(f"🔑 Usando API Key: {E2B_API_KEY[:15]}...")
    
    # Prueba de conexión básica
    basic_test_result = await test_e2b_connection()
    
    print("\n" + "="*60)
    print("📋 RESULTADOS FINALES")
    print("="*60)
    print(f"E2B Connection Test: {'✅ PASSED' if basic_test_result else '❌ FAILED'}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())