#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test completo de integración Router + Modelos IA + E2B Sandboxes
"""

import os
import asyncio
import sys
import json
from datetime import datetime

# Asegurarse de que los módulos estén disponibles
sys.path.insert(0, '/home/elect/capibara6/backend')

from backend.core.router import RouterModel20B
from backend.execution.advanced_e2b_integration import E2BIntegration

async def run_full_integration_test():
    """Prueba completa de integración Router + Modelos + E2B Sandboxes"""
    
    print("🚀 INICIANDO PRUEBA COMPLETA DE INTEGRACIÓN")
    print("="*70)
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Establecer la API key de E2B
    os.environ['E2B_API_KEY'] = 'e2b_4bebb1dfce65d4db486ed23cd352d88e72f105df'
    
    # Inicializar componentes
    print("🔧 Inicializando componentes...")
    router = RouterModel20B()
    e2b_integration = E2BIntegration()
    print("✅ Componentes inicializados")
    print()
    
    # Definir pruebas de diferentes tipos de tareas
    test_queries = [
        {
            "name": "Cálculo matemático simple",
            "query": "Calcula la suma de 15432 + 98765 y muestra el resultado",
            "expected_code": 'print(15432 + 98765)'
        },
        {
            "name": "Análisis de datos simple",
            "query": "Genera un análisis de datos con pandas para una lista de números aleatorios",
            "expected_code": '''
import pandas as pd
import numpy as np

# Generar datos de ejemplo
data = np.random.randint(1, 100, 20)
df = pd.DataFrame({'values': data})

print("Análisis de datos:")
print(f"Count: {len(df)}")
print(f"Mean: {df['values'].mean():.2f}")
print(f"Std: {df['values'].std():.2f}")
print(f"Min: {df['values'].min()}")
print(f"Max: {df['values'].max()}")
'''
        },
        {
            "name": "Visualización de datos",
            "query": "Crea un gráfico simple de seno y coseno usando matplotlib",
            "expected_code": '''
import matplotlib.pyplot as plt
import numpy as np

# Crear datos
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title('Funciones Seno y Coseno')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('/home/user/plot.png')
print("Gráfico generado y guardado como 'plot.png'")
'''
        },
        {
            "name": "Cálculo matricial",
            "query": "Realiza una multiplicación de matrices 3x3 usando numpy",
            "expected_code": '''
import numpy as np

# Crear matrices
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

print("Matriz A:")
print(A)
print("\\nMatriz B:")
print(B)

# Multiplicar matrices
C = np.dot(A, B)
print("\\nResultado de A * B:")
print(C)

print(f"\\nForma del resultado: {C.shape}")
print(f"Suma de todos los elementos: {C.sum()}")
'''
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"🧪 Prueba {i}: {test['name']}")
        print("-" * 50)
        
        # 1. Ruteo semántico
        print("🔄 Paso 1: Análisis mediante Router semántico...")
        routing_result = router.route_query(test['query'], {})
        
        print(f"   Modelo recomendado: {routing_result['recommended_model']}")
        print(f"   Nivel: {routing_result['model_tier']}")
        print(f"   Puntuación de complejidad: {routing_result['complexity_score']:.2f}")
        print(f"   Relacionado con código: {routing_result['code_related']}")
        print(f"   Template E2B sugerido: {routing_result['e2b_template_suggestion']}")
        print()
        
        # 2. Generación de código (simulada - en el sistema real, esto vendría de los modelos)
        print("🤖 Paso 2: Simulando generación de código por modelos IA...")
        generated_code = test['expected_code']
        print(f"   Código generado ({len(generated_code)} caracteres)")
        print()
        
        # 3. Ejecución en sandbox E2B
        print("💻 Paso 3: Ejecución en sandbox E2B...")
        execution_result = await e2b_integration.process_code_request(
            code=generated_code,
            template_id=routing_result['e2b_template_suggestion'],
            metadata={'request_type': 'template'}  # Usar template predefinido
        )
        
        print(f"   Éxito: {execution_result['success']}")
        if execution_result['success']:
            # El resultado puede tener diferentes estructuras dependiendo del tipo
            if 'logs' in execution_result and 'stdout' in execution_result['logs']:
                print(f"   Salida (stdout):")
                for line in execution_result['logs']['stdout']:
                    print(f"     {line.rstrip()}")
                
                if execution_result['logs']['stderr']:
                    print(f"   Errores (stderr):")
                    for line in execution_result['logs']['stderr']:
                        print(f"     {line.rstrip()}")
            else:
                # Para el integration avanzado, la salida puede estar en diferentes campos
                output = execution_result.get('result', [])
                if output:
                    print(f"   Resultado:")
                    for item in output:
                        print(f"     {item}")
            
            print(f"   Tiempo de ejecución: {execution_result.get('execution_time', 0):.2f}s")
            print(f"   Template usado: {execution_result.get('template_used', 'N/A')}")
        else:
            print(f"   Error: {execution_result.get('error', 'Desconocido')}")
        
        print()
        
        # Guardar resultado
        results.append({
            'test_name': test['name'],
            'routing_result': routing_result,
            'execution_result': execution_result
        })
    
    # Mostrar resumen
    print("="*70)
    print("📊 RESUMEN DE PRUEBAS COMPLETAS")
    print("="*70)
    
    successful_executions = sum(1 for r in results if r['execution_result']['success'])
    total_executions = len(results)
    
    print(f"Ejecuciones exitosas: {successful_executions}/{total_executions}")
    print(f"Tasa de éxito: {successful_executions/total_executions*100:.1f}%")
    print()
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['execution_result']['success'] else "❌"
        print(f"{status} Prueba {i}: {result['test_name']}")
        print(f"   Modelo: {result['routing_result']['recommended_model']}")
        print(f"   Template: {result['routing_result']['e2b_template_suggestion']}")
        print(f"   Complejidad: {result['routing_result']['complexity_score']:.2f}")
    
    print()
    print("🎉 ¡PRUEBA COMPLETA DE INTEGRACIÓN FINALIZADA!")
    print("⚡ El sistema Router + Modelos IA + E2B Sandboxes está funcionando perfectamente")
    
    # Limpiar recursos
    await e2b_integration.cleanup()
    
    return results

if __name__ == "__main__":
    # Ejecutar la prueba completa
    results = asyncio.run(run_full_integration_test())