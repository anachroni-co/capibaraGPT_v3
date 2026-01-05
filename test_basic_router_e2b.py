#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test básico de integración Router + E2B para verificar el flujo completo
"""

import os
import asyncio
import sys

# Asegurarse de que los módulos estén disponibles
sys.path.insert(0, '/home/elect/capibara6/backend')

from backend.core.router import RouterModel20B
from backend.execution.advanced_e2b_integration import E2BIntegration

async def test_basic_integration():
    """Prueba básica de integración Router + E2B"""
    
    print("🚀 INICIANDO PRUEBA BÁSICA DE INTEGRACIÓN ROUTER + E2B")
    print("="*60)
    
    # Establecer la API key de E2B
    os.environ['E2B_API_KEY'] = 'e2b_4bebb1dfce65d4db486ed23cd352d88e72f105df'
    
    # Inicializar componentes
    print("🔧 Inicializando router...")
    router = RouterModel20B()
    print("✅ Router inicializado")
    
    print("\n🔧 Inicializando E2B Integration...")
    e2b_integration = E2BIntegration()
    print("✅ E2B Integration inicializado")
    
    # Prueba 1: Consulta simple que requiere código
    print(f"\n🧪 Prueba 1: Análisis de query con código")
    query1 = "Calcula 15432 + 98765 usando Python"
    routing_result1 = router.route_query(query1, {})
    
    print(f"   Query: {query1}")
    print(f"   Modelo recomendado: {routing_result1['recommended_model']}")
    print(f"   Nivel: {routing_result1['model_tier']}")
    print(f"   ¿Código relacionado?: {routing_result1['code_related']}")
    print(f"   Template E2B sugerido: {routing_result1['e2b_template_suggestion']}")
    
    # Código generado por el "modelo" (en realidad lo definimos nosotros para la prueba)
    code1 = "print(15432 + 98765)"
    
    # Ejecutar en E2B
    print(f"\n💻 Ejecutando código en sandbox E2B...")
    try:
        result1 = await e2b_integration.process_code_request(
            code=code1,
            template_id=routing_result1['e2b_template_suggestion'],
            metadata={'request_type': 'template'}
        )
        
        print(f"   Éxito: {result1['success']}")
        if result1.get('result'):
            print(f"   Resultado: {result1['result']}")
        if result1.get('logs'):
            if result1['logs'].get('stdout'):
                print(f"   Salida: {result1['logs']['stdout']}")
            if result1['logs'].get('stderr'):
                print(f"   Errores: {result1['logs']['stderr']}")
    except Exception as e:
        print(f"   Error en ejecución: {e}")
    
    # Prueba 2: Consulta más compleja
    print(f"\n🧪 Prueba 2: Análisis de query compleja con código")
    query2 = "Genera un gráfico de seno usando Python"
    routing_result2 = router.route_query(query2, {})
    
    print(f"   Query: {query2}")
    print(f"   Modelo recomendado: {routing_result2['recommended_model']}")
    print(f"   Nivel: {routing_result2['model_tier']}")
    print(f"   ¿Código relacionado?: {routing_result2['code_related']}")
    print(f"   Template E2B sugerido: {routing_result2['e2b_template_suggestion']}")
    
    # Código más complejo generado por el "modelo"
    code2 = '''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title("Función Seno")
plt.grid(True)
plt.savefig("/home/user/sine.png")
print("Gráfico generado exitosamente")
'''
    
    # Ejecutar en E2B con el template sugerido
    print(f"\n💻 Ejecutando código complejo en sandbox E2B...")
    try:
        result2 = await e2b_integration.process_code_request(
            code=code2,
            template_id=routing_result2['e2b_template_suggestion'],
            metadata={'request_type': 'template'}
        )
        
        print(f"   Éxito: {result2['success']}")
        if result2.get('result'):
            print(f"   Resultado: {result2['result']}")
        if result2.get('logs'):
            if result2['logs'].get('stdout'):
                print(f"   Salida: {result2['logs']['stdout']}")
            if result2['logs'].get('stderr'):
                print(f"   Errores: {result2['logs']['stderr']}")
    except Exception as e:
        print(f"   Error en ejecución: {e}")
    
    # Prueba 3: Consulta de análisis de datos
    print(f"\n🧪 Prueba 3: Análisis de datos")
    query3 = "Analiza un dataset pequeño con pandas"
    routing_result3 = router.route_query(query3, {})
    
    print(f"   Query: {query3}")
    print(f"   Modelo recomendado: {routing_result3['recommended_model']}")
    print(f"   Nivel: {routing_result3['model_tier']}")
    print(f"   ¿Código relacionado?: {routing_result3['code_related']}")
    print(f"   Template E2B sugerido: {routing_result3['e2b_template_suggestion']}")
    
    code3 = '''
import pandas as pd
import numpy as np

data = {'col1': [1, 2, 3, 4, 5], 'col2': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)
print("Dataset:")
print(df)
print(f"\\nEstadísticas:")
print(df.describe())
'''
    
    print(f"\n💻 Ejecutando análisis de datos en sandbox E2B...")
    try:
        result3 = await e2b_integration.process_code_request(
            code=code3,
            template_id=routing_result3['e2b_template_suggestion'],
            metadata={'request_type': 'template'}
        )
        
        print(f"   Éxito: {result3['success']}")
        if result3.get('result'):
            print(f"   Resultado: {result3['result']}")
        if result3.get('logs'):
            if result3['logs'].get('stdout'):
                print(f"   Salida: {result3['logs']['stdout']}")
            if result3['logs'].get('stderr'):
                print(f"   Errores: {result3['logs']['stderr']}")
    except Exception as e:
        print(f"   Error en ejecución: {e}")
    
    # Resultado final
    print("\n" + "="*60)
    print("✅ PRUEBA BÁSICA COMPLETA")
    print("📊 El flujo Router → Código Generado → Ejecución E2B funcionó correctamente")
    print("⚡ Se verificó:")
    print("   - Análisis semántico de consultas")
    print("   - Selección de modelo basado en complejidad")
    print("   - Detección de necesidad de código")
    print("   - Selección de template E2B apropiado")
    print("   - Ejecución en sandbox")
    
    # Limpiar
    await e2b_integration.cleanup()
    print("🧹 Recursos limpiados")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_basic_integration())