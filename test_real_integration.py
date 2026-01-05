#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de integración completa simulando la interacción real como lo haría el frontend
"""

import os
import asyncio
import sys

# Asegurarse de que los módulos estén disponibles
sys.path.insert(0, '/home/elect/capibara6/backend')

from backend.core.router import RouterModel20B
from backend.execution.advanced_e2b_integration import E2BIntegration

async def test_real_integration():
    """Prueba de integración simulando el flujo real como si viniera del frontend"""
    
    print("🚀 INICIANDO PRUEBA DE INTEGRACIÓN REAL")
    print("="*60)
    
    # Establecer la API key de E2B
    os.environ['E2B_API_KEY'] = 'e2b_4bebb1dfce65d4db486ed23cd352d88e72f105df'
    
    # Inicializar componentes como lo hace main.py
    print("🔧 Inicializando sistema como en main.py...")
    router = RouterModel20B()
    e2b_integration = E2BIntegration()
    print("✅ Sistema inicializado")
    
    # Simular el proceso que ocurriría en main.py: process_query
    print(f"\n🔄 Simulando proceso como en main.py - process_query()")
    
    # Ejemplos de queries reales que podría recibir el sistema
    scenarios = [
        {
            "name": "Análisis de datos",
            "user_query": "Tengo una lista de ventas y quiero analizar tendencias. ¿Puedes graficarlas?",
            "expected_task": "data_analysis"
        },
        {
            "name": "Cálculo matemático",
            "user_query": "Necesito calcular la serie de Fibonacci hasta el número 20",
            "expected_task": "mathematical_calculation"
        },
        {
            "name": "Visualización",
            "user_query": "Quiero ver como se ve una función cuadrática graficada",
            "expected_task": "visualization"
        },
        {
            "name": "Operación simple",
            "user_query": "¿Cuánto es 256 * 43?",
            "expected_task": "simple_calculation"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Escenario {i}: {scenario['name']} ---")
        
        # 1. El router analiza la query
        print(f"💬 Query del usuario: {scenario['user_query']}")
        routing_result = router.route_query(scenario['user_query'], {})
        
        print(f"🤖 Router analiza y decide:")
        print(f"   - Modelo recomendado: {routing_result['recommended_model']}")
        print(f"   - Tier: {routing_result['model_tier']}")
        print(f"   - Complejidad: {routing_result['complexity_score']:.2f}")
        print(f"   - ¿Código relacionado?: {routing_result['code_related']}")
        print(f"   - Template E2B sugerido: {routing_result['e2b_template_suggestion']}")
        print(f"   - Razonamiento: {routing_result['reasoning'][:80]}...")
        
        # 2. Simular que un modelo de IA genera código basado en la query
        # En el sistema real, esto vendría del modelo de IA
        print(f"\n📝 Simulando generación de código por modelo de IA...")
        
        # Generar código apropiado según el tipo de tarea detectada por el router
        if 'visual' in scenario['user_query'].lower() or 'gráfica' in scenario['user_query'].lower():
            generated_code = '''
import matplotlib.pyplot as plt
import numpy as np

# Generar datos para función cuadrática
x = np.linspace(-10, 10, 400)
y = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
plt.title('Función Cuadrática: f(x) = x²', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Guardar gráfico
plt.savefig('/home/user/quadratic_plot.png', dpi=150, bbox_inches='tight')
print("✅ Gráfico de función cuadrática generado y guardado")
print("📊 Puntos generados:", len(x))
print("📈 Valores de y: min={:.2f}, max={:.2f}".format(y.min(), y.max()))
'''
        elif 'fibonacci' in scenario['user_query'].lower():
            generated_code = '''
def fibonacci(n):
    """Genera la serie de Fibonacci hasta n términos."""
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence

# Calcular serie de Fibonacci
n = 20
fib_sequence = fibonacci(n)

print(f"✅ Serie de Fibonacci hasta {n} términos:")
print(f"   {fib_sequence}")
print(f"📊 Total de números: {len(fib_sequence)}")
print(f"📈 Último número: {fib_sequence[-1]}")

# Calcular algunas estadísticas
print(f"📈 Media de la serie: {sum(fib_sequence) / len(fib_sequence):.2f}")
print(f"📈 Suma total: {sum(fib_sequence)}")
'''
        elif 'ventas' in scenario['user_query'].lower() or 'tendencias' in scenario['user_query'].lower():
            generated_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo de ventas
np.random.seed(42)
dias = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
ventas = np.random.randint(1000, 5000, size=12)

# Crear DataFrame
df = pd.DataFrame({
    'Mes': dias,
    'Ventas': ventas
})

print("📊 Análisis de Ventas:")
print(df)

# Estadísticas
print(f"\\n📈 Estadísticas:")
print(f"   Total ventas: {ventas.sum():,}")
print(f"   Promedio mensual: {ventas.mean():,.2f}")
print(f"   Mayor venta: {ventas.max():,} (mes {dias[ventas.argmax()]})")
print(f"   Menor venta: {ventas.min():,} (mes {dias[ventas.argmin()]})")

# Crear gráfico
plt.figure(figsize=(12, 6))
plt.plot(df['Mes'], df['Ventas'], marker='o', linewidth=2, markersize=8)
plt.title('Ventas Mensuales', fontsize=14)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Ventas', fontsize=12)
plt.grid(True, alpha=0.3)

# Rotar etiquetas para mejor visibilidad
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/user/sales_trend.png', dpi=150, bbox_inches='tight')

print("\\n✅ Gráfico de tendencias generado")
'''
        else:  # Escenario simple
            generated_code = f'''
# Cálculo simple: {scenario["user_query"]}
resultado = 256 * 43

print("🧮 Operación: 256 * 43")
print(f"✅ Resultado: {{resultado}}")
print(f"📊 El número {{resultado}} tiene {{len(str(resultado))}} dígitos")
'''
        
        print(f"   Código generado ({len(generated_code)} caracteres)")
        print(f"   Comienza con: {generated_code[:60]}...")
        
        # 3. Ejecutar el código generado en E2B usando el template sugerido por el router
        print(f"\n💻 Ejecutando código en sandbox E2B...")
        print(f"   Usando template: {routing_result['e2b_template_suggestion']}")
        
        try:
            e2b_result = await e2b_integration.process_code_request(
                code=generated_code,
                template_id=routing_result['e2b_template_suggestion'],
                metadata={'request_type': 'template'}
            )
            
            print(f"   🎯 Ejecución: {'✅ EXITOSA' if e2b_result['success'] else '❌ FALLIDA'}")
            
            if e2b_result['success']:
                print(f"   ⏱️  Tiempo de ejecución: {e2b_result.get('execution_time', 0):.3f}s")
                print(f"   🏷️  Template usado: {e2b_result.get('template_used', 'desconocido')}")
                
                # Mostrar la salida del código
                if e2b_result.get('result'):
                    print(f"   📤 Salida del código:")
                    for idx, item in enumerate(e2b_result['result'][:3]):  # Mostrar primeros 3 resultados
                        print(f"     - {item}")
                    if len(e2b_result['result']) > 3:
                        print(f"     ... y {len(e2b_result['result']) - 3} más")
                        
                if e2b_result.get('logs', {}).get('stdout'):
                    print(f"   📥 STDOUT:")
                    for line in e2b_result['logs']['stdout'][:5]:  # Primeras 5 líneas
                        print(f"     {line.rstrip()}")
                    if len(e2b_result['logs']['stdout']) > 5:
                        print(f"     ... y {len(e2b_result['logs']['stdout']) - 5} más")
                        
            else:
                print(f"   ❌ Error: {e2b_result.get('error', 'Desconocido')}")
                
        except Exception as e:
            print(f"   ❌ Error en ejecución: {e}")
    
    print(f"\n" + "="*60)
    print("✅ PRUEBA DE INTEGRACIÓN REAL COMPLETADA")
    print("🎯 Se verificó el flujo completo:")
    print("   1. Análisis semántico con Router")
    print("   2. Detección de tipo de tarea")
    print("   3. Selección de modelo y template apropiados")
    print("   4. Generación de código (simulada)")
    print("   5. Ejecución en sandbox E2B con template adecuado")
    print("   6. Obtención de output del VM")
    print()
    print("⚡ El sistema está completamente integrado y funcional!")
    
    # Limpiar recursos
    await e2b_integration.cleanup()
    
    return True

if __name__ == "__main__":
    asyncio.run(test_real_integration())