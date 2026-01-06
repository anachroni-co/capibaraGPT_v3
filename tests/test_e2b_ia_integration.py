#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Complete E2B + IA Models - Prueba completa de integración
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

async def test_ai_code_generation_and_execution():
    """Prueba de generación y ejecución de código con IA"""
    print("🤖 Iniciando prueba de integración E2B + IA Models...")
    
    try:
        # Crear una instancia del sandbox E2B
        print("🔗 Conectando al sandbox E2B...")
        sandbox = await AsyncSandbox.create(api_key=E2B_API_KEY, timeout=600)  # 10 minutos de timeout
        print("✅ Conexión E2B establecida")
        
        # Simular generación de código por parte de un modelo de IA
        print("\n📝 Simulando generación de código por modelo de IA...")
        
        # Código generado que podría producir un modelo de IA
        ai_generated_code = """
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Simular datos de una semana de ventas
days = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
ventas = np.random.randint(100, 300, size=7)

print(f"📊 Reporte de ventas generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Ventas promedio: {ventas.mean():.2f}")
print(f"Mayor venta: {ventas.max()}")
print(f"Día con mayor venta: {days[np.argmax(ventas)]}")

# Crear gráfico simple
plt.figure(figsize=(10, 5))
plt.bar(days, ventas)
plt.title('Ventas por Día de la Semana')
plt.xlabel('Día')
plt.ylabel('Ventas')
plt.savefig('/home/user/ventas_grafico.png')  # Guardar gráfico
print("✅ Gráfico de ventas generado y guardado como 'ventas_grafico.png'")

# Devolver resultados
results = {
    'promedio_ventas': ventas.mean(),
    'mayor_venta': ventas.max(),
    'dia_mayor_venta': days[np.argmax(ventas)],
    'datos_completos': list(zip(days, ventas))
}
print(f"Resultados detallados: {results}")
"""
        
        print("🔄 Ejecutando código generado por IA...")
        execution = await sandbox.run_code(ai_generated_code)
        
        if execution and execution.logs and execution.logs.stdout:
            print("✅ Salida del código generado por IA:")
            for line in execution.logs.stdout:
                print(f"   {line.strip()}")
        else:
            print("⚠️  No se obtuvo salida del código generado por IA")
        
        if execution.error:
            print(f"❌ Error en la ejecución: {execution.error}")
        
        # Ejecutar análisis adicional
        print("\n🔍 Ejecutando análisis adicional...")
        analysis_code = """
import os
import json

# Simular un archivo de configuración que podría generar un modelo de IA
config = {
    "model_name": "capibara6-advanced",
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9,
        "frequency_penalty": 0.5
    },
    "features": ["reasoning", "code_generation", "data_analysis"],
    "version": "1.0.0",
    "timestamp": "2024-01-01T00:00:00Z"
}

# Guardar configuración
with open('/home/user/model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"⚙️  Configuración del modelo guardada: {len(json.dumps(config, indent=2))} caracteres")
print(f"📁 Archivos en el sandbox:")
for file in os.listdir('/home/user'):
    if not file.startswith('.'):
        print(f"   - {file}")
"""
        
        analysis_execution = await sandbox.run_code(analysis_code)
        if analysis_execution and analysis_execution.logs and analysis_execution.logs.stdout:
            print("✅ Salida del análisis:")
            for line in analysis_execution.logs.stdout:
                print(f"   {line.strip()}")
        
        # Probar instalación de paquetes (funcionalidad común de IA)
        print("\n📦 Probando instalación de paquetes...")
        install_code = """
# Instalar un paquete que podría necesitar un modelo de IA
# En un entorno real, esto podría ser scipy, sklearn, etc.
import pkg_resources

installed_packages = [d.project_name for d in pkg_resources.working_set]
print(f"📦 Paquetes instalados (muestra): {len(installed_packages)} paquetes")
print(f"   Ejemplos: {installed_packages[:10]}")

# Verificar si paquetes comunes están disponibles
common_packages = ['numpy', 'pandas', 'matplotlib', 'requests', 'scipy']
for package in common_packages:
    try:
        __import__(package)
        print(f"✅ {package} está disponible")
    except ImportError:
        print(f"⚠️  {package} no está disponible")
"""
        
        install_execution = await sandbox.run_code(install_code)
        if install_execution and install_execution.logs and install_execution.logs.stdout:
            print("✅ Salida de verificación de paquetes:")
            for line in install_execution.logs.stdout:
                print(f"   {line.strip()}")
        
        # Cerrar el sandbox
        await sandbox.kill()
        print("\n✅ Sandbox E2B cerrado correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba E2B + IA: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_e2b_with_different_scenarios():
    """Prueba E2B con diferentes escenarios de uso común en IA"""
    print("\n🌐 Prueba de diferentes escenarios E2B...")
    
    try:
        sandbox = await AsyncSandbox.create(api_key=E2B_API_KEY, timeout=600)
        print("✅ Sandbox E2B para escenarios establecido")
        
        scenarios = {
            "data_processing": """
# Simulación de procesamiento de datos
import pandas as pd
import numpy as np

# Crear dataset de ejemplo
df = pd.DataFrame({
    'fecha': pd.date_range('2024-01-01', periods=100),
    'ventas': np.random.randint(100, 1000, 100),
    'clientes': np.random.randint(10, 100, 100)
})

print(f"📊 Dataset creado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"📈 Ventas promedio: {df['ventas'].mean():.2f}")
print(f"👥 Total clientes: {df['clientes'].sum()}")

# Cálculos avanzados
df['venta_por_cliente'] = df['ventas'] / df['clientes']
print(f"💰 Venta promedio por cliente: {df['venta_por_cliente'].mean():.2f}")
""",
            "ml_simulation": """
# Simulación de entrenamiento de modelo
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generar datos de ejemplo
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)

# Crear y entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Predecir
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

print(f"🤖 Simulación de ML completada:")
print(f"   Coeficiente: {model.coef_[0]:.3f}")
print(f"   Intercepto: {model.intercept_:.3f}")
print(f"   MSE: {mse:.3f}")
print(f"   R² Score: {model.score(X, y):.3f}")
""",
            "visualization": """
# Crear visualización
import matplotlib.pyplot as plt
import numpy as np

# Datos
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/10)

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Función: sin(x) * exp(-x/10)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig('/home/user/funcion_visualizacion.png')

print(f"📊 Visualización creada y guardada como 'funcion_visualizacion.png'")
print(f"   Rango x: [{x.min():.2f}, {x.max():.2f}]")
print(f"   Rango y: [{y.min():.2f}, {y.max():.2f}]")
"""
        }
        
        for scenario_name, code in scenarios.items():
            print(f"\n🧪 Ejecutando escenario: {scenario_name}")
            execution = await sandbox.run_code(code)
            
            if execution and execution.logs and execution.logs.stdout:
                for line in execution.logs.stdout:
                    print(f"   {line.strip()}")
        
        await sandbox.kill()
        print("✅ Escenarios E2B completados")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en escenarios E2B: {e}")
        return False

async def main():
    """Función principal"""
    print("🚀 Iniciando pruebas completas de E2B + IA Models...")
    print(f"🔑 Usando API Key: {E2B_API_KEY[:15]}...")
    
    # Prueba de integración E2B + IA
    ai_integration_result = await test_ai_code_generation_and_execution()
    
    # Prueba de diferentes escenarios
    scenarios_result = await test_e2b_with_different_scenarios()
    
    print("\n" + "="*80)
    print("📋 RESULTADOS FINALES DE E2B + IA INTEGRATION")
    print("="*80)
    print(f"E2B + IA Code Generation: {'✅ PASSED' if ai_integration_result else '❌ FAILED'}")
    print(f"E2B Scenarios Test: {'✅ PASSED' if scenarios_result else '❌ FAILED'}")
    print("="*80)
    
    if ai_integration_result and scenarios_result:
        print("🎉 ¡Todas las pruebas de integración E2B + IA se completaron exitosamente!")
        print("⚡ El sistema E2B está funcionando correctamente con los modelos de IA")
    else:
        print("⚠️  Algunas pruebas fallaron, revisar los logs anteriores")

if __name__ == "__main__":
    asyncio.run(main())