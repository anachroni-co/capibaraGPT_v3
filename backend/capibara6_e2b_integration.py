"""
Ejemplo de integración de e2b con el sistema Capibara6.
Este archivo demuestra cómo usar e2b para crear VMs rápidas en respuesta
a diferentes tipos de tareas clasificadas por el sistema CTM.
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any

# Agregar el directorio backend al path para importar correctamente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from e2b_manager import E2BManager, create_e2b_manager
from models_config import MODELS_CONFIG

# Usar la clave API proporcionada
E2B_API_TOKEN = "e2b_d8df23b5de5214b7bfb4ebe227a308b61a2ae172"

# Importar funciones desde el archivo utils.py principal en el directorio backend
def _import_utils():
    """Función para importar utils de forma diferida"""
    import importlib.util
    import sys

    # Obtener la ruta del archivo utils.py
    utils_spec = importlib.util.spec_from_file_location("utils", "/home/elect/capibara6/backend/utils.py")
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules["utils"] = utils_module
    utils_spec.loader.exec_module(utils_module)

    return (
        utils_module.analyze_context,
        utils_module.understand_query,
        utils_module.determine_action,
        utils_module.calculate_relevance
    )

class Capibara6E2BIntegration:
    """
    Integración de e2b con el sistema Capibara6 para generar VMs rápidas
    según las necesidades de las tareas clasificadas por CTM.
    """
    
    def __init__(self):
        self.e2b_manager = create_e2b_manager(E2B_API_TOKEN)
        # Importar funciones en la inicialización
        self.analyze_context, self.understand_query, self.determine_action, self.calculate_relevance = _import_utils()

    async def handle_complex_task_with_e2b(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """
        Maneja tareas complejas usando e2b para crear entornos aislados.

        :param prompt: Prompt original del usuario
        :param context: Contexto adicional
        :return: Resultado de la operación
        """
        print(f"Procesando tarea compleja con e2b: {prompt[:50]}...")

        # Determinar qué tipo de tarea es y qué recursos necesita
        query_analysis = self.understand_query(prompt)
        context_analysis = self.analyze_context(context)
        action_recommendation = self.determine_action(context, prompt)
        
        # Basado en el análisis, elegir la plantilla más adecuada
        template = self._select_template_based_on_analysis(query_analysis, context_analysis, action_recommendation)
        
        # Crear un entorno e2b para procesar la tarea
        try:
            # Ejecutar una operación específica basada en la tarea
            result = await self._execute_task_in_e2b(prompt, template)
            
            return {
                "success": True,
                "e2b_result": result,
                "task_analysis": {
                    "query_analysis": query_analysis,
                    "context_analysis": context_analysis,
                    "action_recommendation": action_recommendation,
                    "selected_template": template
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_analysis": {
                    "query_analysis": query_analysis,
                    "context_analysis": context_analysis,
                    "action_recommendation": action_recommendation,
                    "selected_template": template
                }
            }
    
    def _select_template_based_on_analysis(self, query_analysis, context_analysis, action_recommendation) -> str:
        """
        Selecciona la plantilla e2b más adecuada basada en el análisis de la tarea.
        
        :param query_analysis: Análisis de la consulta
        :param context_analysis: Análisis del contexto
        :param action_recommendation: Recomendación de acción
        :return: Nombre de la plantilla e2b
        """
        # Basado en la complejidad y tipo de tarea, seleccionar la plantilla adecuada
        if query_analysis.get('complexity') == 'high':
            if 'code' in query_analysis.get('query_lower', '') or 'programming' in query_analysis.get('query_lower', ''):
                return "python3"  # Plantilla con Python
            else:
                return "python3"  # Plantilla por defecto para tareas complejas
        
        # Para tareas analíticas específicas
        if any(word in str(query_analysis).lower() for word in ['data', 'analysis', 'calculate', 'compute']):
            return "python3"  # Plantilla con Python y herramientas de análisis
        
        # Para tareas que requerirían herramientas específicas
        if 'web' in str(query_analysis).lower() or 'url' in str(query_analysis).lower():
            return "python3"  # Plantilla con Python para scrapers, etc.
        
        # Por defecto
        return "python3"
    
    async def _execute_task_in_e2b(self, prompt: str, template: str) -> Dict[str, Any]:
        """
        Ejecuta una tarea específica en un entorno e2b.
        
        :param prompt: Prompt a procesar
        :param template: Plantilla e2b a usar
        :return: Resultado de la ejecución
        """
        # Crear un script Python para procesar la tarea
        python_script = self._create_processing_script(prompt)
        
        # Extraer la ruta de visualización del script para saber qué archivos buscar
        visualization_path = None
        # Procesar la salida para identificar rutas de archivos
        import json
        try:
            # Procesar la salida esperando un formato JSON que incluya visualization_path
            lines = python_script.split('\\n')
            # Buscar posibles rutas de archivos en el script
            for line in lines:
                if 'visualization_path' in line and '/home/user/' in line:
                    # Esta lógica se mejorará cuando implementemos la búsqueda real
                    pass
        except:
            pass
        
        # Ejecutar el script en la sandbox e2b
        result = await self.e2b_manager.run_python_code(
            code=python_script,
            template=template,
            timeout=180,  # 3 minutos de timeout
            output_files=['/home/user/visualization.png', '/home/user/data.csv', '/home/user/map.png']  # Archivos potenciales de visualización
        )
        
        return result
    
    async def _find_visualization_files(self, sandbox_id: str) -> list:
        """
        Busca archivos de visualización generados en el sandbox (imágenes, etc.)
        
        :param sandbox_id: ID del sandbox
        :return: Lista de archivos de visualización
        """
        if not sandbox_id:
            return []
        
        try:
            # Este es un ejemplo simplificado - en la práctica, necesitaríamos
            # acceder al sandbox para buscar archivos de visualización
            # En lugar de acceder directamente al sandbox, vamos a mejorar el script
            # para que capture automáticamente las visualizaciones
            return []
        except Exception as e:
            print(f"Error buscando archivos de visualización: {e}")
            return []
    
    def _create_processing_script(self, prompt: str) -> str:
        """
        Crea un script Python para procesar la tarea.
        
        :param prompt: Prompt a procesar
        :return: Código Python como string
        """
        # Basado en el tipo de tarea, crear un script apropiado
        script = f"""
import sys
import time
import json
import os
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para gráficos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def process_task():
    print(f"Procesando tarea: {{repr('{prompt}'[:100])}}...")
    
    # Aquí iría la lógica específica para procesar la tarea
    start_time = time.time()
    
    # Detectar el tipo de tarea para decidir si generar visualizaciones
    task_type = 'general'
    if any(word in '{prompt}'.lower() for word in ['grafico', 'gráfico', 'gráfico de', 'gráfica', 'visualizar', 'visualización', 'plot', 'chart', 'plotear']):
        task_type = 'visualization'
    elif any(word in '{prompt}'.lower() for word in ['datos', 'data', 'tabla', 'análisis', 'dataset', 'csv']):
        task_type = 'data'
    elif any(word in '{prompt}'.lower() for word in ['mapa', 'map', 'geográfico', 'ubicación']):
        task_type = 'map'
    
    # Simulación de procesamiento de tarea basado en tipo
    if task_type == 'visualization':
        # Generar un gráfico simple de ejemplo
        try:
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y)
            plt.title('Visualización generada en e2b sandbox')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            
            # Guardar la imagen
            img_path = '/home/user/visualization.png'
            plt.savefig(img_path)
            plt.close()
            
            result = f"Gráfico generado exitosamente en {{img_path}}"
            visualization_path = img_path
        except Exception as e:
            result = f"Error generando visualización: {{str(e)}}"
            visualization_path = None
    elif task_type == 'data':
        # Generar datos de ejemplo
        try:
            data = {{
                'x': list(range(10)),
                'y': [i**2 for i in range(10)]
            }}
            df = pd.DataFrame(data)
            
            # Guardar como CSV
            csv_path = '/home/user/data.csv'
            df.to_csv(csv_path, index=False)
            
            result = f"Datos generados y guardados en {{csv_path}}"
            visualization_path = csv_path
        except Exception as e:
            result = f"Error generando datos: {{str(e)}}"
            visualization_path = None
    elif task_type == 'map':
        # Simular generación de mapa
        try:
            # Aquí normalmente usaríamos una librería como folium para generar mapas
            # Por ahora simulamos con un gráfico simple
            plt.figure(figsize=(10, 8))
            plt.scatter([0, 1, 2], [0, 1, 2])
            plt.title('Mapa simulado en e2b sandbox')
            plt.xlabel('Longitud')
            plt.ylabel('Latitud')
            
            # Guardar la imagen
            img_path = '/home/user/map.png'
            plt.savefig(img_path)
            plt.close()
            
            result = f"Mapa generado exitosamente en {{img_path}}"
            visualization_path = img_path
        except Exception as e:
            result = f"Error generando mapa: {{str(e)}}"
            visualization_path = None
    else:
        # Procesamiento general
        if 'calculate' in '{prompt}'.lower() or 'math' in '{prompt}'.lower():
            try:
                # Intentar evaluar expresiones matemáticas sencillas
                if any(op in '{prompt}' for op in ['+', '-', '*', '/', '^']):
                    # Esta es solo una simulación, no evaluar expresiones directamente en producción
                    result = "Procesamiento matemático simulado"
                else:
                    result = "Tarea procesada en entorno aislado con e2b"
            except:
                result = "No se pudo evaluar la expresión matemática"
        elif 'python' in '{prompt}'.lower() or 'code' in '{prompt}'.lower() or 'program' in '{prompt}'.lower():
            result = "Entorno de programación Python disponible para desarrollar la solución"
        else:
            result = "Tarea procesada en entorno aislado con e2b"
        
        visualization_path = None
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    response = {{
        "result": result,
        "processing_time": processing_time,
        "task": '{prompt}'[:100],
        "status": "completed",
        "visualization_path": visualization_path  # Ruta del archivo de visualización si existe
    }}
    
    print(json.dumps(response))

if __name__ == "__main__":
    process_task()
"""

        return script

    def estimate_task_resources(self, prompt: str) -> Dict[str, Any]:
        """
        Estima los recursos necesarios para una tarea.

        :param prompt: Prompt a analizar
        :return: Estimación de recursos
        """
        # Importar funciones si no se han importado aún
        if not hasattr(self, 'understand_query'):
            self.analyze_context, self.understand_query, self.determine_action, self.calculate_relevance = _import_utils()

        query_analysis = self.understand_query(prompt)

        # Determinar la complejidad y tipo de recursos necesarios
        if query_analysis.get('complexity') == 'high':
            complexity_level = 'high'
            estimated_runtime = 180  # 3 minutos
            recommended_template = 'python3'
        elif query_analysis.get('is_question') and query_analysis.get('complexity') == 'medium':
            complexity_level = 'medium'
            estimated_runtime = 90   # 1.5 minutos
            recommended_template = 'python3'
        else:
            complexity_level = 'low'
            estimated_runtime = 30   # 30 segundos
            recommended_template = 'python3'

        return {
            "complexity_level": complexity_level,
            "estimated_runtime": estimated_runtime,
            "recommended_template": recommended_template,
            "requires_isolation": True,  # Todas las tareas complejas deberían usar e2b
            "query_analysis": query_analysis
        }

# Función para inicializar la integración
def init_e2b_integration():
    """
    Inicializa la integración de e2b con Capibara6.
    """
    return Capibara6E2BIntegration()

# Ejemplo de uso
async def main():
    print("Inicializando integración e2b con Capibara6...")
    
    # Inicializar la integración
    integration = init_e2b_integration()
    
    # Ejemplo de tarea compleja
    complex_task = "Analiza el siguiente conjunto de datos y crea un modelo predictivo: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    
    print(f"Procesando tarea: {complex_task}")
    
    # Procesar la tarea con e2b
    result = await integration.handle_complex_task_with_e2b(complex_task)
    
    print("Resultado:")
    print(json.dumps(result, indent=2))
    
    # Ejemplo de estimación de recursos
    print("\nEstimación de recursos para la tarea:")
    resources = integration.estimate_task_resources(complex_task)
    print(json.dumps(resources, indent=2))

if __name__ == "__main__":
    asyncio.run(main())