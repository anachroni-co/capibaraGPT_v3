#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2B Integration Module for Capibara6
Módulo de integración con E2B para ejecución segura de código generado por IA
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from e2b_code_interpreter import AsyncSandbox
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class E2BManager:
    """Gestor de la integración con E2B para capibara6"""
    
    def __init__(self, api_key: Optional[str] = None, default_timeout: int = 300):
        """
        Inicializa el gestor E2B
        
        Args:
            api_key: API key de E2B (si no se proporciona, se usa la de entorno)
            default_timeout: Timeout por defecto para los sandboxes (en segundos)
        """
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            raise ValueError("E2B_API_KEY no encontrada en variables de entorno")
        
        self.default_timeout = default_timeout
        self.active_sandboxes = {}
        logger.info("E2BManager inicializado correctamente")
    
    async def create_sandbox(self, timeout: Optional[int] = None, template: str = "code-interpreter-v1"):
        """
        Crea un nuevo sandbox E2B
        
        Args:
            timeout: Timeout para este sandbox (en segundos)
            template: Plantilla a usar para el sandbox
            
        Returns:
            AsyncSandbox: Instancia del sandbox creado
        """
        if not timeout:
            timeout = self.default_timeout
        
        sandbox = await AsyncSandbox.create(
            api_key=self.api_key,
            template=template,
            timeout=timeout
        )
        
        sandbox_id = sandbox.sandbox_id
        self.active_sandboxes[sandbox_id] = {
            'instance': sandbox,
            'created_at': datetime.now(),
            'template': template,
            'timeout': timeout
        }
        
        logger.info(f"Sandbox creado: {sandbox_id}")
        return sandbox
    
    async def execute_code(self, code: str, language: str = "python", 
                          sandbox_timeout: Optional[int] = None,
                          context: Optional[Dict[str, Any]] = None):
        """
        Ejecuta código en un sandbox E2B
        
        Args:
            code: Código a ejecutar
            language: Lenguaje del código (actualmente solo Python en E2B)
            sandbox_timeout: Timeout para este sandbox
            context: Contexto adicional para la ejecución
            
        Returns:
            Dict: Resultados de la ejecución
        """
        sandbox = None
        try:
            logger.info(f"Iniciando ejecución de código de {len(code)} caracteres")
            
            # Crear sandbox temporal
            sandbox = await self.create_sandbox(timeout=sandbox_timeout)
            
            # Determinar el tipo de ejecución basado en el lenguaje
            if language.lower() in ["python", "py"]:
                execution = await sandbox.run_code(code)
            else:
                # Para otros lenguajes, ejecutar como Python (E2B principalment soporta Python)
                execution = await sandbox.run_code(code)
            
            # Procesar resultados
            result = {
                'success': True,
                'sandbox_id': sandbox.sandbox_id,
                'execution_time': (datetime.now() - self.active_sandboxes[sandbox.sandbox_id]['created_at']).total_seconds(),
                'logs': {
                    'stdout': [line.rstrip() for line in execution.logs.stdout] if execution.logs.stdout else [],
                    'stderr': [line.rstrip() for line in execution.logs.stderr] if execution.logs.stderr else []
                },
                'result': execution.results,
                'error': execution.error.message if execution.error else None,
                'timestamp': datetime.now().isoformat()
            }
            
            if execution.error:
                result['success'] = False
                logger.warning(f"Ejecución con error: {execution.error.message}")
            else:
                logger.info(f"Ejecución completada exitosamente en sandbox {sandbox.sandbox_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error ejecutando código: {e}")
            return {
                'success': False,
                'error': str(e),
                'sandbox_id': sandbox.sandbox_id if sandbox else None,
                'timestamp': datetime.now().isoformat()
            }
        finally:
            # Si se creó un sandbox, asegurarse de destruirlo
            if sandbox:
                try:
                    await sandbox.kill()
                    sandbox_id = sandbox.sandbox_id
                    if sandbox_id in self.active_sandboxes:
                        del self.active_sandboxes[sandbox_id]
                    logger.info(f"Sandbox destruido: {sandbox_id}")
                except Exception as e:
                    logger.error(f"Error destruyendo sandbox: {e}")
    
    async def execute_analysis(self, data_description: str, analysis_request: str):
        """
        Ejecuta un análisis de datos en E2B
        
        Args:
            data_description: Descripción de los datos disponibles
            analysis_request: Solicitud específica de análisis
            
        Returns:
            Dict: Resultados del análisis
        """
        # Generar código de análisis basado en la solicitud
        analysis_code = f"""
import pandas as pd
import numpy as np
import json

# Simular datos basados en la descripción
# En una implementación real, estos datos vendrían de la base de datos o archivos
print("Ejecutando análisis: {analysis_request}")
print("Datos disponibles: {data_description}")

# Simular un análisis de datos
sample_data = {{
    'metric1': np.random.rand(10) * 100,
    'metric2': np.random.rand(10) * 50,
    'category': [f'Cat{{i}}' for i in range(10)]
}}
df = pd.DataFrame(sample_data)

# Análisis básico
results = {{
    'mean_metric1': float(df['metric1'].mean()),
    'mean_metric2': float(df['metric2'].mean()),
    'total_records': len(df),
    'categories': df['category'].tolist()
}}

print(f"Análisis completado. Resultados: {{results}}")
results
"""
        
        return await self.execute_code(analysis_code, "python")
    
    async def health_check(self):
        """
        Verifica la salud del sistema E2B
        
        Returns:
            Dict: Estado del sistema E2B
        """
        try:
            # Probar conexión con un comando simple
            test_result = await self.execute_code("print('E2B connection OK')", "python", sandbox_timeout=30)
            
            return {
                'status': 'healthy' if test_result['success'] else 'unhealthy',
                'api_key_valid': bool(self.api_key),
                'test_execution': test_result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Limpia todos los sandboxes activos"""
        logger.info("Iniciando limpieza de sandboxes...")
        
        for sandbox_id, sandbox_info in list(self.active_sandboxes.items()):
            try:
                await sandbox_info['instance'].kill()
                del self.active_sandboxes[sandbox_id]
                logger.info(f"Sandbox limpiado: {sandbox_id}")
            except Exception as e:
                logger.error(f"Error limpiando sandbox {sandbox_id}: {e}")
        
        logger.info("Limpieza completada")


class E2BIntegration:
    """Integración completa de E2B para el sistema capibara6"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Inicializa la integración E2B"""
        self.e2b_manager = E2BManager(api_key)
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0
        }
        logger.info("E2BIntegration inicializado")
    
    async def process_code_request(self, code: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Procesa una solicitud de ejecución de código
        
        Args:
            code: Código a ejecutar
            metadata: Metadatos adicionales de la solicitud
            
        Returns:
            Dict: Resultados del procesamiento
        """
        start_time = datetime.now()
        
        logger.info(f"Procesando solicitud de ejecución de código")
        result = await self.e2b_manager.execute_code(code)
        
        # Actualizar estadísticas
        self.execution_stats['total_executions'] += 1
        if result['success']:
            self.execution_stats['successful_executions'] += 1
        else:
            self.execution_stats['failed_executions'] += 1
        
        execution_time = (datetime.now() - start_time).total_seconds()
        self.execution_stats['total_execution_time'] += execution_time
        
        # Añadir estadísticas al resultado
        result['execution_stats'] = self.execution_stats.copy()
        result['request_metadata'] = metadata or {}
        
        return result
    
    async def get_execution_stats(self):
        """Obtiene estadísticas de ejecución"""
        if self.execution_stats['total_executions'] > 0:
            avg_execution_time = self.execution_stats['total_execution_time'] / self.execution_stats['total_executions']
        else:
            avg_execution_time = 0.0
        
        success_rate = 0.0
        if self.execution_stats['total_executions'] > 0:
            success_rate = (self.execution_stats['successful_executions'] / 
                           self.execution_stats['total_executions']) * 100
        
        return {
            'execution_stats': self.execution_stats,
            'average_execution_time': avg_execution_time,
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self):
        """Realiza un health check del sistema E2B"""
        return await self.e2b_manager.health_check()
    
    async def cleanup(self):
        """Limpia recursos"""
        await self.e2b_manager.cleanup()
    
    async def execute_query(self, query: str):
        """
        Método para ejecutar una query que puede incluir código.
        Este método se usa cuando el sistema principal detecta que la query
        contiene contenido que debe ejecutarse en E2B.
        """
        # Detectar si la query contiene código
        if self._contains_code(query):
            # Si contiene código, ejecutarlo directamente
            return await self.process_code_request(query)
        else:
            # Si no contiene código, no hacer nada
            return {
                'success': False,
                'message': 'No code found to execute',
                'query': query
            }
    
    def _contains_code(self, query: str) -> bool:
        """Detecta si una query contiene código para ejecutar."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'print(',
            'console.log', 'function ', 'var ', 'let ', 'const ',
            'SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ', 'CREATE ',
            '```python', '```javascript', '```sql', '```bash',
            'if __name__ ==', 'for ', 'while ', 'import',
            '#!', 'pip install', 'npm install'
        ]
        
        query_lower = query.lower()
        return any(indicator.lower() in query_lower for indicator in code_indicators)


# Función de ejemplo para probar la integración
async def example_usage():
    """Ejemplo de uso del sistema E2B con capibara6"""
    
    # Inicializar la integración
    e2b_integration = E2BIntegration()
    
    # Ejemplo 1: Ejecutar código simple
    print("1. Ejecutando código simple...")
    result1 = await e2b_integration.process_code_request("""
import numpy as np
a = np.array([1, 2, 3, 4, 5])
result = a * 2
print(f"Resultado: {result}")
result
""")
    
    print(f"Resultado: {result1}")
    
    # Ejemplo 2: Análisis de datos
    print("\n2. Ejecutando análisis de datos...")
    result2 = await e2b_integration.process_code_request("""
import pandas as pd
import numpy as np

# Crear datos de ejemplo
data = {
    'ventas': np.random.randint(100, 1000, 20),
    'clientes': np.random.randint(10, 100, 20),
    'mes': [f'Mes{i+1}' for i in range(20)]
}
df = pd.DataFrame(data)

# Análisis
promedio_ventas = df['ventas'].mean()
promedio_clientes = df['clientes'].mean()

print(f"Promedio de ventas: {promedio_ventas:.2f}")
print(f"Promedio de clientes: {promedio_clientes:.2f}")
print(f"Total registros: {len(df)}")

# Regresar resultados
{
    'promedio_ventas': float(promedio_ventas),
    'promedio_clientes': float(promedio_clientes),
    'total_registros': len(df)
}
""")
    
    print(f"Resultado del análisis: {result2}")
    
    # Mostrar estadísticas
    print("\n3. Estadísticas de ejecución:")
    stats = await e2b_integration.get_execution_stats()
    print(f"Estadísticas: {stats}")
    
    # Health check
    print("\n4. Health check:")
    health = await e2b_integration.health_check()
    print(f"Salud: {health}")
    
    # Limpieza
    await e2b_integration.cleanup()
    print("\n5. Recursos limpiados")


if __name__ == "__main__":
    print("🧪 Iniciando ejemplo de integración E2B para capibara6...")
    asyncio.run(example_usage())
    print("✅ Ejemplo completado")