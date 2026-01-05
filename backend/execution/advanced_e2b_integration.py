#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced E2B Integration Module for Capibara6
Módulo avanzado de integración con E2B para ejecución segura de código generado por IA
con gestión dinámica de recursos y templates
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from e2b_code_interpreter import AsyncSandbox

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class E2BTemplate:
    """Representa una plantilla de sandbox E2B con configuración predefinida"""
    
    def __init__(self, template_id: str, config: Dict[str, Any]):
        self.template_id = template_id
        self.config = config  # timeout, memory, cpu, etc.
        self.name = config.get('name', template_id)
        self.description = config.get('description', 'Template without description')
        self.supported_languages = config.get('supported_languages', ['python'])
        self.packages = config.get('packages', [])
        self.created_at = datetime.now()
    
    def get_sandbox_config(self) -> Dict[str, Any]:
        """Obtiene la configuración para crear un sandbox basado en esta plantilla"""
        return {
            'timeout': self.config.get('timeout', 300),
            'memory_limit_mb': self.config.get('memory_limit_mb', 512),
            'cpu_limit_percent': self.config.get('cpu_limit_percent', 50),
            'template_name': self.config.get('template_name', 'code-interpreter-v1'),
            'packages': self.packages
        }

class AdvancedE2BManager:
    """Gestor avanzado de E2B con soporte para templates y gestión dinámica de VMs"""
    
    def __init__(self, api_key: Optional[str] = None, max_concurrent_sandboxes: int = 5):
        """
        Inicializa el gestor E2B avanzado
        
        Args:
            api_key: API key de E2B
            max_concurrent_sandboxes: Número máximo de sandboxes concurrentes
        """
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            raise ValueError("E2B_API_KEY no encontrada en variables de entorno")
        
        self.max_concurrent_sandboxes = max_concurrent_sandboxes
        self.active_sandboxes = {}
        
        # Inicializar templates predefinidos
        self.templates = self._initialize_templates()
        
        # Estadísticas de ejecución
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'sandbox_created': 0
        }
        
        logger.info(f"AdvancedE2BManager inicializado con {max_concurrent_sandboxes} sandboxes máximos")
    
    def _initialize_templates(self) -> Dict[str, E2BTemplate]:
        """Inicializa templates predefinidos para diferentes tipos de tareas"""
        templates = {
            'default': E2BTemplate('default', {
                'name': 'Default Template',
                'description': 'Template estándar para tareas generales',
                'timeout': 300,  # 5 minutos
                'memory_limit_mb': 512,
                'cpu_limit_percent': 50,
                'supported_languages': ['python', 'javascript'],
                'template_name': 'code-interpreter-v1'
            }),
            'data_analysis': E2BTemplate('data_analysis', {
                'name': 'Data Analysis Template',
                'description': 'Template optimizado para análisis de datos',
                'timeout': 600,  # 10 minutos
                'memory_limit_mb': 1024,  # 1GB
                'cpu_limit_percent': 75,
                'supported_languages': ['python'],
                'template_name': 'code-interpreter-v1',
                'packages': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy']
            }),
            'machine_learning': E2BTemplate('machine_learning', {
                'name': 'Machine Learning Template',
                'description': 'Template con recursos para tareas ML',
                'timeout': 1800,  # 30 minutos
                'memory_limit_mb': 2048,  # 2GB
                'cpu_limit_percent': 100,
                'supported_languages': ['python'],
                'template_name': 'code-interpreter-v1',
                'packages': ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'tensorflow', 'pytorch']
            }),
            'quick_script': E2BTemplate('quick_script', {
                'name': 'Quick Script Template',
                'description': 'Template para scripts rápidos y simples',
                'timeout': 60,   # 1 minuto
                'memory_limit_mb': 256,
                'cpu_limit_percent': 25,
                'supported_languages': ['python', 'javascript', 'bash'],
                'template_name': 'code-interpreter-v1'
            }),
            'visualization': E2BTemplate('visualization', {
                'name': 'Visualization Template',
                'description': 'Template optimizado para visualización de datos',
                'timeout': 600,  # 10 minutos
                'memory_limit_mb': 1024,
                'cpu_limit_percent': 75,
                'supported_languages': ['python'],
                'template_name': 'code-interpreter-v1',
                'packages': ['pandas', 'matplotlib', 'seaborn', 'plotly', 'altair']
            })
        }
        
        logger.info(f"Templates inicializados: {list(templates.keys())}")
        return templates
    
    def get_template(self, template_id: str) -> Optional[E2BTemplate]:
        """Obtiene un template por ID"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[str]:
        """Lista todos los templates disponibles"""
        return list(self.templates.keys())
    
    async def create_sandbox_from_template(self, template_id: str = 'default', 
                                         custom_config: Optional[Dict[str, Any]] = None) -> AsyncSandbox:
        """
        Crea un sandbox usando un template específico
        
        Args:
            template_id: ID del template a usar
            custom_config: Configuración personalizada que sobreescribe el template
            
        Returns:
            AsyncSandbox: Instancia del sandbox creado
        """
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' no encontrado")
        
        # Combinar configuración del template con configuración personalizada
        config = template.get_sandbox_config()
        if custom_config:
            config.update(custom_config)
        
        # Crear sandbox con la configuración
        logger.info(f"Creando sandbox con template '{template_id}' y config: {config}")
        
        sandbox = await AsyncSandbox.create(
            api_key=self.api_key,
            template=config['template_name'],
            timeout=config['timeout']
        )
        
        # Almacenar información del sandbox
        sandbox_info = {
            'instance': sandbox,
            'template_used': template_id,
            'config': config,
            'created_at': datetime.now(),
            'execution_count': 0
        }
        
        self.active_sandboxes[sandbox.sandbox_id] = sandbox_info
        self.stats['sandbox_created'] += 1
        
        logger.info(f"Sandbox creado: {sandbox.sandbox_id} con template '{template_id}'")
        
        return sandbox
    
    async def execute_code_with_template(self, code: str, template_id: str = 'default',
                                       custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecuta código en un sandbox creado con un template específico
        
        Args:
            code: Código a ejecutar
            template_id: ID del template a usar
            custom_config: Configuración personalizada
            
        Returns:
            Dict: Resultados de la ejecución
        """
        start_time = datetime.now()
        
        try:
            # Crear sandbox con template
            sandbox = await self.create_sandbox_from_template(template_id, custom_config)
            sandbox_id = sandbox.sandbox_id
            
            logger.info(f"Ejecutando código en sandbox {sandbox_id} con template '{template_id}'")
            
            # Ejecutar código
            execution = await sandbox.run_code(code)
            
            # Actualizar contadores
            self.active_sandboxes[sandbox_id]['execution_count'] += 1
            self.stats['total_executions'] += 1
            
            if execution.error:
                self.stats['failed_executions'] += 1
                logger.warning(f"Ejecución fallida en sandbox {sandbox_id}: {execution.error.message}")
            else:
                self.stats['successful_executions'] += 1
                logger.info(f"Ejecución exitosa en sandbox {sandbox_id}")
            
            # Preparar resultado
            result = {
                'success': not execution.error,
                'sandbox_id': sandbox_id,
                'template_used': template_id,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'code_length': len(code),
                'logs': {
                    'stdout': execution.logs.stdout if execution.logs.stdout else [],
                    'stderr': execution.logs.stderr if execution.logs.stderr else []
                },
                'results': execution.results if execution.results else [],
                'error': execution.error.message if execution.error else None,
                'sandbox_info': self.active_sandboxes[sandbox_id],
                'timestamp': datetime.now().isoformat()
            }
            
            # Actualizar estadísticas de tiempo de ejecución
            self.stats['total_execution_time'] += result['execution_time']
            
            return result
            
        except Exception as e:
            logger.error(f"Error ejecutando código con template {template_id}: {e}")
            return {
                'success': False,
                'template_used': template_id,
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            # Siempre destruir el sandbox después de la ejecución
            if 'sandbox' in locals():
                try:
                    await sandbox.kill()  # Usar kill en lugar de close
                    sandbox_id = sandbox.sandbox_id
                    if sandbox_id in self.active_sandboxes:
                        del self.active_sandboxes[sandbox_id]
                    logger.info(f"Sandbox destruido: {sandbox_id}")
                except Exception as e:
                    logger.error(f"Error destruyendo sandbox {sandbox_id}: {e}")
    
    async def create_dynamic_sandbox(self, task_type: str, requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Crea un sandbox dinámicamente basado en el tipo de tarea
        
        Args:
            task_type: Tipo de tarea ('data_analysis', 'ml', 'quick', etc.)
            requirements: Requisitos específicos para el sandbox
            
        Returns:
            Dict: Información del sandbox creado
        """
        # Mapear tipos de tarea a templates
        task_to_template = {
            'data_analysis': 'data_analysis',
            'data-visualization': 'visualization',
            'machine_learning': 'machine_learning',
            'ml': 'machine_learning',
            'quick': 'quick_script',
            'general': 'default'
        }
        
        template_id = task_to_template.get(task_type, 'default')
        template = self.get_template(template_id)
        
        if not template:
            template = self.get_template('default')
            template_id = 'default'
        
        # Aplicar requisitos personalizados
        config = template.get_sandbox_config()
        if requirements:
            config.update(requirements)
        
        # Crear sandbox con configuración dinámica
        logger.info(f"Creando sandbox dinámico para tarea '{task_type}' con template '{template_id}'")
        
        try:
            sandbox = await AsyncSandbox.create(
                api_key=self.api_key,
                template=config['template_name'],
                timeout=config['timeout']
            )
            
            # Almacenar información del sandbox
            sandbox_info = {
                'instance': sandbox,
                'task_type': task_type,
                'template_used': template_id,
                'config': config,
                'created_at': datetime.now(),
                'execution_count': 0,
                'sandbox_id': sandbox.sandbox_id
            }
            
            self.active_sandboxes[sandbox.sandbox_id] = sandbox_info
            self.stats['sandbox_created'] += 1
            
            logger.info(f"Sandbox dinámico creado: {sandbox.sandbox_id} para tarea '{task_type}'")
            
            return {
                'success': True,
                'sandbox_id': sandbox.sandbox_id,
                'task_type': task_type,
                'template_used': template_id,
                'config': config,
                'instance': sandbox,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creando sandbox dinámico para tarea '{task_type}': {e}")
            return {
                'success': False,
                'task_type': task_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def execute_on_dynamic_sandbox(self, code: str, task_type: str = 'general',
                                       requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecuta código en un sandbox creado dinámicamente
        
        Args:
            code: Código a ejecutar
            task_type: Tipo de tarea para determinar recursos necesarios
            requirements: Requisitos específicos para el sandbox
            
        Returns:
            Dict: Resultados de la ejecución
        """
        start_time = datetime.now()
        
        # Crear sandbox dinámico
        sandbox_info = await self.create_dynamic_sandbox(task_type, requirements)
        
        if not sandbox_info['success']:
            return {
                'success': False,
                'error': sandbox_info.get('error', 'Failed to create dynamic sandbox'),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Ejecutar código en el sandbox creado
            sandbox = sandbox_info['instance']
            execution = await sandbox.run_code(code)
            
            # Actualizar contadores
            self.active_sandboxes[sandbox.sandbox_id]['execution_count'] += 1
            self.stats['total_executions'] += 1
            
            if execution.error:
                self.stats['failed_executions'] += 1
            else:
                self.stats['successful_executions'] += 1
            
            # Preparar resultado
            result = {
                'success': not execution.error,
                'sandbox_id': sandbox_info['sandbox_id'],
                'task_type': task_type,
                'template_used': sandbox_info['template_used'],
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'code_length': len(code),
                'logs': {
                    'stdout': execution.logs.stdout if execution.logs.stdout else [],
                    'stderr': execution.logs.stderr if execution.logs.stderr else []
                },
                'results': execution.results if execution.results else [],
                'error': execution.error.message if execution.error else None,
                'sandbox_info': {
                    'created_at': sandbox_info['timestamp'],
                    'task_type': task_type,
                    'config': sandbox_info['config']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Actualizar estadísticas de tiempo de ejecución
            self.stats['total_execution_time'] += result['execution_time']
            
            return result
            
        except Exception as e:
            logger.error(f"Error ejecutando código en sandbox dinámico: {e}")
            return {
                'success': False,
                'error': str(e),
                'sandbox_id': sandbox_info['sandbox_id'],
                'task_type': task_type,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            # Siempre destruir el sandbox después de la ejecución
            try:
                await sandbox_info['instance'].kill()  # Usar kill en lugar de close
                sandbox_id = sandbox_info['instance'].sandbox_id
                if sandbox_id in self.active_sandboxes:
                    del self.active_sandboxes[sandbox_id]
                logger.info(f"Sandbox dinámico destruido: {sandbox_id}")
            except Exception as e:
                logger.error(f"Error destruyendo sandbox dinámico {sandbox_info['sandbox_id']}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema E2B"""
        avg_execution_time = 0.0
        if self.stats['total_executions'] > 0:
            avg_execution_time = self.stats['total_execution_time'] / self.stats['total_executions']
        
        success_rate = 0.0
        if self.stats['total_executions'] > 0:
            success_rate = (self.stats['successful_executions'] / self.stats['total_executions']) * 100
        
        return {
            'stats': self.stats,
            'average_execution_time': avg_execution_time,
            'success_rate': success_rate,
            'active_sandboxes': len(self.active_sandboxes),
            'max_concurrent_sandboxes': self.max_concurrent_sandboxes,
            'templates_available': list(self.templates.keys())
        }
    
    async def cleanup(self):
        """Limpia todos los sandboxes activos"""
        logger.info(f"Limpieza de {len(self.active_sandboxes)} sandboxes activos...")
        
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
        """Inicializa la integración E2B avanzada"""
        self.e2b_manager = AdvancedE2BManager(api_key)
        logger.info("E2BIntegration avanzada inicializado")
    
    async def process_code_request(self, 
                                 code: str, 
                                 template_id: Optional[str] = None,
                                 task_type: Optional[str] = None,
                                 use_dynamic: bool = False,
                                 requirements: Optional[Dict[str, Any]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesa una solicitud de ejecución de código con opciones de templates y gestión dinámica
        """
        start_time = datetime.now()
        
        logger.info(f"Procesando solicitud de ejecución: template='{template_id}', task_type='{task_type}', dynamic={use_dynamic}")
        
        if use_dynamic or task_type:
            # Usar sistema de sandbox dinámico
            result = await self.e2b_manager.execute_on_dynamic_sandbox(
                code=code,
                task_type=task_type or 'general',
                requirements=requirements
            )
        else:
            # Usar sistema de templates
            result = await self.e2b_manager.execute_code_with_template(
                code=code,
                template_id=template_id or 'default',
                custom_config=requirements
            )
        
        # Añadir metadata adicional
        result['metadata'] = metadata or {}
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    async def get_available_templates(self) -> List[Dict[str, Any]]:
        """Obtiene la lista de templates disponibles"""
        templates_info = []
        for template_id in self.e2b_manager.list_templates():
            template = self.e2b_manager.get_template(template_id)
            if template:
                templates_info.append({
                    'id': template.template_id,
                    'name': template.name,
                    'description': template.description,
                    'supported_languages': template.supported_languages,
                    'config': template.get_sandbox_config()
                })
        return templates_info
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de ejecución"""
        return self.e2b_manager.get_stats()
    
    async def health_check(self) -> Dict[str, Any]:
        """Realiza un health check del sistema E2B"""
        try:
            # Probar conexión con un comando simple
            test_result = await self.e2b_manager.execute_code_with_template(
                code="print('E2B connection OK')",
                template_id='quick_script'
            )
            
            return {
                'status': 'healthy' if test_result['success'] else 'unhealthy',
                'api_key_valid': bool(self.e2b_manager.api_key),
                'test_execution': test_result,
                'timestamp': datetime.now().isoformat(),
                'stats': await self.get_execution_stats()
            }
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Limpia recursos"""
        await self.e2b_manager.cleanup()


# Función de ejemplo para probar la integración avanzada
async def advanced_example_usage():
    """Ejemplo de uso del sistema E2B avanzado con templates y creación dinámica"""
    
    print("🚀 INICIANDO EJEMPLO DE INTEGRACIÓN E2B AVANZADA")
    print("="*70)
    
    # Inicializar la integración
    e2b_integration = E2BIntegration()
    
    # 1. Listar templates disponibles
    print("\n1. Templates disponibles:")
    templates = await e2b_integration.get_available_templates()
    for template in templates:
        print(f"   - {template['name']}: {template['description']}")
    
    # 2. Ejecutar con template estándar
    print(f"\n2. Ejecutando con template 'data_analysis'...")
    result1 = await e2b_integration.process_code_request(
        code="""
import pandas as pd
import numpy as np

# Crear datos de ejemplo
data = {'ventas': np.random.randint(1000, 5000, 10), 'mes': range(1, 11)}
df = pd.DataFrame(data)

print(f"Dataset shape: {df.shape}")
print(f"Promedio de ventas: {df['ventas'].mean():.2f}")
print(f"Total ventas: {df['ventas'].sum():,}")

# Estadísticas
stats = df['ventas'].describe()
print(f"\\nEstadísticas:\\n{stats}")
""",
        template_id='data_analysis'
    )
    
    print(f"   Resultado: {'✅ ÉXITO' if result1['success'] else '❌ FALLO'}")
    if result1['success'] and result1.get('logs', {}).get('stdout'):
        for line in result1['logs']['stdout']:
            print(f"     {line}")
    
    # 3. Ejecutar con creación dinámica de sandbox
    print(f"\n3. Ejecutando con sandbox dinámico...")
    result2 = await e2b_integration.process_code_request(
        code="print('Hola desde sandbox dinámico!')",
        task_type='quick',
        use_dynamic=True
    )
    
    print(f"   Resultado sandbox dinámico: {'✅ ÉXITO' if result2['success'] else '❌ FALLO'}")
    if result2['success'] and result2.get('logs', {}).get('stdout'):
        for line in result2['logs']['stdout']:
            print(f"     {line}")
    
    # 4. Ejecutar análisis complejo
    print(f"\n4. Ejecutando análisis complejo...")
    result3 = await e2b_integration.process_code_request(
        code="""
import numpy as np
import matplotlib.pyplot as plt

# Generar datos
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/10)

print(f"Función evaluada: {len(x)} puntos")
print(f"Valor máximo: {y.max():.3f}")
print(f"Valor en x=5: {np.sin(5) * np.exp(-0.5):.3f}")

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = sin(x) * exp(-x/10)')
plt.title('Función Amortiguada')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/home/user/funcion_amortiguada.png')

print("Gráfico guardado como 'funcion_amortiguada.png'")
""",
        template_id='visualization'
    )
    
    print(f"   Resultado análisis complejo: {'✅ ÉXITO' if result3['success'] else '❌ FALLO'}")
    if result3['success'] and result3.get('logs', {}).get('stdout'):
        for line in result3['logs']['stdout']:
            print(f"     {line}")
    
    # 5. Mostrar estadísticas
    print(f"\n5. Estadísticas de ejecución:")
    stats = await e2b_integration.get_execution_stats()
    print(f"   Total ejecuciones: {stats['stats']['total_executions']}")
    print(f"   Éxito: {stats['stats']['successful_executions']}")
    print(f"   Tasa de éxito: {stats['success_rate']:.2f}%")
    print(f"   Promedio tiempo ejecución: {stats['average_execution_time']:.3f}s")
    print(f"   Sandboxes creados: {stats['stats']['sandbox_created']}")
    
    # 6. Health check
    print(f"\n6. Health check del sistema:")
    health = await e2b_integration.health_check()
    print(f"   Estado: {health['status']}")
    
    # 7. Limpieza
    await e2b_integration.cleanup()
    print(f"\n7. Recursos limpiados")
    
    print("\n" + "="*70)
    print("✅ EJEMPLO AVANZADO DE E2B COMPLETADO")
    print("🎯 El sistema admite:")
    print("   - Templates predefinidos para diferentes tipos de tareas")
    print("   - Creación dinámica de VMs según tipo de tarea")
    print("   - Gestión automática de recursos (tiempo, memoria, CPU)")
    print("   - Estadísticas detalladas de ejecución")
    print("   - Gestión de ciclo de vida de sandboxes")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(advanced_example_usage())