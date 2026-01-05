"""
Integración de e2b para generación rápida de VMs en el sistema Capibara6.
"""

import os
import asyncio
import json
from typing import Optional, Dict, Any
from e2b import Sandbox
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class E2BManager:
    """
    Gestor para la integración de e2b con el sistema Capibara6.
    Permite crear y gestionar sandboxes/VMs rápidas para tareas específicas.
    """
    
    def __init__(self, api_key: Optional[str] = None, config_path: str = "e2b_config.json"):
        """
        Inicializa el gestor e2b.
        
        :param api_key: Clave de API de e2b. Si no se proporciona, se leerá de la variable de entorno E2B_API_KEY.
        :param config_path: Ruta al archivo de configuración de e2b
        """
        self.api_key = api_key or os.getenv('E2B_API_KEY')
        if not self.api_key:
            raise ValueError("Se requiere una clave de API de e2b. Proporciona una clave o establece la variable de entorno E2B_API_KEY.")
        
        # Establecer la clave de API para el SDK de e2b
        os.environ['E2B_API_KEY'] = self.api_key
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        logger.info("E2BManager inicializado con éxito")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración de e2b desde un archivo JSON."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('e2b', {})
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración {config_path} no encontrado, usando valores por defecto")
            return {
                "timeout": 86400,
                "default_template": "python3",
                "limits": {
                    "max_session_length_seconds": 86400,
                    "max_concurrent_sandboxes": 100,
                    "vcpus_per_sandbox": 8,
                    "ram_per_sandbox_gb": 8,
                    "disk_per_sandbox_gb": 20
                }
            }
    
    async def create_sandbox(self, template: Optional[str] = None, timeout: Optional[int] = None) -> Sandbox:
        """
        Crea una nueva sandbox e2b.
        
        :param template: ID o alias de la plantilla a usar. Si no se especifica, usa la plantilla por defecto.
        :param timeout: Tiempo de espera máximo para la creación de la sandbox en segundos. Si no se especifica, usa el valor por defecto.
        :return: Instancia de la sandbox creada.
        """
        if timeout is None:
            timeout = self.config.get('timeout', 86400)  # 24 horas por defecto
        
        template = template or self.config.get('default_template', 'python3')
        
        try:
            logger.info(f"Creando sandbox con template: {template}, timeout: {timeout}s")
            
            # Establecer el tiempo de espera
            sandbox = await Sandbox.create(
                template=template,
                timeout=timeout
            )
            
            logger.info(f"Sandbox creada con ID: {sandbox.id}")
            return sandbox
        except Exception as e:
            logger.error(f"Error al crear la sandbox: {str(e)}")
            raise
    
    def create_sandbox_sync(self, template: Optional[str] = None, timeout: Optional[int] = None) -> Sandbox:
        """
        Crea una nueva sandbox e2b de forma síncrona.
        
        :param template: ID o alias de la plantilla a usar. Si no se especifica, usa la plantilla por defecto.
        :param timeout: Tiempo de espera máximo para la creación de la sandbox en segundos. Si no se especifica, usa el valor por defecto.
        :return: Instancia de la sandbox creada.
        """
        if timeout is None:
            timeout = self.config.get('timeout', 86400)  # 24 horas por defecto
            
        template = template or self.config.get('default_template', 'python3')
        
        try:
            logger.info(f"Creando sandbox (sync) con template: {template}, timeout: {timeout}s")
            
            # Establecer el tiempo de espera
            sandbox = Sandbox.create(
                template=template,
                timeout=timeout
            )
            
            logger.info(f"Sandbox (sync) creada con ID: {sandbox.id}")
            return sandbox
        except Exception as e:
            logger.error(f"Error al crear la sandbox (sync): {str(e)}")
            raise
    
    async def run_python_code(self, code: str, template: Optional[str] = None, timeout: Optional[int] = None, output_files: Optional[list] = None) -> Dict[str, Any]:
        """
        Ejecuta código Python en una sandbox e2b.
        
        :param code: Código Python a ejecutar.
        :param template: Plantilla de sandbox a usar. Si no se especifica, usa la plantilla por defecto.
        :param timeout: Tiempo de espera máximo en segundos. Si no se especifica, usa el valor por defecto.
        :param output_files: Lista de archivos de salida a leer después de la ejecución.
        :return: Resultado de la ejecución del código.
        """
        sandbox = None
        output_files_content = {}
        try:
            final_template = template or self.config.get('default_template', 'python3')
            final_timeout = timeout or self.config.get('timeout', 86400)
            
            logger.info(f"Ejecutando código Python en sandbox e2b con template: {final_template}, timeout: {final_timeout}s")
            sandbox = await self.create_sandbox(template=final_template, timeout=final_timeout)
            
            # Escribir el código en un archivo temporal
            await sandbox.filesystem.write('/code.py', code)
            
            # Ejecutar el código
            process = await sandbox.process.start_and_wait('python', '/code.py')
            
            # Si se especificaron archivos de salida, leerlos
            if output_files:
                for file_path in output_files:
                    try:
                        # Verificar si el archivo existe
                        stat = await sandbox.filesystem.stat(file_path)
                        if stat:
                            # Leer el archivo
                            content = await sandbox.filesystem.read(file_path)
                            # Codificar como base64 si es un archivo binario como imagen
                            if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                                import base64
                                content = base64.b64encode(content).decode('utf-8')
                                output_files_content[file_path] = f"data:image/png;base64,{content}"
                            else:
                                output_files_content[file_path] = content
                    except Exception as e:
                        logger.warning(f"Error leyendo archivo {file_path}: {str(e)}")
                        output_files_content[file_path] = f"Error al leer archivo: {str(e)}"
            
            result = {
                'success': True,
                'stdout': process.stdout,
                'stderr': process.stderr,
                'exit_code': process.exit_code,
                'sandbox_id': sandbox.id,
                'output_files': output_files_content
            }
            
            logger.info(f"Código Python ejecutado con éxito. Salida: {process.stdout[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error al ejecutar código Python: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'sandbox_id': sandbox.id if sandbox else None,
                'output_files': {}
            }
        finally:
            if sandbox:
                await sandbox.close()
    
    def run_python_code_sync(self, code: str, template: Optional[str] = None, timeout: Optional[int] = None, output_files: Optional[list] = None) -> Dict[str, Any]:
        """
        Ejecuta código Python en una sandbox e2b de forma síncrona.
        
        :param code: Código Python a ejecutar.
        :param template: Plantilla de sandbox a usar. Si no se especifica, usa la plantilla por defecto.
        :param timeout: Tiempo de espera máximo en segundos. Si no se especifica, usa el valor por defecto.
        :param output_files: Lista de archivos de salida a leer después de la ejecución.
        :return: Resultado de la ejecución del código.
        """
        sandbox = None
        output_files_content = {}
        try:
            final_template = template or self.config.get('default_template', 'python3')
            final_timeout = timeout or self.config.get('timeout', 86400)
            
            logger.info(f"Ejecutando código Python en sandbox e2b (sync) con template: {final_template}, timeout: {final_timeout}s")
            sandbox = self.create_sandbox_sync(template=final_template, timeout=final_timeout)
            
            # Escribir el código en un archivo temporal
            sandbox.filesystem.write('/code.py', code)
            
            # Ejecutar el código
            process = sandbox.process.start_and_wait('python', '/code.py')
            
            # Si se especificaron archivos de salida, leerlos
            if output_files:
                for file_path in output_files:
                    try:
                        # Verificar si el archivo existe
                        stat = sandbox.filesystem.stat(file_path)
                        if stat:
                            # Leer el archivo
                            content = sandbox.filesystem.read(file_path)
                            # Codificar como base64 si es un archivo binario como imagen
                            if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                                import base64
                                if isinstance(content, bytes):
                                    content = base64.b64encode(content).decode('utf-8')
                                    output_files_content[file_path] = f"data:image/png;base64,{content}"
                                else:
                                    output_files_content[file_path] = content
                            else:
                                output_files_content[file_path] = content
                    except Exception as e:
                        logger.warning(f"Error leyendo archivo {file_path}: {str(e)}")
                        output_files_content[file_path] = f"Error al leer archivo: {str(e)}"
            
            result = {
                'success': True,
                'stdout': process.stdout,
                'stderr': process.stderr,
                'exit_code': process.exit_code,
                'sandbox_id': sandbox.id,
                'output_files': output_files_content
            }
            
            logger.info(f"Código Python ejecutado con éxito (sync). Salida: {process.stdout[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error al ejecutar código Python (sync): {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'sandbox_id': sandbox.id if sandbox else None,
                'output_files': {}
            }
        finally:
            if sandbox:
                sandbox.close()
    
    async def run_command(self, command: str, template: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta un comando en una sandbox e2b.
        
        :param command: Comando a ejecutar.
        :param template: Plantilla de sandbox a usar.
        :param timeout: Tiempo de espera máximo en segundos. Si no se especifica, usa el valor por defecto.
        :return: Resultado de la ejecución del comando.
        """
        sandbox = None
        try:
            final_template = template or self.config.get('default_template', 'python3')
            final_timeout = timeout or self.config.get('timeout', 86400)
            
            logger.info(f"Ejecutando comando en sandbox e2b: {command}, template: {final_template}, timeout: {final_timeout}s")
            sandbox = await self.create_sandbox(template=final_template, timeout=final_timeout)
            
            process = await sandbox.process.start_and_wait(command)
            
            result = {
                'success': True,
                'stdout': process.stdout,
                'stderr': process.stderr,
                'exit_code': process.exit_code,
                'sandbox_id': sandbox.id
            }
            
            logger.info(f"Comando ejecutado con éxito. Salida: {process.stdout[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error al ejecutar comando: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'sandbox_id': sandbox.id if sandbox else None
            }
        finally:
            if sandbox:
                await sandbox.close()
    
    def run_command_sync(self, command: str, template: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta un comando en una sandbox e2b de forma síncrona.
        
        :param command: Comando a ejecutar.
        :param template: Plantilla de sandbox a usar.
        :param timeout: Tiempo de espera máximo en segundos. Si no se especifica, usa el valor por defecto.
        :return: Resultado de la ejecución del comando.
        """
        sandbox = None
        try:
            final_template = template or self.config.get('default_template', 'python3')
            final_timeout = timeout or self.config.get('timeout', 86400)
            
            logger.info(f"Ejecutando comando (sync) en sandbox e2b: {command}, template: {final_template}, timeout: {final_timeout}s")
            sandbox = self.create_sandbox_sync(template=final_template, timeout=final_timeout)
            
            process = sandbox.process.start_and_wait(command)
            
            result = {
                'success': True,
                'stdout': process.stdout,
                'stderr': process.stderr,
                'exit_code': process.exit_code,
                'sandbox_id': sandbox.id
            }
            
            logger.info(f"Comando (sync) ejecutado con éxito. Salida: {process.stdout[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error al ejecutar comando (sync): {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'sandbox_id': sandbox.id if sandbox else None
            }
        finally:
            if sandbox:
                sandbox.close()

# Función para crear el gestor con la clave API proporcionada
def create_e2b_manager(api_token: str) -> E2BManager:
    """
    Crea una instancia de E2BManager con la clave API especificada.
    
    :param api_token: Token de API de e2b
    :return: Instancia de E2BManager
    """
    return E2BManager(api_key=api_token)

# Ejemplo de uso
if __name__ == "__main__":
    # Usar la clave API proporcionada
    API_TOKEN = "e2b_d8df23b5de5214b7bfb4ebe227a308b61a2ae172"
    
    # Crear el gestor e2b
    e2b_manager = create_e2b_manager(API_TOKEN)
    
    # Ejemplo de uso
    async def main():
        # Ejecutar un simple script Python
        result = await e2b_manager.run_python_code("""
import platform
import json

result = {
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "message": "Sandbox e2b funcionando correctamente"
}

print(json.dumps(result))
""")
        
        print("Resultado de ejecución Python:")
        print(result)
    
    # Ejecutar el ejemplo
    asyncio.run(main())