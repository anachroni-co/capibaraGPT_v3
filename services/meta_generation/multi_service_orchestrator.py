#!/usr/bin/env python3
"""
 CAPIBARA TEXT-TO-GEN MULTI-SERVICE ORCHESTRATOR
=================================================

Orquestador avanzado que combina múltiples servicios text-to-* para crear
proyectos completos desde una sola descripción.

Ejemplo: "Carcasa 3D con PCB, LEDs y firmware"
→ Text-to-Print3D + Text-to-Circuit + Text-to-Code + Text-to-CAD

Capacidades:
- Análisis de dependencias entre servicios
- Ejecución paralela y secuencial inteligente
- Integración automática de resultados
- Validación de compatibilidad
- Generación de documentación del proyecto
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectType(Enum):
    """Tipos de proyectos multi-servicio"""
    IOT_DEVICE = "iot_device"           # Dispositivo IoT completo
    ELECTRONIC_GADGET = "electronic_gadget"  # Gadget electrónico
    SMART_HOME = "smart_home"           # Dispositivo domótica
    WEARABLE = "wearable"               # Dispositivo wearable
    ROBOTICS = "robotics"               # Proyecto robótico
    AUTOMATION = "automation"           # Sistema de automatización
    PROTOTYPE = "prototype"             # Prototipo general

class ServiceDependency(Enum):
    """Tipos de dependencias entre servicios"""
    PARALLEL = "parallel"               # Servicios independientes
    SEQUENTIAL = "sequential"           # Uno depende del otro
    INTEGRATED = "integrated"           # Resultados se integran
    CONDITIONAL = "conditional"         # Depende de condiciones

@dataclass
class ServiceTask:
    """Tarea individual de un servicio"""
    service_name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1  # 1=high, 2=medium, 3=low
    timeout_seconds: int = 300
    retry_count: int = 2

@dataclass
class MultiServiceRequest:
    """Request para proyecto multi-servicio"""
    project_description: str
    project_name: str
    project_type: ProjectType
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    user_id: Optional[str] = None

@dataclass
class MultiServiceResult:
    """Resultado de proyecto multi-servicio"""
    success: bool
    project_name: str
    project_path: str = ""
    
    # Resultados por servicio
    service_results: Dict[str, Any] = field(default_factory=dict)
    integration_result: Optional[Dict[str, Any]] = None
    
    # Métricas
    total_execution_time: float = 0.0
    services_used: List[str] = field(default_factory=list)
    files_generated: List[str] = field(default_factory=list)
    
    # Documentación del proyecto
    project_documentation: str = ""
    assembly_instructions: str = ""
    bom_list: List[Dict[str, Any]] = field(default_factory=list)  # Bill of Materials
    
    error: Optional[str] = None

class ProjectAnalyzer:
    """Analizador de proyectos para determinar servicios necesarios"""
    
    def __init__(self):
        # Patrones de detección por proyecto
        self.project_patterns = {
            "carcasa": ["print3d", "cad"],
            "pcb": ["circuit"],
            "placa": ["circuit"],
            "electronica": ["circuit"],
            "circuito": ["circuit"],
            "led": ["circuit"],
            "iluminacion": ["circuit"],
            "firmware": ["code"],
            "software": ["code"],
            "programacion": ["code"],
            "bateria": ["print3d", "circuit"],
            "alimentacion": ["circuit"],
            "sensor": ["circuit", "code"],
            "control": ["code", "api"],
            "interfaz": ["api"],
            "web": ["api"],
            "app": ["code"],
            "motor": ["circuit", "code"],
            "servo": ["circuit", "code"],
            "display": ["circuit", "code"],
            "pantalla": ["circuit", "code"],
            "bluetooth": ["circuit", "code"],
            "wifi": ["circuit", "code"],
            "iot": ["circuit", "code", "api"]
        }
        
        # Dependencias típicas entre servicios
        self.service_dependencies = {
            "print3d": [],  # Generalmente independiente
            "circuit": [],  # Puede ser independiente
            "code": ["circuit"],  # Depende del hardware
            "cad": ["print3d", "circuit"],  # Integra componentes físicos
            "api": ["code"]  # Depende del software
        }
    
    def analyze_project(self, description: str) -> Dict[str, Any]:
        """Analiza descripción y determina servicios necesarios"""
        description_lower = description.lower()
        
        # Detectar servicios necesarios
        required_services = self._detect_required_services(description_lower)
        
        # Determinar tipo de proyecto
        project_type = self._detect_project_type(description_lower)
        
        # Crear plan de ejecución
        execution_plan = self._create_execution_plan(required_services)
        
        # Extraer requerimientos específicos
        requirements = self._extract_requirements(description_lower)
        
        return {
            "required_services": required_services,
            "project_type": project_type,
            "execution_plan": execution_plan,
            "requirements": requirements,
            "estimated_time": self._estimate_total_time(execution_plan),
            "complexity_score": self._calculate_complexity(required_services)
        }
    
    def _detect_required_services(self, description: str) -> List[str]:
        """Detecta servicios necesarios desde la descripción"""
        detected_services = set()
        
        for keyword, services in self.project_patterns.items():
            if keyword in description:
                detected_services.update(services)
        
        # Servicios mínimos por defecto
        if not detected_services:
            detected_services.add("print3d")  # Siempre generar algo físico
        
        return list(detected_services)
    
    def _detect_project_type(self, description: str) -> ProjectType:
        """Detecta el tipo de proyecto"""
        if any(word in description for word in ["iot", "internet", "conectado", "smart"]):
            return ProjectType.IOT_DEVICE
        elif any(word in description for word in ["robot", "robotica", "autonomo"]):
            return ProjectType.ROBOTICS
        elif any(word in description for word in ["casa", "hogar", "domotica"]):
            return ProjectType.SMART_HOME
        elif any(word in description for word in ["pulsera", "reloj", "wearable"]):
            return ProjectType.WEARABLE
        elif any(word in description for word in ["automatizacion", "control"]):
            return ProjectType.AUTOMATION
        elif any(word in description for word in ["gadget", "dispositivo"]):
            return ProjectType.ELECTRONIC_GADGET
        else:
            return ProjectType.PROTOTYPE
    
    def _create_execution_plan(self, services: List[str]) -> List[Dict[str, Any]]:
        """Crea plan de ejecución con dependencias"""
        plan = []
        
        # Ordenar servicios por dependencias
        ordered_services = self._order_by_dependencies(services)
        
        for i, service in enumerate(ordered_services):
            step = {
                "step": i + 1,
                "service": service,
                "dependencies": self.service_dependencies.get(service, []),
                "can_run_parallel": len(self.service_dependencies.get(service, [])) == 0
            }
            plan.append(step)
        
        return plan
    
    def _order_by_dependencies(self, services: List[str]) -> List[str]:
        """Ordena servicios respetando dependencias"""
        ordered = []
        remaining = services.copy()
        
        while remaining:
            # Buscar servicios sin dependencias pendientes
            for service in remaining[:]:
                deps = self.service_dependencies.get(service, [])
                if all(dep in ordered or dep not in services for dep in deps):
                    ordered.append(service)
                    remaining.remove(service)
                    break
            else:
                # Si no hay servicios sin dependencias, agregar el primero
                ordered.append(remaining.pop(0))
        
        return ordered
    
    def _extract_requirements(self, description: str) -> List[str]:
        """Extrae requerimientos específicos"""
        requirements = []
        
        # Detectar especificaciones técnicas
        if "3.3v" in description or "5v" in description:
            requirements.append("voltage_specified")
        if "arduino" in description:
            requirements.append("arduino_compatible")
        if "esp32" in description or "esp8266" in description:
            requirements.append("esp_platform")
        if "battery" in description or "bateria" in description:
            requirements.append("battery_powered")
        if "usb" in description:
            requirements.append("usb_interface")
        if "wireless" in description or "wifi" in description:
            requirements.append("wireless_connectivity")
        
        return requirements
    
    def _estimate_total_time(self, execution_plan: List[Dict[str, Any]]) -> int:
        """Estima tiempo total de ejecución"""
        service_times = {
            "print3d": 300,  # 5 min
            "circuit": 600,  # 10 min
            "code": 240,     # 4 min
            "cad": 480,      # 8 min
            "api": 180       # 3 min
        }
        
        total_time = 0
        for step in execution_plan:
            service = step["service"]
            total_time += service_times.get(service, 300)
        
        # Reducir tiempo si hay paralelización
        parallel_steps = sum(1 for step in execution_plan if step["can_run_parallel"])
        if parallel_steps > 1:
            total_time *= 0.7  # 30% reducción por paralelización
        
        return int(total_time)
    
    def _calculate_complexity(self, services: List[str]) -> float:
        """Calcula score de complejidad del proyecto"""
        base_complexity = len(services) * 0.2
        
        # Complejidad por tipo de servicio
        service_complexity = {
            "print3d": 0.1,
            "circuit": 0.3,
            "code": 0.25,
            "cad": 0.2,
            "api": 0.15
        }
        
        total_complexity = sum(service_complexity.get(s, 0.2) for s in services)
        
        return min(1.0, base_complexity + total_complexity)

class MultiServiceOrchestrator:
    """Orquestador principal de servicios múltiples"""
    
    def __init__(self):
        self.analyzer = ProjectAnalyzer()
        self.project_dir = Path("./generated_projects")
        self.project_dir.mkdir(exist_ok=True)
        
        # Servicios disponibles (mocks para demostración)
        self.available_services = {
            "print3d": self._mock_print3d_service,
            "circuit": self._mock_circuit_service,
            "code": self._mock_code_service,
            "cad": self._mock_cad_service,
            "api": self._mock_api_service
        }
        
        logger.info(" MultiServiceOrchestrator initialized")
    
    async def execute_multi_service_project(self, request: MultiServiceRequest) -> MultiServiceResult:
        """Ejecuta proyecto multi-servicio completo"""
        start_time = datetime.now()
        
        try:
            logger.info(f" Starting multi-service project: {request.project_name}")
            logger.info(f" Description: {request.project_description[:200]}...")
            
            # Análisis del proyecto
            analysis = self.analyzer.analyze_project(request.project_description)
            logger.info(f" Services required: {analysis['required_services']}")
            logger.info(f" Project type: {analysis['project_type'].value}")
            logger.info(f"️ Estimated time: {analysis['estimated_time']}s")
            
            # Crear directorio del proyecto
            project_path = self.project_dir / request.project_name
            project_path.mkdir(exist_ok=True)
            
            # Ejecutar servicios según plan
            service_results = await self._execute_services(
                analysis['execution_plan'], 
                request, 
                project_path
            )
            
            # Integrar resultados
            integration_result = await self._integrate_results(
                service_results, 
                project_path,
                analysis
            )
            
            # Generar documentación
            documentation = self._generate_project_documentation(
                request, analysis, service_results, integration_result
            )
            
            # Crear BOM (Bill of Materials)
            bom = self._generate_bom(service_results, analysis)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = MultiServiceResult(
                success=True,
                project_name=request.project_name,
                project_path=str(project_path),
                service_results=service_results,
                integration_result=integration_result,
                total_execution_time=execution_time,
                services_used=analysis['required_services'],
                files_generated=self._collect_generated_files(project_path),
                project_documentation=documentation,
                assembly_instructions=self._generate_assembly_instructions(service_results),
                bom_list=bom
            )
            
            # Guardar resultado del proyecto
            await self._save_project_result(result, project_path)
            
            logger.info(f" Multi-service project completed: {request.project_name}")
            logger.info(f" Services executed: {len(service_results)}")
            logger.info(f"️ Total time: {execution_time:.1f}s")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f" Multi-service project failed: {e}")
            
            return MultiServiceResult(
                success=False,
                project_name=request.project_name,
                total_execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_services(self, execution_plan: List[Dict[str, Any]], request: MultiServiceRequest, project_path: Path) -> Dict[str, Any]:
        """Ejecuta servicios según el plan de ejecución"""
        service_results = {}
        
        # Agrupar servicios por etapas (paralelos vs secuenciales)
        parallel_services = [step for step in execution_plan if step["can_run_parallel"]]
        sequential_services = [step for step in execution_plan if not step["can_run_parallel"]]
        
        # Ejecutar servicios paralelos primero
        if parallel_services:
            logger.info(f" Executing {len(parallel_services)} parallel services...")
            parallel_tasks = []
            
            for step in parallel_services:
                service_name = step["service"]
                if service_name in self.available_services:
                    task = self._execute_single_service(
                        service_name, request, project_path, step
                    )
                    parallel_tasks.append((service_name, task))
            
            # Esperar a que terminen todos los paralelos
            for service_name, task in parallel_tasks:
                try:
                    result = await task
                    service_results[service_name] = result
                    logger.info(f"    {service_name} completed")
                except Exception as e:
                    logger.error(f"    {service_name} failed: {e}")
                    service_results[service_name] = {"success": False, "error": str(e)}
        
        # Ejecutar servicios secuenciales
        if sequential_services:
            logger.info(f" Executing {len(sequential_services)} sequential services...")
            
            for step in sequential_services:
                service_name = step["service"]
                if service_name in self.available_services:
                    try:
                        result = await self._execute_single_service(
                            service_name, request, project_path, step
                        )
                        service_results[service_name] = result
                        logger.info(f"    {service_name} completed")
                    except Exception as e:
                        logger.error(f"    {service_name} failed: {e}")
                        service_results[service_name] = {"success": False, "error": str(e)}
        
        return service_results
    
    async def _execute_single_service(self, service_name: str, request: MultiServiceRequest, project_path: Path, step: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un servicio individual"""
        service_func = self.available_services[service_name]
        
        # Preparar parámetros específicos del servicio
        service_params = {
            "description": request.project_description,
            "project_name": request.project_name,
            "project_path": project_path,
            "step_info": step,
            "requirements": getattr(request, 'requirements', [])
        }
        
        return await service_func(service_params)
    
    # Mock services para demostración
    async def _mock_print3d_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock del servicio Text-to-Print3D"""
        await asyncio.sleep(2.0)  # Simular procesamiento
        
        return {
            "success": True,
            "service": "print3d",
            "files": [
                f"{params['project_path']}/enclosure.stl",
                f"{params['project_path']}/battery_holder.stl"
            ],
            "metadata": {
                "technology": "FDM",
                "material": "PLA",
                "print_time": "3h 45m",
                "volume": "45.2 cm³",
                "cost": "$2.85"
            },
            "specifications": {
                "dimensions": {"x": 80, "y": 60, "z": 25},
                "wall_thickness": 2.0,
                "infill": 20,
                "supports": True
            }
        }
    
    async def _mock_circuit_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock del servicio Text-to-Circuit"""
        await asyncio.sleep(3.0)  # Simular procesamiento
        
        return {
            "success": True,
            "service": "circuit",
            "files": [
                f"{params['project_path']}/pcb_schematic.kicad_sch",
                f"{params['project_path']}/pcb_layout.kicad_pcb",
                f"{params['project_path']}/gerber_files.zip"
            ],
            "metadata": {
                "board_size": "50x40mm",
                "layers": 2,
                "components_count": 15,
                "voltage": "3.3V",
                "current": "150mA"
            },
            "components": [
                {"name": "ESP32-WROOM", "qty": 1, "cost": 4.50},
                {"name": "LED RGB WS2812", "qty": 8, "cost": 0.35},
                {"name": "Battery Connector", "qty": 1, "cost": 0.25},
                {"name": "Push Button", "qty": 2, "cost": 0.15},
                {"name": "Resistors 330Ω", "qty": 3, "cost": 0.05}
            ]
        }
    
    async def _mock_code_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock del servicio Text-to-Code (firmware)"""
        await asyncio.sleep(1.5)  # Simular procesamiento
        
        return {
            "success": True,
            "service": "code",
            "files": [
                f"{params['project_path']}/firmware.ino",
                f"{params['project_path']}/config.h",
                f"{params['project_path']}/led_controller.cpp",
                f"{params['project_path']}/README_code.md"
            ],
            "metadata": {
                "language": "C++",
                "platform": "Arduino/ESP32",
                "libraries": ["FastLED", "WiFi", "ArduinoJson"],
                "features": ["LED control", "WiFi connectivity", "Web server"]
            },
            "code_stats": {
                "lines_of_code": 450,
                "functions": 12,
                "files": 4,
                "flash_usage": "65%",
                "ram_usage": "40%"
            }
        }
    
    async def _mock_cad_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock del servicio Text-to-CAD (ensamblaje)"""
        await asyncio.sleep(2.5)  # Simular procesamiento
        
        return {
            "success": True,
            "service": "cad",
            "files": [
                f"{params['project_path']}/assembly.step",
                f"{params['project_path']}/exploded_view.png",
                f"{params['project_path']}/technical_drawing.pdf"
            ],
            "metadata": {
                "assembly_parts": 5,
                "fasteners": ["M3x8 screws", "M3 nuts"],
                "total_volume": "120 cm³",
                "weight": "85g"
            },
            "assembly_info": {
                "complexity": "medium",
                "assembly_time": "15 minutes",
                "tools_required": ["Screwdriver", "Allen key 2.5mm"]
            }
        }
    
    async def _mock_api_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock del servicio Text-to-API (control interface)"""
        await asyncio.sleep(1.0)  # Simular procesamiento
        
        return {
            "success": True,
            "service": "api",
            "files": [
                f"{params['project_path']}/control_api.py",
                f"{params['project_path']}/web_interface.html",
                f"{params['project_path']}/api_documentation.md"
            ],
            "metadata": {
                "endpoints": 6,
                "authentication": "API key",
                "protocol": "REST",
                "format": "JSON"
            },
            "api_endpoints": [
                "GET /status",
                "POST /led/color",
                "POST /led/brightness",
                "GET /battery",
                "POST /wifi/config",
                "GET /device/info"
            ]
        }
    
    async def _integrate_results(self, service_results: Dict[str, Any], project_path: Path, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Integra resultados de múltiples servicios"""
        await asyncio.sleep(1.0)  # Simular integración
        
        integration = {
            "integration_success": True,
            "compatibility_checks": {
                "pcb_fits_enclosure": True,
                "battery_placement": True,
                "led_visibility": True,
                "connector_accessibility": True
            },
            "generated_files": {
                "project_summary.json": f"{project_path}/project_summary.json",
                "integration_report.md": f"{project_path}/integration_report.md",
                "manufacturing_guide.pdf": f"{project_path}/manufacturing_guide.pdf"
            },
            "optimization_suggestions": [
                "Consider adding heat dissipation vents",
                "USB connector could be more accessible",
                "LED diffuser would improve light distribution"
            ]
        }
        
        return integration
    
    def _generate_project_documentation(self, request: MultiServiceRequest, analysis: Dict[str, Any], service_results: Dict[str, Any], integration: Dict[str, Any]) -> str:
        """Genera documentación completa del proyecto"""
        
        doc = f"""# {request.project_name}

## Descripción del Proyecto
{request.project_description}

## Especificaciones Técnicas

### Servicios Utilizados
{', '.join(analysis['required_services'])}

### Tipo de Proyecto
{analysis['project_type'].value}

### Complejidad
{analysis['complexity_score']:.2f}/1.0

## Componentes Generados

"""
        
        for service_name, result in service_results.items():
            if result.get("success"):
                doc += f"### {service_name.upper()}\n"
                doc += f"- Archivos: {len(result.get('files', []))}\n"
                
                if 'metadata' in result:
                    for key, value in result['metadata'].items():
                        doc += f"- {key}: {value}\n"
                doc += "\n"
        
        doc += f"""## Integración

### Compatibilidad
- Verificaciones realizadas: {len(integration.get('compatibility_checks', {}))}
- Estado: {' Exitosa' if integration.get('integration_success') else ' Con problemas'}

### Sugerencias de Optimización
"""
        
        for suggestion in integration.get('optimization_suggestions', []):
            doc += f"- {suggestion}\n"
        
        return doc
    
    def _generate_bom(self, service_results: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera Bill of Materials (lista de materiales)"""
        bom = []
        
        # Componentes del circuito
        if 'circuit' in service_results and service_results['circuit'].get('success'):
            circuit_components = service_results['circuit'].get('components', [])
            for comp in circuit_components:
                bom.append({
                    "category": "Electronics",
                    "item": comp["name"],
                    "quantity": comp["qty"],
                    "unit_cost": comp["cost"],
                    "total_cost": comp["qty"] * comp["cost"],
                    "supplier": "Electronic supplier"
                })
        
        # Materiales de impresión 3D
        if 'print3d' in service_results and service_results['print3d'].get('success'):
            bom.append({
                "category": "3D Printing",
                "item": "PLA Filament",
                "quantity": "50g",
                "unit_cost": 0.03,
                "total_cost": 1.50,
                "supplier": "3D printing supplier"
            })
        
        # Tornillería (del CAD)
        if 'cad' in service_results and service_results['cad'].get('success'):
            bom.extend([
                {
                    "category": "Hardware",
                    "item": "M3x8 screws",
                    "quantity": 4,
                    "unit_cost": 0.05,
                    "total_cost": 0.20,
                    "supplier": "Hardware store"
                },
                {
                    "category": "Hardware", 
                    "item": "M3 nuts",
                    "quantity": 4,
                    "unit_cost": 0.03,
                    "total_cost": 0.12,
                    "supplier": "Hardware store"
                }
            ])
        
        return bom
    
    def _generate_assembly_instructions(self, service_results: Dict[str, Any]) -> str:
        """Genera instrucciones de ensamblaje"""
        
        instructions = """# Instrucciones de Ensamblaje

## Materiales Necesarios
- Componentes electrónicos (ver BOM)
- Carcasa impresa en 3D
- Tornillería
- Herramientas: destornillador, llave Allen 2.5mm

## Pasos de Ensamblaje

### 1. Preparación de Componentes
- Verificar que todos los componentes estén disponibles
- Limpiar las piezas impresas en 3D de soportes

### 2. Soldadura del PCB
- Soldar componentes según esquemático
- Verificar continuidad con multímetro
- Probar alimentación antes de continuar

### 3. Programación del Firmware
- Conectar programador al microcontrolador
- Cargar firmware generado
- Verificar funcionamiento básico

### 4. Ensamblaje Mecánico
- Colocar PCB en la carcasa inferior
- Conectar batería
- Cerrar carcasa con tornillos M3x8

### 5. Pruebas Finales
- Verificar encendido
- Probar todas las funciones
- Comprobar conectividad WiFi (si aplica)

## Solución de Problemas
- Si no enciende: verificar conexión de batería
- Si LEDs no funcionan: revisar soldaduras
- Si no conecta WiFi: verificar configuración de red

"""
        return instructions
    
    def _collect_generated_files(self, project_path: Path) -> List[str]:
        """Recopila lista de archivos generados"""
        # En implementación real, escanearia el directorio
        # Para demo, simulamos archivos típicos
        files = [
            "enclosure.stl",
            "battery_holder.stl", 
            "pcb_schematic.kicad_sch",
            "pcb_layout.kicad_pcb",
            "firmware.ino",
            "config.h",
            "assembly.step",
            "project_summary.json",
            "manufacturing_guide.pdf",
            "README.md"
        ]
        
        return [str(project_path / f) for f in files]
    
    async def _save_project_result(self, result: MultiServiceResult, project_path: Path):
        """Guarda resultado del proyecto"""
        # Guardar resumen del proyecto
        summary = {
            "project_name": result.project_name,
            "success": result.success,
            "execution_time": result.total_execution_time,
            "services_used": result.services_used,
            "files_generated": len(result.files_generated),
            "total_cost": sum(item["total_cost"] for item in result.bom_list),
            "generated_at": datetime.now().isoformat()
        }
        
        summary_file = project_path / "project_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Guardar documentación
        doc_file = project_path / "README.md"
        with open(doc_file, 'w') as f:
            f.write(result.project_documentation)
        
        # Guardar instrucciones de ensamblaje
        assembly_file = project_path / "ASSEMBLY.md"
        with open(assembly_file, 'w') as f:
            f.write(result.assembly_instructions)

# Factory function
def create_multi_service_orchestrator():
    """Crea instancia del orquestador multi-servicio"""
    return MultiServiceOrchestrator()