#!/usr/bin/env python3
"""
 CAPIBARA DIGITAL MANUFACTURING ECOSYSTEM
==========================================

Ecosistema completo de fabricación digital que integra con Text-to-Gen.
El VERDADERO BOOM: De descripción → producto fabricado automáticamente.

Componentes:
- RAG actualizado diariamente con proveedores globales
- N8N workflows para automatización de fabricación
- APIs de proveedores por zona geográfica
- Optimización automática de costes y envíos
- Selección inteligente de servicios de fabricación

Pipeline completo:
Text-to-Gen → Manufacturing RAG → N8N Automation → Fabricación Global
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import requests
import sqlite3

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManufacturingType(Enum):
    """Tipos de fabricación digital"""
    PCB_ASSEMBLY = "pcb_assembly"           # Ensamblaje de PCBs
    PRINT_3D = "3d_printing"               # Impresión 3D
    CNC_MACHINING = "cnc_machining"        # Mecanizado CNC
    LASER_CUTTING = "laser_cutting"        # Corte láser
    INJECTION_MOLDING = "injection_molding" # Moldeo por inyección
    SHEET_METAL = "sheet_metal"            # Chapa metálica
    ELECTRONIC_ASSEMBLY = "electronic_assembly" # Ensamblaje electrónico
    CUSTOM_ASSEMBLY = "custom_assembly"     # Ensamblaje personalizado

class Region(Enum):
    """Regiones geográficas para optimización"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    SOUTH_AMERICA = "south_america"
    AFRICA = "africa"
    MIDDLE_EAST = "middle_east"

class PriorityLevel(Enum):
    """Niveles de prioridad para fabricación"""
    ECONOMY = "economy"         # Más barato, más lento
    STANDARD = "standard"       # Balance precio/tiempo
    EXPRESS = "express"         # Rápido, más caro
    PREMIUM = "premium"         # Máxima calidad y velocidad

@dataclass
class ManufacturingRequirement:
    """Requerimiento de fabricación"""
    manufacturing_type: ManufacturingType
    files: List[str] = field(default_factory=list)
    quantity: int = 1
    material: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: List[str] = field(default_factory=list)
    delivery_date: Optional[datetime] = None
    budget_limit: Optional[float] = None

@dataclass
class SupplierInfo:
    """Información de proveedor"""
    supplier_id: str
    name: str
    region: Region
    manufacturing_types: List[ManufacturingType]
    base_location: str
    
    # Métricas
    quality_score: float = 0.0      # 0-1
    delivery_score: float = 0.0     # 0-1
    cost_competitiveness: float = 0.0 # 0-1
    
    # Capacidades
    min_quantity: int = 1
    max_quantity: int = 10000
    lead_time_days: int = 7
    
    # APIs
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    supports_auto_quote: bool = False
    
    # Certificaciones
    certifications: List[str] = field(default_factory=list)
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Quote:
    """Cotización de fabricación"""
    quote_id: str
    supplier_id: str
    supplier_name: str
    
    # Costes
    unit_cost: float
    total_cost: float
    shipping_cost: float
    tax_cost: float
    final_cost: float
    
    # Tiempos
    lead_time_days: int
    shipping_days: int
    total_days: int
    estimated_delivery: datetime
    
    # Detalles
    currency: str = "USD"
    valid_until: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    
    # Scoring
    cost_score: float = 0.0         # 0-1 (1 = más barato)
    time_score: float = 0.0         # 0-1 (1 = más rápido)
    quality_score: float = 0.0      # 0-1 (1 = mejor calidad)
    overall_score: float = 0.0      # Score combinado
    
    quote_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ManufacturingOrder:
    """Orden de fabricación"""
    order_id: str
    project_name: str
    customer_region: Region
    priority: PriorityLevel
    
    requirements: List[ManufacturingRequirement]
    selected_quotes: List[Quote]
    
    total_cost: float = 0.0
    estimated_delivery: Optional[datetime] = None
    
    # Estado
    status: str = "pending"  # pending, quoted, ordered, manufacturing, shipped, delivered
    created_at: datetime = field(default_factory=datetime.now)
    
    # N8N workflow
    workflow_id: Optional[str] = None
    automation_enabled: bool = True

class ManufacturingRAG:
    """RAG para proveedores y fabricación digital"""
    
    def __init__(self, db_path: str = "./manufacturing_rag.db"):
        self.db_path = db_path
        self.last_update = None
        self._init_database()
        
        # URLs de APIs de proveedores (ejemplos reales)
        self.supplier_apis = {
            "jlcpcb": {
                "base_url": "https://cart.jlcpcb.com/api",
                "quote_endpoint": "/quote",
                "capabilities": ["pcb_assembly", "3d_printing"]
            },
            "pcbway": {
                "base_url": "https://www.pcbway.com/api",
                "quote_endpoint": "/quotation",
                "capabilities": ["pcb_assembly", "cnc_machining", "3d_printing"]
            },
            "shapeways": {
                "base_url": "https://api.shapeways.com/v1",
                "quote_endpoint": "/price",
                "capabilities": ["3d_printing", "injection_molding"]
            },
            "protolabs": {
                "base_url": "https://www.protolabs.com/api",
                "quote_endpoint": "/quote",
                "capabilities": ["cnc_machining", "injection_molding", "sheet_metal"]
            },
            "xometry": {
                "base_url": "https://www.xometry.com/api",
                "quote_endpoint": "/instant-quote",
                "capabilities": ["cnc_machining", "3d_printing", "sheet_metal"]
            }
        }
        
        logger.info(" Manufacturing RAG initialized")
    
    def _init_database(self):
        """Inicializa base de datos SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de proveedores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suppliers (
                supplier_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL,
                manufacturing_types TEXT NOT NULL,
                base_location TEXT,
                quality_score REAL,
                delivery_score REAL,
                cost_competitiveness REAL,
                api_endpoint TEXT,
                supports_auto_quote BOOLEAN,
                certifications TEXT,
                last_updated TIMESTAMP,
                raw_data TEXT
            )
        """)
        
        # Tabla de cotizaciones históricas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_quotes (
                quote_id TEXT PRIMARY KEY,
                supplier_id TEXT,
                manufacturing_type TEXT,
                quantity INTEGER,
                unit_cost REAL,
                lead_time_days INTEGER,
                created_at TIMESTAMP,
                project_complexity REAL,
                FOREIGN KEY (supplier_id) REFERENCES suppliers (supplier_id)
            )
        """)
        
        # Tabla de actualizaciones RAG
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_type TEXT,
                timestamp TIMESTAMP,
                records_updated INTEGER,
                status TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(" Manufacturing RAG database initialized")
    
    async def daily_update(self):
        """Actualización diaria automática del RAG"""
        logger.info(" Starting daily RAG update...")
        
        start_time = datetime.now()
        total_updated = 0
        
        try:
            # Actualizar proveedores por región
            for region in Region:
                updated = await self._update_suppliers_by_region(region)
                total_updated += updated
            
            # Actualizar precios y capacidades
            await self._update_pricing_data()
            
            # Actualizar métricas de calidad
            await self._update_quality_metrics()
            
            # Limpiar datos obsoletos
            await self._cleanup_old_data()
            
            # Registrar actualización
            self._log_update("daily_update", total_updated, "success")
            
            self.last_update = datetime.now()
            update_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f" Daily RAG update completed in {update_time:.1f}s")
            logger.info(f" Updated {total_updated} supplier records")
            
        except Exception as e:
            logger.error(f" Daily RAG update failed: {e}")
            self._log_update("daily_update", 0, f"failed: {e}")
    
    async def _update_suppliers_by_region(self, region: Region) -> int:
        """Actualiza proveedores por región"""
        # En implementación real, consultaría APIs y directorios
        # Para demo, simulamos datos
        
        suppliers_data = await self._fetch_regional_suppliers(region)
        updated_count = 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for supplier_data in suppliers_data:
            cursor.execute("""
                INSERT OR REPLACE INTO suppliers 
                (supplier_id, name, region, manufacturing_types, base_location,
                 quality_score, delivery_score, cost_competitiveness, 
                 api_endpoint, supports_auto_quote, certifications, 
                 last_updated, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                supplier_data["supplier_id"],
                supplier_data["name"],
                region.value,
                json.dumps([t.value for t in supplier_data["manufacturing_types"]]),
                supplier_data["base_location"],
                supplier_data["quality_score"],
                supplier_data["delivery_score"],
                supplier_data["cost_competitiveness"],
                supplier_data.get("api_endpoint"),
                supplier_data.get("supports_auto_quote", False),
                json.dumps(supplier_data.get("certifications", [])),
                datetime.now(),
                json.dumps(supplier_data)
            ))
            updated_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"    {region.value}: {updated_count} suppliers updated")
        return updated_count
    
    async def _fetch_regional_suppliers(self, region: Region) -> List[Dict[str, Any]]:
        """Obtiene proveedores por región (simulado)"""
        # En implementación real, consultaría APIs reales, directorios B2B, etc.
        
        regional_suppliers = {
            Region.ASIA_PACIFIC: [
                {
                    "supplier_id": "jlcpcb_asia",
                    "name": "JLCPCB Asia",
                    "manufacturing_types": [ManufacturingType.PCB_ASSEMBLY, ManufacturingType.PRINT_3D],
                    "base_location": "Shenzhen, China",
                    "quality_score": 0.85,
                    "delivery_score": 0.90,
                    "cost_competitiveness": 0.95,
                    "api_endpoint": "https://cart.jlcpcb.com/api",
                    "supports_auto_quote": True,
                    "certifications": ["ISO9001", "IPC-A-610"]
                },
                {
                    "supplier_id": "pcbway_asia",
                    "name": "PCBWay Manufacturing",
                    "manufacturing_types": [ManufacturingType.PCB_ASSEMBLY, ManufacturingType.CNC_MACHINING],
                    "base_location": "Hangzhou, China",
                    "quality_score": 0.88,
                    "delivery_score": 0.85,
                    "cost_competitiveness": 0.92,
                    "api_endpoint": "https://www.pcbway.com/api",
                    "supports_auto_quote": True,
                    "certifications": ["ISO9001", "UL", "RoHS"]
                }
            ],
            Region.NORTH_AMERICA: [
                {
                    "supplier_id": "protolabs_na",
                    "name": "Protolabs North America", 
                    "manufacturing_types": [ManufacturingType.CNC_MACHINING, ManufacturingType.INJECTION_MOLDING],
                    "base_location": "Minneapolis, USA",
                    "quality_score": 0.95,
                    "delivery_score": 0.92,
                    "cost_competitiveness": 0.75,
                    "api_endpoint": "https://www.protolabs.com/api",
                    "supports_auto_quote": True,
                    "certifications": ["ISO9001", "AS9100", "ISO13485"]
                },
                {
                    "supplier_id": "xometry_na",
                    "name": "Xometry Manufacturing",
                    "manufacturing_types": [ManufacturingType.CNC_MACHINING, ManufacturingType.PRINT_3D, ManufacturingType.SHEET_METAL],
                    "base_location": "Gaithersburg, USA",
                    "quality_score": 0.90,
                    "delivery_score": 0.88,
                    "cost_competitiveness": 0.80,
                    "api_endpoint": "https://www.xometry.com/api",
                    "supports_auto_quote": True,
                    "certifications": ["ISO9001", "ITAR", "ISO13485"]
                }
            ],
            Region.EUROPE: [
                {
                    "supplier_id": "sculpteo_eu",
                    "name": "Sculpteo Europe",
                    "manufacturing_types": [ManufacturingType.PRINT_3D, ManufacturingType.LASER_CUTTING],
                    "base_location": "Paris, France",
                    "quality_score": 0.87,
                    "delivery_score": 0.90,
                    "cost_competitiveness": 0.82,
                    "api_endpoint": "https://www.sculpteo.com/api",
                    "supports_auto_quote": True,
                    "certifications": ["ISO9001", "REACH"]
                }
            ]
        }
        
        return regional_suppliers.get(region, [])
    
    async def _update_pricing_data(self):
        """Actualiza datos de precios desde APIs"""
        logger.info(" Updating pricing data...")
        
        # En implementación real, consultaría APIs de proveedores
        # Para obtener precios actualizados
        await asyncio.sleep(1.0)  # Simular procesamiento
        
        logger.info("    Pricing data updated")
    
    async def _update_quality_metrics(self):
        """Actualiza métricas de calidad basadas en historial"""
        logger.info(" Updating quality metrics...")
        
        # En implementación real, analizaría feedback de clientes,
        # tiempos de entrega reales, calidad de productos, etc.
        await asyncio.sleep(0.5)  # Simular procesamiento
        
        logger.info("    Quality metrics updated")
    
    async def _cleanup_old_data(self):
        """Limpia datos obsoletos"""
        cutoff_date = datetime.now() - timedelta(days=90)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM historical_quotes 
            WHERE created_at < ?
        """, (cutoff_date,))
        
        deleted_quotes = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"    Cleaned {deleted_quotes} old quote records")
    
    def _log_update(self, update_type: str, records_updated: int, status: str):
        """Registra actualización en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rag_updates (update_type, timestamp, records_updated, status)
            VALUES (?, ?, ?, ?)
        """, (update_type, datetime.now(), records_updated, status))
        
        conn.commit()
        conn.close()
    
    def find_optimal_suppliers(self, requirements: List[ManufacturingRequirement], customer_region: Region, priority: PriorityLevel) -> List[SupplierInfo]:
        """Encuentra proveedores óptimos basado en requerimientos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        suitable_suppliers = []
        
        # Buscar proveedores por tipo de fabricación
        for req in requirements:
            cursor.execute("""
                SELECT * FROM suppliers 
                WHERE manufacturing_types LIKE ? 
                ORDER BY 
                    CASE WHEN region = ? THEN 0 ELSE 1 END,
                    (quality_score + delivery_score + cost_competitiveness) DESC
            """, (f'%{req.manufacturing_type.value}%', customer_region.value))
            
            results = cursor.fetchall()
            
            for row in results:
                supplier = SupplierInfo(
                    supplier_id=row[0],
                    name=row[1],
                    region=Region(row[2]),
                    manufacturing_types=[ManufacturingType(t) for t in json.loads(row[3])],
                    base_location=row[4],
                    quality_score=row[5],
                    delivery_score=row[6],
                    cost_competitiveness=row[7],
                    api_endpoint=row[8],
                    supports_auto_quote=bool(row[9]),
                    certifications=json.loads(row[10] or "[]")
                )
                
                if supplier not in suitable_suppliers:
                    suitable_suppliers.append(supplier)
        
        conn.close()
        
        # Optimizar por prioridad
        if priority == PriorityLevel.ECONOMY:
            suitable_suppliers.sort(key=lambda s: s.cost_competitiveness, reverse=True)
        elif priority == PriorityLevel.EXPRESS:
            suitable_suppliers.sort(key=lambda s: s.delivery_score, reverse=True)
        elif priority == PriorityLevel.PREMIUM:
            suitable_suppliers.sort(key=lambda s: s.quality_score, reverse=True)
        else:  # STANDARD
            suitable_suppliers.sort(key=lambda s: (s.quality_score + s.delivery_score + s.cost_competitiveness) / 3, reverse=True)
        
        return suitable_suppliers[:10]  # Top 10 proveedores

class N8nManufacturingAutomation:
    """Automatización de fabricación con N8N"""
    
    def __init__(self, n8n_base_url: str = "http://localhost:5678", api_key: Optional[str] = None):
        self.n8n_base_url = n8n_base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        # IDs de workflows predefinidos en N8N
        self.workflows = {
            "auto_quote": "auto_quote_workflow_id",
            "order_placement": "order_placement_workflow_id", 
            "status_tracking": "status_tracking_workflow_id",
            "quality_verification": "quality_verification_workflow_id",
            "shipping_coordination": "shipping_coordination_workflow_id"
        }
        
        logger.info(" N8N Manufacturing Automation initialized")
    
    async def trigger_auto_quote_workflow(self, order: ManufacturingOrder) -> Dict[str, Any]:
        """Trigger workflow automático de cotización"""
        logger.info(f" Triggering auto-quote workflow for order {order.order_id}")
        
        workflow_data = {
            "order_id": order.order_id,
            "project_name": order.project_name,
            "customer_region": order.customer_region.value,
            "priority": order.priority.value,
            "requirements": [
                {
                    "manufacturing_type": req.manufacturing_type.value,
                    "files": req.files,
                    "quantity": req.quantity,
                    "material": req.material,
                    "specifications": req.specifications
                }
                for req in order.requirements
            ]
        }
        
        try:
            # En implementación real, haría llamada a N8N API
            # response = requests.post(
            #     f"{self.n8n_base_url}/webhook/{self.workflows['auto_quote']}",
            #     json=workflow_data,
            #     headers=self.headers
            # )
            
            # Para demo, simulamos respuesta
            await asyncio.sleep(2.0)
            
            result = {
                "workflow_id": f"wf_{order.order_id}_quote",
                "status": "started",
                "estimated_quotes": len(order.requirements) * 3,  # 3 quotes per requirement
                "completion_time": datetime.now() + timedelta(minutes=10)
            }
            
            logger.info(f"    Auto-quote workflow started: {result['workflow_id']}")
            return result
            
        except Exception as e:
            logger.error(f"    Failed to trigger auto-quote workflow: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def trigger_order_placement_workflow(self, order: ManufacturingOrder, selected_quotes: List[Quote]) -> Dict[str, Any]:
        """Trigger workflow de colocación de órdenes"""
        logger.info(f" Triggering order placement workflow for {len(selected_quotes)} suppliers")
        
        workflow_data = {
            "order_id": order.order_id,
            "selected_quotes": [
                {
                    "quote_id": quote.quote_id,
                    "supplier_id": quote.supplier_id,
                    "total_cost": quote.final_cost,
                    "lead_time_days": quote.lead_time_days
                }
                for quote in selected_quotes
            ]
        }
        
        try:
            await asyncio.sleep(1.5)  # Simular procesamiento
            
            result = {
                "workflow_id": f"wf_{order.order_id}_placement",
                "status": "started",
                "orders_placed": len(selected_quotes),
                "total_value": sum(q.final_cost for q in selected_quotes)
            }
            
            logger.info(f"    Order placement workflow started for ${result['total_value']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"    Failed to trigger order placement workflow: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def setup_status_tracking(self, order: ManufacturingOrder) -> Dict[str, Any]:
        """Configura tracking automático de estado"""
        logger.info(f" Setting up status tracking for order {order.order_id}")
        
        try:
            await asyncio.sleep(0.5)  # Simular configuración
            
            result = {
                "tracking_workflow_id": f"wf_{order.order_id}_tracking",
                "webhook_urls": [
                    f"{self.n8n_base_url}/webhook/status-update/{order.order_id}",
                    f"{self.n8n_base_url}/webhook/shipping-update/{order.order_id}"
                ],
                "monitoring_enabled": True,
                "update_frequency": "daily"
            }
            
            logger.info("    Status tracking configured")
            return result
            
        except Exception as e:
            logger.error(f"    Failed to setup status tracking: {e}")
            return {"status": "failed", "error": str(e)}

class DigitalManufacturingEcosystem:
    """Ecosistema completo de fabricación digital"""
    
    def __init__(self):
        self.rag = ManufacturingRAG()
        self.n8n = N8nManufacturingAutomation()
        
        # Configuración de regiones y costes de envío
        self.shipping_costs = self._load_shipping_matrix()
        
        logger.info(" Digital Manufacturing Ecosystem initialized")
    
    def _load_shipping_matrix(self) -> Dict[str, Dict[str, float]]:
        """Carga matriz de costes de envío entre regiones"""
        return {
            "asia_pacific": {
                "asia_pacific": 15.0,
                "north_america": 45.0,
                "europe": 50.0,
                "south_america": 60.0,
                "africa": 55.0,
                "middle_east": 40.0
            },
            "north_america": {
                "north_america": 12.0,
                "europe": 35.0,
                "asia_pacific": 45.0,
                "south_america": 30.0,
                "africa": 50.0,
                "middle_east": 40.0
            },
            "europe": {
                "europe": 10.0,
                "north_america": 35.0,
                "asia_pacific": 50.0,
                "south_america": 40.0,
                "africa": 25.0,
                "middle_east": 30.0
            }
            # ... más regiones
        }
    
    async def process_manufacturing_request(self, project_result, customer_region: Region, priority: PriorityLevel = PriorityLevel.STANDARD) -> ManufacturingOrder:
        """Procesa request completo de fabricación desde resultado de Text-to-Gen"""
        logger.info(" Processing complete manufacturing request...")
        
        # Extraer requerimientos de fabricación desde resultado de Text-to-Gen
        requirements = self._extract_manufacturing_requirements(project_result)
        
        # Crear orden de fabricación
        order = ManufacturingOrder(
            order_id=f"mfg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            project_name=project_result.project_name,
            customer_region=customer_region,
            priority=priority,
            requirements=requirements
        )
        
        logger.info(f" Created manufacturing order: {order.order_id}")
        logger.info(f" Requirements: {len(requirements)} items")
        
        # Procesar automáticamente
        await self._process_order_automatically(order)
        
        return order
    
    def _extract_manufacturing_requirements(self, project_result) -> List[ManufacturingRequirement]:
        """Extrae requerimientos de fabricación desde resultado de Text-to-Gen"""
        requirements = []
        
        # Analizar resultados por servicio
        service_results = project_result.service_results
        
        # Text-to-Print3D → Impresión 3D
        if 'print3d' in service_results and service_results['print3d'].get('success'):
            print3d_data = service_results['print3d']
            requirements.append(ManufacturingRequirement(
                manufacturing_type=ManufacturingType.PRINT_3D,
                files=[f for f in print3d_data.get('files', []) if f.endswith('.stl')],
                quantity=1,
                material=print3d_data.get('metadata', {}).get('material', 'PLA'),
                specifications={
                    "technology": print3d_data.get('metadata', {}).get('technology', 'FDM'),
                    "layer_height": 0.2,
                    "infill": print3d_data.get('specifications', {}).get('infill', 20)
                }
            ))
        
        # Text-to-Circuit → Ensamblaje PCB
        if 'circuit' in service_results and service_results['circuit'].get('success'):
            circuit_data = service_results['circuit']
            requirements.append(ManufacturingRequirement(
                manufacturing_type=ManufacturingType.PCB_ASSEMBLY,
                files=[f for f in circuit_data.get('files', []) if f.endswith(('.kicad_pcb', '.zip'))],
                quantity=1,
                specifications={
                    "board_size": circuit_data.get('metadata', {}).get('board_size', '50x40mm'),
                    "layers": circuit_data.get('metadata', {}).get('layers', 2),
                    "voltage": circuit_data.get('metadata', {}).get('voltage', '3.3V'),
                    "components": circuit_data.get('components', [])
                },
                quality_requirements=["IPC-A-610", "RoHS compliant"]
            ))
        
        # Text-to-CAD → Mecanizado CNC (si hay partes metálicas)
        if 'cad' in service_results and service_results['cad'].get('success'):
            cad_data = service_results['cad']
            # Si hay partes que requieren mecanizado
            fasteners = cad_data.get('metadata', {}).get('fasteners', [])
            if any('metal' in str(f).lower() for f in fasteners):
                requirements.append(ManufacturingRequirement(
                    manufacturing_type=ManufacturingType.CNC_MACHINING,
                    files=[f for f in cad_data.get('files', []) if f.endswith('.step')],
                    quantity=1,
                    material='Aluminum 6061',
                    specifications={
                        "tolerance": "±0.1mm",
                        "surface_finish": "As machined"
                    }
                ))
        
        # Ensamblaje final personalizado
        if len(requirements) > 1:
            requirements.append(ManufacturingRequirement(
                manufacturing_type=ManufacturingType.CUSTOM_ASSEMBLY,
                files=[],  # Se basa en otros componentes
                quantity=1,
                specifications={
                    "assembly_instructions": "Included in project documentation",
                    "testing_required": True,
                    "packaging": "Anti-static bag + bubble wrap"
                }
            ))
        
        return requirements
    
    async def _process_order_automatically(self, order: ManufacturingOrder):
        """Procesa orden automáticamente usando RAG + N8N"""
        logger.info(f" Processing order {order.order_id} automatically...")
        
        try:
            # 1. Encontrar proveedores óptimos usando RAG
            logger.info(" Finding optimal suppliers using RAG...")
            suppliers = self.rag.find_optimal_suppliers(
                order.requirements, 
                order.customer_region, 
                order.priority
            )
            logger.info(f"    Found {len(suppliers)} suitable suppliers")
            
            # 2. Trigger workflow N8N para cotizaciones automáticas
            logger.info(" Triggering N8N auto-quote workflow...")
            quote_workflow = await self.n8n.trigger_auto_quote_workflow(order)
            order.workflow_id = quote_workflow.get('workflow_id')
            
            # 3. Simular obtención de cotizaciones
            quotes = await self._get_automated_quotes(order, suppliers)
            logger.info(f"    Received {len(quotes)} automated quotes")
            
            # 4. Seleccionar mejores cotizaciones automáticamente
            selected_quotes = self._select_optimal_quotes(quotes, order.priority)
            order.selected_quotes = selected_quotes
            order.total_cost = sum(q.final_cost for q in selected_quotes)
            order.estimated_delivery = max(q.estimated_delivery for q in selected_quotes) if selected_quotes else None
            
            logger.info(f"    Selected {len(selected_quotes)} optimal quotes")
            logger.info(f"    Total cost: ${order.total_cost:.2f}")
            
            # 5. Si está habilitada automatización completa, colocar órdenes
            if order.automation_enabled:
                logger.info(" Placing orders automatically...")
                placement_workflow = await self.n8n.trigger_order_placement_workflow(order, selected_quotes)
                
                # 6. Configurar tracking automático
                await self.n8n.setup_status_tracking(order)
                
                order.status = "ordered"
                logger.info("    Orders placed and tracking configured")
            else:
                order.status = "quoted"
                logger.info("   ️ Quotes ready for manual approval")
            
        except Exception as e:
            logger.error(f" Error processing order automatically: {e}")
            order.status = "error"
    
    async def _get_automated_quotes(self, order: ManufacturingOrder, suppliers: List[SupplierInfo]) -> List[Quote]:
        """Obtiene cotizaciones automáticas de proveedores"""
        quotes = []
        
        for req in order.requirements:
            # Filtrar proveedores que manejan este tipo de fabricación
            suitable_suppliers = [
                s for s in suppliers 
                if req.manufacturing_type in s.manufacturing_types
            ]
            
            for supplier in suitable_suppliers[:3]:  # Top 3 por requerimiento
                quote = await self._request_quote_from_supplier(supplier, req, order.customer_region)
                if quote:
                    quotes.append(quote)
        
        return quotes
    
    async def _request_quote_from_supplier(self, supplier: SupplierInfo, requirement: ManufacturingRequirement, customer_region: Region) -> Optional[Quote]:
        """Solicita cotización a proveedor específico"""
        try:
            # Simular llamada a API del proveedor
            await asyncio.sleep(0.5)  # Simular latencia de API
            
            # Calcular costes base (simulado)
            base_cost = self._calculate_base_cost(requirement)
            
            # Aplicar multiplicadores del proveedor
            unit_cost = base_cost * (1 + (1 - supplier.cost_competitiveness) * 0.5)
            total_cost = unit_cost * requirement.quantity
            
            # Calcular envío
            shipping_cost = self._calculate_shipping_cost(supplier.region, customer_region, total_cost)
            
            # Calcular impuestos (simplificado)
            tax_cost = total_cost * 0.1  # 10% aprox
            
            final_cost = total_cost + shipping_cost + tax_cost
            
            # Tiempo de entrega
            base_lead_time = 7  # días base
            lead_time_days = int(base_lead_time * (1 + (1 - supplier.delivery_score) * 0.5))
            shipping_days = 3 if supplier.region == customer_region else 7
            
            quote = Quote(
                quote_id=f"q_{supplier.supplier_id}_{requirement.manufacturing_type.value}_{datetime.now().strftime('%H%M%S')}",
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.name,
                unit_cost=unit_cost,
                total_cost=total_cost,
                shipping_cost=shipping_cost,
                tax_cost=tax_cost,
                final_cost=final_cost,
                lead_time_days=lead_time_days,
                shipping_days=shipping_days,
                total_days=lead_time_days + shipping_days,
                estimated_delivery=datetime.now() + timedelta(days=lead_time_days + shipping_days),
                cost_score=1 - (final_cost / (final_cost * 1.5)),  # Score relativo
                time_score=1 - (lead_time_days / 14),  # Score relativo
                quality_score=supplier.quality_score
            )
            
            # Score combinado
            quote.overall_score = (quote.cost_score + quote.time_score + quote.quality_score) / 3
            
            return quote
            
        except Exception as e:
            logger.error(f"    Failed to get quote from {supplier.name}: {e}")
            return None
    
    def _calculate_base_cost(self, requirement: ManufacturingRequirement) -> float:
        """Calcula coste base por tipo de fabricación"""
        base_costs = {
            ManufacturingType.PRINT_3D: 25.0,
            ManufacturingType.PCB_ASSEMBLY: 45.0,
            ManufacturingType.CNC_MACHINING: 85.0,
            ManufacturingType.LASER_CUTTING: 35.0,
            ManufacturingType.ELECTRONIC_ASSEMBLY: 65.0,
            ManufacturingType.CUSTOM_ASSEMBLY: 40.0
        }
        
        return base_costs.get(requirement.manufacturing_type, 50.0)
    
    def _calculate_shipping_cost(self, supplier_region: Region, customer_region: Region, order_value: float) -> float:
        """Calcula coste de envío entre regiones"""
        base_shipping = self.shipping_costs.get(
            supplier_region.value, {}
        ).get(customer_region.value, 30.0)
        
        # Ajustar por valor del pedido
        if order_value > 100:
            base_shipping *= 0.8  # Descuento por volumen
        elif order_value > 500:
            base_shipping *= 0.6
        
        return base_shipping
    
    def _select_optimal_quotes(self, quotes: List[Quote], priority: PriorityLevel) -> List[Quote]:
        """Selecciona cotizaciones óptimas basado en prioridad"""
        if not quotes:
            return []
        
        # Agrupar por tipo de fabricación
        quotes_by_type = {}
        for quote in quotes:
            # Extraer tipo del quote_id (simplificado)
            quote_type = quote.quote_id.split('_')[2] if '_' in quote.quote_id else 'unknown'
            if quote_type not in quotes_by_type:
                quotes_by_type[quote_type] = []
            quotes_by_type[quote_type].append(quote)
        
        selected = []
        
        # Seleccionar mejor cotización por tipo según prioridad
        for quote_type, type_quotes in quotes_by_type.items():
            if priority == PriorityLevel.ECONOMY:
                best_quote = min(type_quotes, key=lambda q: q.final_cost)
            elif priority == PriorityLevel.EXPRESS:
                best_quote = min(type_quotes, key=lambda q: q.total_days)
            elif priority == PriorityLevel.PREMIUM:
                best_quote = max(type_quotes, key=lambda q: q.quality_score)
            else:  # STANDARD
                best_quote = max(type_quotes, key=lambda q: q.overall_score)
            
            selected.append(best_quote)
        
        return selected

# Factory functions
def create_manufacturing_ecosystem():
    """Crea instancia del ecosistema de fabricación digital"""
    return DigitalManufacturingEcosystem()

async def setup_daily_rag_updates(ecosystem: DigitalManufacturingEcosystem):
    """Configura actualizaciones diarias automáticas del RAG"""
    logger.info(" Setting up daily RAG updates...")
    
    # En implementación real, usarías un scheduler como APScheduler
    # Para demo, ejecutamos una vez
    await ecosystem.rag.daily_update()
    
    logger.info(" Daily RAG updates configured")