"""
Resource Manager - Resource allocation and tracking for models.

This module provides resource management functionality for the Model
Control Plane (MCP), handling allocation, release, and tracking of
computational resources (CPU, memory, GPU) across model instances.

Key Components:
    - ResourceAllocation: Data class for resource allocation records
    - ResourceManager: Main class for resource management

Author: Skydesk International Dev Team.
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

@dataclass
class ResourceAllocation:
    model_id: str
    version: str
    resources: Dict[str, Any]
    allocated_at: datetime
    last_used: datetime

class ResourceManager:
    def __init__(self):
        self.allocations: Dict[str, ResourceAllocation] = {}  # model_id -> allocation
        self.total_resources = {
            "cpu": 100,  # porcentaje
            "memory": 16384,  # MB
            "gpu": 1  # number of GPUs
        }
        self.used_resources = {
            "cpu": 0,
            "memory": 0,
            "gpu": 0
        }

    async def allocate_resources(self, model_id: str, version: str, resources: Dict[str, Any]) -> ResourceAllocation:
        """Asigna recursos a un model específico."""
        if model_id in self.allocations:
            raise ValueError(f"El modelo {model_id} ya tiene recursos asignados")

        # verify disponibilidad de recursos
        for resource, amount in resources.items():
            if self.used_resources[resource] + amount > self.total_resources[resource]:
                raise ValueError(f"No hay suficientes recursos de {resource} disponibles")

        # create nueva asignación
        allocation = ResourceAllocation(
            model_id=model_id,
            version=version,
            resources=resources,
            allocated_at=datetime.now(),
            last_used=datetime.now()
        )

        # update recursos usados
        for resource, amount in resources.items():
            self.used_resources[resource] += amount

        self.allocations[model_id] = allocation
        logger.info(f"Recursos asignados para el modelo {model_id}")
        return allocation

    async def release_resources(self, model_id: str) -> None:
        """Libera los recursos asignados a un model."""
        if model_id not in self.allocations:
            raise ValueError(f"El modelo {model_id} no tiene recursos asignados")

        # free recursos
        for resource, amount in self.allocations[model_id].resources.items():
            self.used_resources[resource] -= amount

        del self.allocations[model_id]
        logger.info(f"Recursos liberados para el modelo {model_id}")

    async def update_usage(self, model_id: str) -> None:
        """Actualiza el timestamp de last uso de un model."""
        if model_id in self.allocations:
            self.allocations[model_id].last_used = datetime.now()

    async def get_resource_usage(self) -> Dict[str, Any]:
        """Obtiene el uso current de recursos."""
        return {
            "total": self.total_resources,
            "used": self.used_resources,
            "available": {
                resource: self.total_resources[resource] - self.used_resources[resource]
                for resource in self.total_resources
            }
        }

    async def monitor_resources(self, interval: int = 60) -> None:
        """Monitorea el uso de recursos periódicamente."""
        while True:
            usage = await self.get_resource_usage()
            logger.info(f"Estado de recursos: {usage}")
            
            # verify uso excesivo
            for resource, used in self.used_resources.items():
                if used > self.total_resources[resource] * 0.9:  # 90% de uso
                    logger.warning(f"Uso alto de {resource}: {used}/{self.total_resources[resource]}")

            await asyncio.sleep(interval)

    async def optimize_allocations(self) -> None:
        """Optimiza las asignaciones de recursos."""
        # implement lógica de optimization
        # by example, free recursos de modelos inactivos
        current_time = datetime.now()
        for model_id, allocation in list(self.allocations.items()):
            if (current_time - allocation.last_used).total_seconds() > 3600:  # 1 hora
                await self.release_resources(model_id)
                logger.info(f"Recursos liberados para el modelo inactivo {model_id}")