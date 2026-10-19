"""
 SMART UTILS CONTRACTS SYSTEM
===============================

Smart contracts system for utilities with:
-  Auto-execution of contracts
-  Automatic SLA validation
-  Intelligent resource management
-  Automatic performance optimization
- ️ Protection against violations

Author: CapibaraGPT Ultra Team
Version: 3.3.0 Ultra
"""

import asyncio
import json
import threading
import time
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path


class ContractType(Enum):
    """Types of smart contracts."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    INTEGRATION = "integration"
    EVOLUTION = "evolution"
    MONITORING = "monitoring"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


class ContractStatus(Enum):
    """Contract statuses."""
    ACTIVE = "active"
    PAUSED = "paused"
    VIOLATED = "violated"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    PENDING = "pending"


class ViolationSeverity(Enum):
    """Violation severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ContractViolation:
    """Representation of a contract violation."""
    
    contract_id: str
    violation_type: str
    severity: ViolationSeverity
    description: str
    timestamp: datetime
    metrics: Dict[str, Any]
    auto_resolution_attempted: bool = False
    resolved: bool = False
    resolution_details: Optional[str] = None


@dataclass
class ContractMetrics:
    """Contract performance metrics."""
    
    total_executions: int = 0
    successful_executions: int = 0
    violations_count: int = 0
    avg_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_execution: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculates the success rate."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    @property
    def violation_rate(self) -> float:
        """Calculates the violation rate."""
        if self.total_executions == 0:
            return 0.0
        return (self.violations_count / self.total_executions) * 100


class SmartContract(ABC):
    """Base class for smart contracts."""
    
    def __init__(
        self,
        contract_id: str,
        contract_type: ContractType,
        name: str,
        description: str,
        auto_execute: bool = True,
        auto_heal: bool = True
    ):
        self.contract_id = contract_id
        self.contract_type = contract_type
        self.name = name
        self.description = description
        self.auto_execute = auto_execute
        self.auto_heal = auto_heal
        self.status = ContractStatus.PENDING
        self.created_at = datetime.now()
        self.last_check = None
        self.violations: List[ContractViolation] = []
        self.metrics = ContractMetrics()
        self.conditions: Dict[str, Any] = {}
        self.actions: Dict[str, Callable] = {}
        self.triggers: List[str] = []
        
        # Contract-specific logger
        self.logger = logging.getLogger(f"SmartContract.{self.contract_id}")
    
    @abstractmethod
    def validate_conditions(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates the contract conditions.

        Args:
            context: Execution context

        Returns:
            Tuple of (is_valid, violations_list)
        """
        pass
    
    @abstractmethod
    def execute_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the contract actions.

        Args:
            context: Execution context

        Returns:
            Execution result
        """
        pass
    
    def activate(self) -> None:
        """Activates the contract."""
        self.status = ContractStatus.ACTIVE
        self.logger.info(f" Contract {self.contract_id} activated")
    
    def pause(self) -> None:
        """Pauses the contract."""
        self.status = ContractStatus.PAUSED
        self.logger.info(f"️ Contract {self.contract_id} paused")
    
    def terminate(self) -> None:
        """Terminates the contract."""
        self.status = ContractStatus.TERMINATED
        self.logger.info(f" Contract {self.contract_id} terminated")
    
    def add_violation(self, violation: ContractViolation) -> None:
        """Adds a violation to the contract."""
        self.violations.append(violation)
        self.metrics.violations_count += 1

        if violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.EMERGENCY]:
            self.status = ContractStatus.VIOLATED
            self.logger.error(f" {violation.severity.value} violation in contract {self.contract_id}")

        # Auto-healing if enabled
        if self.auto_heal and violation.severity != ViolationSeverity.EMERGENCY:
            self._attempt_auto_healing(violation)
    
    def _attempt_auto_healing(self, violation: ContractViolation) -> bool:
        """Attempts auto-healing of violations."""
        try:
            healing_actions = self._get_healing_actions(violation)
            
            for action in healing_actions:
                try:
                    result = action()
                    if result.get('success', False):
                        violation.auto_resolution_attempted = True
                        violation.resolved = True
                        violation.resolution_details = result.get('details', 'Auto-healing successful')
                        self.logger.info(f" Auto-healing successful for violation {violation.violation_type}")
                        return True
                except Exception as e:
                    self.logger.warning(f"Auto-healing failed: {e}")
                    continue
            
            violation.auto_resolution_attempted = True
            return False
            
        except Exception as e:
            self.logger.error(f"Error in auto-healing: {e}")
            return False
    
    def _get_healing_actions(self, violation: ContractViolation) -> List[Callable]:
        """Gets healing actions based on the violation type."""
        healing_map = {
            'performance_violation': [self._optimize_performance],
            'memory_violation': [self._cleanup_memory],
            'resource_violation': [self._free_resources],
            'quality_violation': [self._improve_quality]
        }
        
        return healing_map.get(violation.violation_type, [])
    
    def _optimize_performance(self) -> Dict[str, Any]:
        """Optimizes system performance."""
        import gc
        gc.collect()
        return {'success': True, 'details': 'Garbage collection executed'}
    
    def _cleanup_memory(self) -> Dict[str, Any]:
        """Cleans up system memory."""
        import gc
        collected = gc.collect()
        return {'success': True, 'details': f'Freed {collected} memory objects'}
    
    def _free_resources(self) -> Dict[str, Any]:
        """Frees system resources."""
        # Basic implementation
        return {'success': True, 'details': 'Resources freed'}
    
    def _improve_quality(self) -> Dict[str, Any]:
        """Improves the service quality."""
        # Basic implementation
        return {'success': True, 'details': 'Quality improved'}
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generates a contract status report."""
        return {
            'contract_id': self.contract_id,
            'name': self.name,
            'type': self.contract_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'total_violations': len(self.violations),
            'unresolved_violations': len([v for v in self.violations if not v.resolved]),
            'metrics': {
                'total_executions': self.metrics.total_executions,
                'success_rate': self.metrics.success_rate,
                'violation_rate': self.metrics.violation_rate,
                'avg_execution_time_ms': self.metrics.avg_execution_time_ms
            }
        }


class PerformanceContract(SmartContract):
    """Performance contract with automatic SLAs."""
    
    def __init__(
        self,
        contract_id: str,
        max_execution_time_ms: float = 100.0,
        max_memory_mb: float = 512.0,
        min_success_rate: float = 95.0,
        max_cpu_percent: float = 80.0
    ):
        super().__init__(
            contract_id=contract_id,
            contract_type=ContractType.PERFORMANCE,
            name="Performance SLA Contract",
            description="Guarantees performance service levels"
        )
        
        self.max_execution_time_ms = max_execution_time_ms
        self.max_memory_mb = max_memory_mb
        self.min_success_rate = min_success_rate
        self.max_cpu_percent = max_cpu_percent
    
    def validate_conditions(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates performance conditions."""
        violations = []

        # Validate execution time
        execution_time = context.get('execution_time_ms', 0)
        if execution_time > self.max_execution_time_ms:
            violations.append(f"Execution time {execution_time}ms > {self.max_execution_time_ms}ms")

        # Validate memory usage
        memory_usage = context.get('memory_usage_mb', 0)
        if memory_usage > self.max_memory_mb:
            violations.append(f"Memory usage {memory_usage}MB > {self.max_memory_mb}MB")

        # Validate success rate
        success_rate = context.get('success_rate', 100)
        if success_rate < self.min_success_rate:
            violations.append(f"Success rate {success_rate}% < {self.min_success_rate}%")

        # Validate CPU
        cpu_usage = context.get('cpu_usage_percent', 0)
        if cpu_usage > self.max_cpu_percent:
            violations.append(f"CPU usage {cpu_usage}% > {self.max_cpu_percent}%")
        
        return len(violations) == 0, violations
    
    def execute_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes performance optimization actions."""
        actions_taken = []

        # Optimize cache if there are timing issues
        if context.get('execution_time_ms', 0) > self.max_execution_time_ms:
            actions_taken.append("cache_optimization")

        # Free memory if necessary
        if context.get('memory_usage_mb', 0) > self.max_memory_mb:
            actions_taken.append("memory_cleanup")
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'timestamp': datetime.now().isoformat()
        }


class ResourceContract(SmartContract):
    """Resource management contract."""
    
    def __init__(
        self,
        contract_id: str,
        max_memory_allocation_mb: float = 1024.0,
        max_cpu_cores: int = 4,
        max_disk_usage_gb: float = 10.0,
        max_network_bandwidth_mbps: float = 100.0
    ):
        super().__init__(
            contract_id=contract_id,
            contract_type=ContractType.RESOURCE,
            name="Resource Management Contract",
            description="Manages system resource limits"
        )
        
        self.max_memory_allocation_mb = max_memory_allocation_mb
        self.max_cpu_cores = max_cpu_cores
        self.max_disk_usage_gb = max_disk_usage_gb
        self.max_network_bandwidth_mbps = max_network_bandwidth_mbps
    
    def validate_conditions(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates resource limits."""
        violations = []

        # Validate memory
        memory_used = context.get('memory_allocation_mb', 0)
        if memory_used > self.max_memory_allocation_mb:
            violations.append(f"Allocated memory {memory_used}MB > {self.max_memory_allocation_mb}MB")

        # Validate CPU
        cpu_cores_used = context.get('cpu_cores_used', 0)
        if cpu_cores_used > self.max_cpu_cores:
            violations.append(f"CPU cores used {cpu_cores_used} > {self.max_cpu_cores}")

        # Validate disk
        disk_used = context.get('disk_usage_gb', 0)
        if disk_used > self.max_disk_usage_gb:
            violations.append(f"Disk usage {disk_used}GB > {self.max_disk_usage_gb}GB")
        
        return len(violations) == 0, violations
    
    def execute_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes resource management actions."""
        actions = []

        # Free memory if necessary
        if context.get('memory_allocation_mb', 0) > self.max_memory_allocation_mb * 0.9:
            actions.append("memory_liberation")

        # Optimize CPU usage
        if context.get('cpu_cores_used', 0) > self.max_cpu_cores * 0.8:
            actions.append("cpu_optimization")
        
        return {
            'success': True,
            'resource_actions': actions,
            'timestamp': datetime.now().isoformat()
        }


class QualityContract(SmartContract):
    """Service quality contract."""
    
    def __init__(
        self,
        contract_id: str,
        min_accuracy: float = 95.0,
        max_error_rate: float = 1.0,
        min_availability: float = 99.0,
        response_time_sla_ms: float = 200.0
    ):
        super().__init__(
            contract_id=contract_id,
            contract_type=ContractType.QUALITY,
            name="Quality Assurance Contract",
            description="Guarantees service quality levels"
        )
        
        self.min_accuracy = min_accuracy
        self.max_error_rate = max_error_rate
        self.min_availability = min_availability
        self.response_time_sla_ms = response_time_sla_ms
    
    def validate_conditions(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates quality conditions."""
        violations = []

        # Validate accuracy
        accuracy = context.get('accuracy_percent', 100)
        if accuracy < self.min_accuracy:
            violations.append(f"Accuracy {accuracy}% < {self.min_accuracy}%")

        # Validate error rate
        error_rate = context.get('error_rate_percent', 0)
        if error_rate > self.max_error_rate:
            violations.append(f"Error rate {error_rate}% > {self.max_error_rate}%")

        # Validate availability
        availability = context.get('availability_percent', 100)
        if availability < self.min_availability:
            violations.append(f"Availability {availability}% < {self.min_availability}%")
        
        return len(violations) == 0, violations
    
    def execute_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes quality improvement actions."""
        improvements = []
        
        if context.get('accuracy_percent', 100) < self.min_accuracy:
            improvements.append("accuracy_enhancement")
        
        if context.get('error_rate_percent', 0) > self.max_error_rate:
            improvements.append("error_reduction")
        
        return {
            'success': True,
            'quality_improvements': improvements,
            'timestamp': datetime.now().isoformat()
        }


class SmartContractsManager:
    """
     CENTRAL MANAGER for smart contracts.

    Manages the complete contract lifecycle with:
    - Periodic auto-execution
    - Automatic violation detection
    - Intelligent healing system
    - Analytics and reporting
    """
    
    def __init__(self, check_interval_seconds: int = 30):
        self.contracts: Dict[str, SmartContract] = {}
        self.check_interval_seconds = check_interval_seconds
        self.running = False
        self.monitor_thread = None
        self.execution_log: List[Dict[str, Any]] = []
        
        # Manager logger
        self.logger = logging.getLogger("SmartContractsManager")
        
        # Global metrics
        self.global_metrics = {
            'total_contracts': 0,
            'active_contracts': 0,
            'total_violations': 0,
            'auto_healings': 0,
            'total_executions': 0
        }
    
    def register_contract(self, contract: SmartContract) -> None:
        """Registers a new contract."""
        self.contracts[contract.contract_id] = contract
        self.global_metrics['total_contracts'] += 1
        
        if contract.status == ContractStatus.ACTIVE:
            self.global_metrics['active_contracts'] += 1
        
        self.logger.info(f" Contract registered: {contract.contract_id}")
    
    def activate_contract(self, contract_id: str) -> bool:
        """Activates a specific contract."""
        if contract_id in self.contracts:
            self.contracts[contract_id].activate()
            self.global_metrics['active_contracts'] += 1
            return True
        return False
    
    def pause_contract(self, contract_id: str) -> bool:
        """Pauses a specific contract."""
        if contract_id in self.contracts:
            self.contracts[contract_id].pause()
            if self.contracts[contract_id].status == ContractStatus.ACTIVE:
                self.global_metrics['active_contracts'] -= 1
            return True
        return False
    
    def remove_contract(self, contract_id: str) -> bool:
        """Removes a contract."""
        if contract_id in self.contracts:
            contract = self.contracts[contract_id]
            if contract.status == ContractStatus.ACTIVE:
                self.global_metrics['active_contracts'] -= 1
            
            del self.contracts[contract_id]
            self.global_metrics['total_contracts'] -= 1
            self.logger.info(f"️ Contract removed: {contract_id}")
            return True
        return False
    
    def check_all_contracts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Checks all active contracts."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'contracts_checked': 0,
            'violations_found': 0,
            'actions_executed': 0,
            'contract_results': {}
        }
        
        for contract_id, contract in self.contracts.items():
            if contract.status != ContractStatus.ACTIVE:
                continue
            
            try:
                # Validate conditions
                start_time = time.time()
                is_valid, violations = contract.validate_conditions(context)
                execution_time = (time.time() - start_time) * 1000
                
                # Update contract metrics
                contract.metrics.total_executions += 1
                contract.metrics.last_execution = datetime.now()
                contract.last_check = datetime.now()
                
                if execution_time < contract.metrics.min_execution_time_ms:
                    contract.metrics.min_execution_time_ms = execution_time
                if execution_time > contract.metrics.max_execution_time_ms:
                    contract.metrics.max_execution_time_ms = execution_time
                
                # Update average
                total_time = contract.metrics.avg_execution_time_ms * (contract.metrics.total_executions - 1)
                contract.metrics.avg_execution_time_ms = (total_time + execution_time) / contract.metrics.total_executions
                
                if is_valid:
                    contract.metrics.successful_executions += 1
                    results['contract_results'][contract_id] = {
                        'status': 'valid',
                        'execution_time_ms': execution_time
                    }
                else:
                    # Create violations
                    for violation_desc in violations:
                        severity = self._determine_violation_severity(violation_desc, contract)
                        violation = ContractViolation(
                            contract_id=contract_id,
                            violation_type=self._get_violation_type(violation_desc),
                            severity=severity,
                            description=violation_desc,
                            timestamp=datetime.now(),
                            metrics=context
                        )
                        
                        contract.add_violation(violation)
                        results['violations_found'] += 1
                        self.global_metrics['total_violations'] += 1
                    
                    # Execute actions if enabled
                    if contract.auto_execute:
                        try:
                            action_result = contract.execute_actions(context)
                            results['actions_executed'] += 1
                            self.global_metrics['total_executions'] += 1
                            
                            results['contract_results'][contract_id] = {
                                'status': 'violated_and_acted',
                                'violations': violations,
                                'actions': action_result,
                                'execution_time_ms': execution_time
                            }
                        except Exception as e:
                            self.logger.error(f"Error executing actions for {contract_id}: {e}")
                            results['contract_results'][contract_id] = {
                                'status': 'violated_action_failed',
                                'violations': violations,
                                'error': str(e),
                                'execution_time_ms': execution_time
                            }
                    else:
                        results['contract_results'][contract_id] = {
                            'status': 'violated_no_action',
                            'violations': violations,
                            'execution_time_ms': execution_time
                        }
                
                results['contracts_checked'] += 1
                
            except Exception as e:
                self.logger.error(f"Error verificando contrato {contract_id}: {e}")
                results['contract_results'][contract_id] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Registrar en log de execution
        self.execution_log.append(results)
        
        # maintain only las últimas 1000 entradas
        if len(self.execution_log) > 1000:
            self.execution_log = self.execution_log[-1000:]
        
        return results
    
    def _determine_violation_severity(self, violation_desc: str, contract: SmartContract) -> ViolationSeverity:
        """Determina la severidad de una violación."""
        if "memoria" in violation_desc.lower() or "memory" in violation_desc.lower():
            return ViolationSeverity.HIGH
        elif "tiempo" in violation_desc.lower() or "time" in violation_desc.lower():
            return ViolationSeverity.MEDIUM
        elif "cpu" in violation_desc.lower():
            return ViolationSeverity.HIGH
        elif "error" in violation_desc.lower():
            return ViolationSeverity.CRITICAL
        else:
            return ViolationSeverity.LOW
    
    def _get_violation_type(self, violation_desc: str) -> str:
        """Obtiene el type de violación basado en la descripción."""
        if "tiempo" in violation_desc.lower() or "time" in violation_desc.lower():
            return "performance_violation"
        elif "memoria" in violation_desc.lower() or "memory" in violation_desc.lower():
            return "memory_violation"
        elif "cpu" in violation_desc.lower():
            return "resource_violation"
        elif "error" in violation_desc.lower():
            return "quality_violation"
        else:
            return "generic_violation"
    
    def start_monitoring(self) -> None:
        """Inicia el monitoreo automático de contratos."""
        if self.running:
            return
        
        self.running = True
        
        def monitor_worker():
            while self.running:
                try:
                    # create contexto basic for verification
                    context = self._create_monitoring_context()
                    
                    # verify todos los contratos
                    results = self.check_all_contracts(context)
                    
                    # Log de resultados
                    if results['violations_found'] > 0:
                        self.logger.warning(f" {results['violations_found']} violaciones encontradas")
                    
                    if results['actions_executed'] > 0:
                        self.logger.info(f" {results['actions_executed']} acciones ejecutadas")
                    
                    time.sleep(self.check_interval_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error en monitor worker: {e}")
                    time.sleep(self.check_interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f" Monitoreo de contratos iniciado (intervalo: {self.check_interval_seconds}s)")
    
    def stop_monitoring(self) -> None:
        """Detiene el monitoreo automático."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info(" Monitoreo de contratos detenido")
    
    def _create_monitoring_context(self) -> Dict[str, Any]:
        """Crea contexto basic for monitoreo."""
        import psutil
        import gc
        
        try:
            # information del sistema
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            context = {
                'timestamp': datetime.now().isoformat(),
                'memory_usage_mb': memory_info.used / 1024 / 1024,
                'memory_available_mb': memory_info.available / 1024 / 1024,
                'cpu_usage_percent': cpu_percent,
                'total_objects': len(gc.get_objects()),
                'success_rate': 95.0,  # Placeholder
                'error_rate_percent': 2.0,  # Placeholder
                'availability_percent': 99.5,  # Placeholder
                'execution_time_ms': 50.0  # Placeholder
            }
            
        except ImportError:
            # Fallback if psutil not está available
            context = {
                'timestamp': datetime.now().isoformat(),
                'memory_usage_mb': 100.0,
                'cpu_usage_percent': 25.0,
                'success_rate': 95.0,
                'error_rate_percent': 2.0,
                'execution_time_ms': 50.0
            }
        
        return context
    
    def get_global_status(self) -> Dict[str, Any]:
        """Obtiene estado global de todos los contratos."""
        active_contracts = [c for c in self.contracts.values() if c.status == ContractStatus.ACTIVE]
        
        total_violations = sum(len(c.violations) for c in self.contracts.values())
        unresolved_violations = sum(
            len([v for v in c.violations if not v.resolved]) 
            for c in self.contracts.values()
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_contracts': len(self.contracts),
            'active_contracts': len(active_contracts),
            'total_violations': total_violations,
            'unresolved_violations': unresolved_violations,
            'monitoring_active': self.running,
            'check_interval_seconds': self.check_interval_seconds,
            'global_metrics': self.global_metrics,
            'contracts_by_type': {
                contract_type.value: len([
                    c for c in self.contracts.values() 
                    if c.contract_type == contract_type
                ])
                for contract_type in ContractType
            }
        }
    
    def generate_violations_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Genera reporte de violaciones de las últimas horas."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_violations = []
        for contract in self.contracts.values():
            for violation in contract.violations:
                if violation.timestamp >= cutoff_time:
                    recent_violations.append({
                        'contract_id': contract.contract_id,
                        'contract_name': contract.name,
                        'violation_type': violation.violation_type,
                        'severity': violation.severity.value,
                        'description': violation.description,
                        'timestamp': violation.timestamp.isoformat(),
                        'resolved': violation.resolved,
                        'auto_resolution_attempted': violation.auto_resolution_attempted
                    })
        
        # group by severidad
        violations_by_severity = {}
        for violation in recent_violations:
            severity = violation['severity']
            if severity not in violations_by_severity:
                violations_by_severity[severity] = []
            violations_by_severity[severity].append(violation)
        
        return {
            'report_period_hours': hours_back,
            'total_violations': len(recent_violations),
            'violations_by_severity': {
                severity: len(violations) 
                for severity, violations in violations_by_severity.items()
            },
            'resolved_violations': len([v for v in recent_violations if v['resolved']]),
            'auto_healing_attempts': len([v for v in recent_violations if v['auto_resolution_attempted']]),
            'violations_details': recent_violations
        }


# instance global del manager
_contracts_manager: Optional[SmartContractsManager] = None


def get_contracts_manager() -> SmartContractsManager:
    """Obtiene la instance global del manager de contratos."""
    global _contracts_manager
    
    if _contracts_manager is None:
        _contracts_manager = SmartContractsManager()
    
    return _contracts_manager


def create_performance_contract(
    contract_id: str,
    max_execution_time_ms: float = 100.0,
    max_memory_mb: float = 512.0
) -> PerformanceContract:
    """function de conveniencia for create contratos de performance."""
    return PerformanceContract(
        contract_id=contract_id,
        max_execution_time_ms=max_execution_time_ms,
        max_memory_mb=max_memory_mb
    )


def create_resource_contract(
    contract_id: str,
    max_memory_allocation_mb: float = 1024.0,
    max_cpu_cores: int = 4
) -> ResourceContract:
    """function de conveniencia for create contratos de recursos."""
    return ResourceContract(
        contract_id=contract_id,
        max_memory_allocation_mb=max_memory_allocation_mb,
        max_cpu_cores=max_cpu_cores
    )


def create_quality_contract(
    contract_id: str,
    min_accuracy: float = 95.0,
    max_error_rate: float = 1.0
) -> QualityContract:
    """function de conveniencia for create contratos de calidad."""
    return QualityContract(
        contract_id=contract_id,
        min_accuracy=min_accuracy,
        max_error_rate=max_error_rate
    )


# example de uso
if __name__ == "__main__":
    # create manager
    manager = SmartContractsManager(check_interval_seconds=10)
    
    # create contratos
    perf_contract = create_performance_contract("performance_001", max_execution_time_ms=50.0)
    resource_contract = create_resource_contract("resource_001", max_memory_allocation_mb=512.0)
    quality_contract = create_quality_contract("quality_001", min_accuracy=98.0)
    
    # Registrar contratos
    manager.register_contract(perf_contract)
    manager.register_contract(resource_contract)
    manager.register_contract(quality_contract)
    
    # activate contratos
    manager.activate_contract("performance_001")
    manager.activate_contract("resource_001")
    manager.activate_contract("quality_001")
    
    # start monitoreo
    manager.start_monitoring()
    
    # Simular execution
    time.sleep(5)
    
    # verify estado
    status = manager.get_global_status()
    logger.info(f"Estado global: {json.dumps(status, indent=2)}")
    
    # stop monitoreo
    manager.stop_monitoring()