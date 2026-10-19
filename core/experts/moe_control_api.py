"""
MoE Control API for Capibara-6

Provides comprehensive control and monitoring capabilities for the Dynamic MoE system.
"""

import logging
import time
import math
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# JAX imports with fallbacks
try:
    from jax import numpy as jnp
except ImportError:
    import numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class ExpertStats:
    """Statistics for individual experts."""
    expert_id: int
    expert_type: str
    usage_count: int
    avg_processing_time: float
    specialization_weight: float
    last_used: float
    efficiency_score: float


@dataclass
class LayerStats:
    """Statistics for MoE layers."""
    layer_id: int
    total_tokens_processed: int
    avg_routing_entropy: float
    avg_load_balance_loss: float
    utilization_efficiency: float
    expert_stats: List[ExpertStats]


class MoEControlAPI:
    """API for controlling and monitoring the MoE system."""
    
    def __init__(self, modular_model):
        """
        Initialize MoE Control API.
        
        Args:
            modular_model: ModularCapibaraModel instance with MoE system
        """
        self.model = modular_model
        self.moe_layers = getattr(modular_model, 'moe_layers', {})
        self.moe_manager = getattr(modular_model, 'moe_manager', None)
        
        # API state
        self.monitoring_enabled = True
        self.last_health_check = time.time()
        self.performance_history = []
        self._last_layer_snapshots = {}
        
        logger.info("MoE Control API initialized")
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        
        if not self.moe_layers:
            return {
                "status": "unavailable",
                "message": "MoE system not available",
                "timestamp": time.time()
            }
            
        try:
            # Check layer health
            layer_health = {}
            total_active_layers = 0
            total_layers = len(self.moe_layers)
            
            for layer_name, moe_layer in self.moe_layers.items():
                try:
                    # Compute health check from layer stats
                    layer_metrics = self._get_layer_health_metrics(layer_name, moe_layer)
                    layer_health[layer_name] = layer_metrics
                    
                    if layer_metrics["status"] == "healthy":
                        total_active_layers += 1
                        
                except Exception as e:
                    layer_health[layer_name] = {
                        "status": "error",
                        "error": str(e),
                        "last_check": time.time()
                    }
            
            # Determine overall health
            health_ratio = total_active_layers / total_layers if total_layers > 0 else 0
            
            if health_ratio >= 0.9:
                overall_status = "excellent"
            elif health_ratio >= 0.7:
                overall_status = "good"
            elif health_ratio >= 0.5:
                overall_status = "degraded"
            else:
                overall_status = "critical"
                
            self.last_health_check = time.time()
            
            return {
                "status": overall_status,
                "health_ratio": health_ratio,
                "active_layers": total_active_layers,
                "total_layers": total_layers,
                "layer_health": layer_health,
                "last_check": self.last_health_check,
                "monitoring_enabled": self.monitoring_enabled
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
            
    def _get_layer_health_metrics(self, layer_name: str, moe_layer) -> Dict[str, Any]:
        """Get health metrics for a specific layer from real stats."""
        
        try:
            current_time = time.time()
            layer_stats = self._get_layer_stats(moe_layer)
            expert_stats = layer_stats.get("expert_stats", [])
            expert_count = len(expert_stats) or getattr(moe_layer, 'config', {}).get('num_experts', 0)

            total_usage = sum(stat.get("usage_count", 0) for stat in expert_stats)
            usage_probs = []
            if total_usage > 0:
                usage_probs = [stat.get("usage_count", 0) / total_usage for stat in expert_stats]
            elif expert_stats:
                usage_probs = [1.0 / len(expert_stats)] * len(expert_stats)

            routing_efficiency = 0.0
            load_balance = 0.0
            if usage_probs:
                entropy = -sum(p * math.log(p + 1e-12) for p in usage_probs)
                max_entropy = math.log(len(usage_probs)) if len(usage_probs) > 1 else 1.0
                routing_efficiency = entropy / max_entropy if max_entropy > 0 else 0.0

                mean_usage = sum(usage_probs) / len(usage_probs)
                variance = sum((p - mean_usage) ** 2 for p in usage_probs) / len(usage_probs)
                load_balance = max(0.0, 1.0 - variance * len(usage_probs))

            memory_usage_mb = 0.0
            try:
                import psutil
                memory_usage_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            except Exception:
                memory_usage_mb = 0.0

            if routing_efficiency > 0.8 and load_balance > 0.75:
                status = "healthy"
            elif routing_efficiency > 0.6 and load_balance > 0.6:
                status = "degraded"
            else:
                status = "unhealthy"

            return {
                "status": status,
                "routing_efficiency": routing_efficiency,
                "load_balance": load_balance,
                "memory_usage_mb": memory_usage_mb,
                "last_update": current_time,
                "expert_count": expert_count
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_check": time.time()
            }
            
    def get_expert_specialization_report(self) -> Dict[str, Any]:
        """Generate detailed expert specialization report."""
        
        if not self.moe_layers:
            return {"error": "MoE system not available"}
            
        try:
            specialization_report = {}
            
            for layer_name, moe_layer in self.moe_layers.items():
                layer_report = {
                    "expert_types": [],
                    "utilization": [],
                    "specialization_scores": [],
                    "efficiency_metrics": {}
                }
                
                # Analyze each expert in the layer
                layer_stats = self._get_layer_stats(moe_layer)
                expert_stats = layer_stats.get("expert_stats", [])
                total_usage = sum(stat.get("usage_count", 0) for stat in expert_stats) or 1

                max_weight = max(
                    (stat.get("specialization_weight", 1.0) for stat in expert_stats),
                    default=1.0
                )

                for stat in expert_stats:
                    expert_type = stat.get("expert_type", "general")
                    layer_report["expert_types"].append(expert_type)

                    utilization = stat.get("usage_count", 0) / total_usage
                    layer_report["utilization"].append(utilization)

                    weight = stat.get("specialization_weight", 1.0)
                    score = min(1.0, max(0.0, weight / max_weight))
                    layer_report["specialization_scores"].append(score)
                    
                # Calculate layer efficiency metrics
                if layer_report["utilization"]:
                    avg_utilization = sum(layer_report["utilization"]) / len(layer_report["utilization"])
                    avg_specialization = sum(layer_report["specialization_scores"]) / len(layer_report["specialization_scores"])
                    
                    layer_report["efficiency_metrics"] = {
                        "avg_utilization": avg_utilization,
                        "avg_specialization": avg_specialization,
                        "balance_score": 1.0 - self._calculate_variance(layer_report["utilization"]),
                        "diversity_score": len(set(layer_report["expert_types"])) / len(layer_report["expert_types"])
                    }
                    
                specialization_report[layer_name] = layer_report
                
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_specialization(specialization_report)
            recommendations = self._generate_specialization_recommendations(specialization_report)
            
            return {
                "specialization_by_layer": specialization_report,
                "overall_metrics": overall_metrics,
                "recommendations": recommendations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating specialization report: {e}")
            return {"error": str(e)}
            
    def _calculate_overall_specialization(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall specialization metrics."""
        
        all_utilization = []
        all_specialization = []
        
        for layer_data in report.values():
            all_utilization.extend(layer_data.get("utilization", []))
            all_specialization.extend(layer_data.get("specialization_scores", []))
            
        if not all_utilization or not all_specialization:
            return {"error": "No data available"}
            
        return {
            "avg_utilization": sum(all_utilization) / len(all_utilization),
            "avg_specialization": sum(all_specialization) / len(all_specialization),
            "min_utilization": min(all_utilization),
            "max_utilization": max(all_utilization),
            "utilization_variance": self._calculate_variance(all_utilization),
            "specialization_variance": self._calculate_variance(all_specialization),
            "total_experts": len(all_utilization),
            "active_experts": sum(1 for u in all_utilization if u > 0.1)
        }
        
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) <= 1:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
        
    def _generate_specialization_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving specialization."""
        
        recommendations = []
        
        for layer_name, layer_data in report.items():
            efficiency = layer_data.get("efficiency_metrics", {})
            
            avg_util = efficiency.get("avg_utilization", 0)
            avg_spec = efficiency.get("avg_specialization", 0)
            balance_score = efficiency.get("balance_score", 0)
            
            if avg_util < 0.5:
                recommendations.append(
                    f"{layer_name}: Low average utilization ({avg_util:.2f}), "
                    "consider rebalancing routing weights"
                )
                
            if avg_spec < 0.7:
                recommendations.append(
                    f"{layer_name}: Low specialization ({avg_spec:.2f}), "
                    "needs more domain-specific training"
                )
                
            if balance_score < 0.6:
                recommendations.append(
                    f"{layer_name}: Poor load balancing ({balance_score:.2f}), "
                    "adjust routing temperature or capacity factor"
                )
                
            # Check for underutilized experts
            utilization = layer_data.get("utilization", [])
            underused_count = sum(1 for u in utilization if u < 0.2)
            if underused_count > len(utilization) * 0.3:
                recommendations.append(
                    f"{layer_name}: {underused_count} underutilized experts, "
                    "consider expert pruning or retraining"
                )
                
        return recommendations if recommendations else ["MoE system operating optimally"]
        
    def configure_expert_routing(
        self, 
        layer_name: str, 
        routing_temperature: float = 1.0,
        num_active_experts: int = 4,
        load_balance_weight: float = 0.01
    ) -> Dict[str, Any]:
        """Configure routing parameters for a specific layer."""
        
        if layer_name not in self.moe_layers:
            return {"error": f"Layer {layer_name} not found"}
            
        try:
            moe_layer = self.moe_layers[layer_name]
            
            # Update configuration
            if hasattr(moe_layer, 'router') and hasattr(moe_layer.router, 'config'):
                config = moe_layer.router.config
                config.routing_temperature = routing_temperature
                config.num_active_experts = num_active_experts
                config.load_balance_weight = load_balance_weight
                
                logger.info(f"Updated routing config for {layer_name}")
                
                return {
                    "success": True,
                    "layer": layer_name,
                    "new_config": {
                        "routing_temperature": routing_temperature,
                        "num_active_experts": num_active_experts,
                        "load_balance_weight": load_balance_weight
                    },
                    "timestamp": time.time()
                }
            else:
                return {"error": f"Cannot access router config for {layer_name}"}
                
        except Exception as e:
            logger.error(f"Error configuring {layer_name}: {e}")
            return {"error": str(e)}
            
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics from the MoE system."""
        
        if not self.moe_layers:
            return {"error": "MoE system not available"}
            
        try:
            metrics = {}
            current_time = time.time()
            
            for layer_name, moe_layer in self.moe_layers.items():
                layer_stats = self._get_layer_stats(moe_layer)
                expert_stats = layer_stats.get("expert_stats", [])
                total_usage = sum(stat.get("usage_count", 0) for stat in expert_stats)

                usage_probs = []
                if total_usage > 0 and expert_stats:
                    usage_probs = [stat.get("usage_count", 0) / total_usage for stat in expert_stats]

                routing_efficiency = 0.0
                expert_balance = 0.0
                if usage_probs:
                    entropy = -sum(p * math.log(p + 1e-12) for p in usage_probs)
                    max_entropy = math.log(len(usage_probs)) if len(usage_probs) > 1 else 1.0
                    routing_efficiency = entropy / max_entropy if max_entropy > 0 else 0.0

                    mean_usage = sum(usage_probs) / len(usage_probs)
                    variance = sum((p - mean_usage) ** 2 for p in usage_probs) / len(usage_probs)
                    expert_balance = max(0.0, 1.0 - variance * len(usage_probs))

                avg_processing_time = 0.0
                if expert_stats:
                    avg_processing_time = sum(
                        stat.get("avg_processing_time", 0.0) for stat in expert_stats
                    ) / len(expert_stats)

                memory_usage_mb = 0.0
                try:
                    import psutil
                    memory_usage_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                except Exception:
                    memory_usage_mb = 0.0

                total_tokens = layer_stats.get("total_tokens_processed", 0)
                previous = self._last_layer_snapshots.get(layer_name)
                if previous:
                    delta_time = max(current_time - previous["timestamp"], 1e-6)
                    delta_tokens = total_tokens - previous["total_tokens"]
                    current_load = max(0.0, delta_tokens / delta_time)
                else:
                    current_load = 0.0

                self._last_layer_snapshots[layer_name] = {
                    "timestamp": current_time,
                    "total_tokens": total_tokens
                }

                layer_metrics = {
                    "active": True,
                    "current_load": current_load,
                    "routing_efficiency": routing_efficiency,
                    "expert_balance": expert_balance,
                    "memory_usage_mb": memory_usage_mb,
                    "tokens_processed": total_tokens,
                    "avg_response_time_ms": avg_processing_time * 1000.0,
                    "last_update": current_time
                }
                
                metrics[layer_name] = layer_metrics
                
            # Calculate system-wide metrics
            system_metrics = self._calculate_system_metrics(metrics)
            
            snapshot = {
                "layer_metrics": metrics,
                "system_metrics": system_metrics,
                "system_health": self._assess_system_health(metrics),
                "timestamp": current_time
            }
            self.performance_history.append(snapshot)
            if len(self.performance_history) > 2000:
                self.performance_history = self.performance_history[-2000:]
            return snapshot
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {"error": str(e)}
            
    def _calculate_system_metrics(self, layer_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system-wide metrics from layer metrics."""
        
        if not layer_metrics:
            return {}
            
        total_load = sum(m.get("current_load", 0) for m in layer_metrics.values())
        total_memory = sum(m.get("memory_usage_mb", 0) for m in layer_metrics.values())
        total_tokens = sum(m.get("tokens_processed", 0) for m in layer_metrics.values())
        
        avg_efficiency = sum(m.get("routing_efficiency", 0) for m in layer_metrics.values()) / len(layer_metrics)
        avg_balance = sum(m.get("expert_balance", 0) for m in layer_metrics.values()) / len(layer_metrics)
        avg_response_time = sum(m.get("avg_response_time_ms", 0) for m in layer_metrics.values()) / len(layer_metrics)
        
        return {
            "total_system_load": total_load,
            "total_memory_usage_mb": total_memory,
            "total_tokens_processed": total_tokens,
            "avg_routing_efficiency": avg_efficiency,
            "avg_expert_balance": avg_balance,
            "avg_response_time_ms": avg_response_time,
            "active_layers": len(layer_metrics),
            "throughput_tokens_per_second": total_tokens / 60  # Assuming 1-minute window
        }
        
    def _assess_system_health(self, metrics: Dict[str, Any]) -> str:
        """Assess overall system health based on metrics."""
        
        active_layers = sum(1 for m in metrics.values() if m.get("active", False))
        total_layers = len(metrics)
        
        if active_layers == 0:
            return "critical"
        elif active_layers < total_layers * 0.8:
            return "degraded"
            
        # Evaluate performance metrics
        avg_efficiency = sum(m.get("routing_efficiency", 0) for m in metrics.values()) / len(metrics)
        avg_load = sum(m.get("current_load", 0) for m in metrics.values()) / len(metrics)
        
        if avg_efficiency > 0.85 and avg_load < 0.8:
            return "excellent"
        elif avg_efficiency > 0.7 and avg_load < 0.9:
            return "good"
        else:
            return "needs_attention"
            
    def optimize_system(self) -> Dict[str, Any]:
        """Perform system optimization."""
        
        if not self.moe_manager:
            return {"error": "MoE manager not available"}
            
        try:
            # Trigger optimization in MoE manager
            if hasattr(self.moe_manager, 'optimize_all_layers'):
                self.moe_manager.optimize_all_layers()
                
            # Clean up unused experts
            if hasattr(self.moe_manager, 'cleanup_unused_experts'):
                self.moe_manager.cleanup_unused_experts()
                
            optimization_time = time.time()
            
            return {
                "success": True,
                "optimization_completed": optimization_time,
                "actions_performed": [
                    "Optimized routing for all layers",
                    "Cleaned up unused experts",
                    "Updated performance metrics"
                ],
                "next_optimization": optimization_time + 3600  # 1 hour
            }
            
        except Exception as e:
            logger.error(f"Error during system optimization: {e}")
            return {"error": str(e)}
            
    def get_performance_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance history for the specified time period."""
        
        current_time = time.time()
        start_time = current_time - (hours * 3600)
        history_points = [
            point for point in self.performance_history
            if point.get("timestamp", 0) >= start_time
        ]

        if not history_points:
            return {
                "time_period_hours": hours,
                "data_points": 0,
                "history": [],
                "summary": {}
            }

        efficiencies = []
        loads = []
        memory_usages = []
        throughputs = []

        for point in history_points:
            system_metrics = point.get("system_metrics", {})
            efficiencies.append(system_metrics.get("avg_routing_efficiency", 0.0))
            loads.append(system_metrics.get("total_system_load", 0.0))
            memory_usages.append(system_metrics.get("total_memory_usage_mb", 0.0))
            throughputs.append(system_metrics.get("throughput_tokens_per_second", 0.0))

        return {
            "time_period_hours": hours,
            "data_points": len(history_points),
            "history": history_points,
            "summary": {
                "avg_efficiency": sum(efficiencies) / max(len(efficiencies), 1),
                "peak_throughput": max(throughputs) if throughputs else 0.0,
                "avg_memory_usage_mb": sum(memory_usages) / max(len(memory_usages), 1),
                "avg_system_load": sum(loads) / max(len(loads), 1)
            }
        }

    def _get_layer_stats(self, moe_layer) -> Dict[str, Any]:
        """Safely collect stats from a MoE layer."""
        if hasattr(moe_layer, "get_layer_stats"):
            try:
                return moe_layer.get_layer_stats()
            except Exception:
                pass

        experts = getattr(moe_layer, "experts", [])
        expert_stats = []
        for expert in experts:
            if hasattr(expert, "get_stats"):
                expert_stats.append(expert.get_stats())
            else:
                expert_stats.append({})

        return {
            "layer_id": getattr(moe_layer, "layer_id", None),
            "total_tokens_processed": getattr(moe_layer, "total_tokens_processed", 0),
            "expert_stats": expert_stats
        }
        
    def enable_monitoring(self) -> Dict[str, Any]:
        """Enable system monitoring."""
        self.monitoring_enabled = True
        logger.info("MoE monitoring enabled")
        return {"success": True, "monitoring_enabled": True, "timestamp": time.time()}
        
    def disable_monitoring(self) -> Dict[str, Any]:
        """Disable system monitoring."""
        self.monitoring_enabled = False
        logger.info("MoE monitoring disabled")
        return {"success": True, "monitoring_enabled": False, "timestamp": time.time()}
        
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and capabilities."""
        return {
            "api_version": "1.0.0",
            "moe_system_available": len(self.moe_layers) > 0,
            "total_layers": len(self.moe_layers),
            "monitoring_enabled": self.monitoring_enabled,
            "capabilities": [
                "system_health_monitoring",
                "expert_specialization_analysis",
                "real_time_metrics",
                "routing_configuration",
                "performance_optimization",
                "historical_data_analysis"
            ],
            "endpoints": {
                "health": "/moe/health",
                "metrics": "/moe/metrics", 
                "specialization": "/moe/specialization",
                "configure": "/moe/configure",
                "optimize": "/moe/optimize",
                "history": "/moe/history"
            }
        }
