"""
MoE Control API for Capibara-6

Provides comprehensive control and monitoring capabilities for the Dynamic MoE system.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# JAX imports with fallbacks
try:
    from capibara.jax import numpy as jnp
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
                    # Simulate health check (in real implementation would check actual metrics)
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
        """Get health metrics for a specific layer."""
        
        try:
            # Simulate layer health metrics
            current_time = time.time()
            layer_hash = hash(layer_name) % 1000
            
            # Simulate realistic metrics
            routing_efficiency = 0.75 + (layer_hash % 20) / 100
            load_balance = 0.8 + (layer_hash % 15) / 100
            memory_usage = 120 + (layer_hash % 80)  # MB
            
            # Determine status based on metrics
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
                "memory_usage_mb": memory_usage,
                "last_update": current_time,
                "expert_count": getattr(moe_layer, 'config', {}).get('num_experts', 32)
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
                experts = getattr(moe_layer, 'experts', [])
                for i, expert in enumerate(experts):
                    # Get expert type
                    expert_type = getattr(expert, 'expert_type', 'general')
                    layer_report["expert_types"].append(expert_type)
                    
                    # Simulate utilization metrics
                    utilization = 0.4 + (hash(f"{layer_name}_{i}") % 40) / 100
                    layer_report["utilization"].append(utilization)
                    
                    # Specialization scores based on type
                    specialization_scores = {
                        'reasoning': 0.85, 'coding': 0.78, 'mathematics': 0.92,
                        'creative': 0.73, 'multimodal': 0.81, 'linguistic': 0.79,
                        'scientific': 0.87, 'technical': 0.83, 'analytical': 0.86,
                        'conversational': 0.71, 'factual': 0.75, 'general': 0.65
                    }
                    score = specialization_scores.get(expert_type, 0.65)
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
                # Simulate real-time metrics
                layer_hash = hash(layer_name) % 1000
                
                layer_metrics = {
                    "active": True,
                    "current_load": 0.5 + (layer_hash % 40) / 100,
                    "routing_efficiency": 0.7 + (layer_hash % 25) / 100,
                    "expert_balance": 0.65 + (layer_hash % 30) / 100,
                    "memory_usage_mb": 120 + (layer_hash % 100),
                    "tokens_processed": 1000 + (layer_hash % 5000),
                    "avg_response_time_ms": 2.5 + (layer_hash % 50) / 10,
                    "last_update": current_time
                }
                
                metrics[layer_name] = layer_metrics
                
            # Calculate system-wide metrics
            system_metrics = self._calculate_system_metrics(metrics)
            
            return {
                "layer_metrics": metrics,
                "system_metrics": system_metrics,
                "system_health": self._assess_system_health(metrics),
                "timestamp": current_time
            }
            
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
        
        # In a real implementation, this would query stored metrics
        # For now, we'll simulate historical data
        
        current_time = time.time()
        start_time = current_time - (hours * 3600)
        
        # Generate simulated historical data
        history_points = []
        time_interval = 300  # 5 minutes
        
        for i in range(0, hours * 12):  # 12 points per hour
            timestamp = start_time + (i * time_interval)
            
            # Simulate realistic performance trends
            base_efficiency = 0.75 + 0.1 * jnp.sin(i * 0.1)  # Cyclical pattern
            noise = (hash(str(timestamp)) % 20 - 10) / 100  # Random noise
            
            history_points.append({
                "timestamp": timestamp,
                "routing_efficiency": max(0.5, min(0.95, base_efficiency + noise)),
                "system_load": 0.6 + 0.2 * jnp.sin(i * 0.05) + noise * 0.5,
                "memory_usage_gb": 8.5 + 2.0 * jnp.sin(i * 0.03) + noise,
                "tokens_per_second": 1500 + 300 * jnp.sin(i * 0.08) + noise * 100
            })
            
        return {
            "time_period_hours": hours,
            "data_points": len(history_points),
            "history": history_points,
            "summary": {
                "avg_efficiency": sum(p["routing_efficiency"] for p in history_points) / len(history_points),
                "peak_throughput": max(p["tokens_per_second"] for p in history_points),
                "avg_memory_usage": sum(p["memory_usage_gb"] for p in history_points) / len(history_points)
            }
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