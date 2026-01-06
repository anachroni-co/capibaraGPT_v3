"""
VQ Monitoring Module for Capibara-6

Comprehensive monitoring and analytics for VQ systems:
- Real-time performance tracking
- Resource utilization monitoring
- Codebook health analysis
- Adaptation decision logging
- Performance regression detection
- Integration with observability systems
"""

import logging
import time
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configurestion for VQ monitoring."""
    # Monitoring settings
    enable_real_time_monitoring: bool = True
    metrics_collection_interval: float = 1.0  # seconds
    performance_window_size: int = 100
    
    # Storage settings
    enable_persistent_logging: bool = True
    log_directory: str = "./vq_monitoring_logs"
    max_log_files: int = 10
    log_rotation_size_mb: float = 50.0
    
    # Alert settings
    enable_alerts: bool = True
    performance_threshold: float = 0.8
    memory_threshold: float = 0.9
    latency_threshold_ms: float = 100.0
    
    # Analysis settings
    enable_trend_analysis: bool = True
    trend_analysis_window: int = 1000
    anomaly_detection_sensitivity: float = 2.0  # Standard deviations
    
    # Integration settings
    enable_prometheus_export: bool = False
    prometheus_port: int = 8090
    enable_tensorboard_logging: bool = False
    tensorboard_log_dir: str = "./vq_tensorboard_logs"


class MetricsCollector:
    """Collect and aggregate VQ metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_buffer = deque(maxlen=config.performance_window_size)
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # Aggregated statistics
        self.aggregated_stats = defaultdict(list)
        self.performance_trends = {}
        
    def record_metrics(self, metrics: Dict[str, Any], source: str = "unknown"):
        """Record metrics from a VQ operation."""
        
        timestamp = time.time()
        
        # Create metrics entry
        metrics_entry = {
            'timestamp': timestamp,
            'source': source,
            'metrics': metrics.copy(),
            'elapsed_since_start': timestamp - self.start_time
        }
        
        with self.lock:
            self.metrics_buffer.append(metrics_entry)
            
            # Update aggregated statistics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.aggregated_stats[key].append(value)
                    
                    # Keep only recent values
                    if len(self.aggregated_stats[key]) > self.config.performance_window_size:
                        self.aggregated_stats[key].pop(0)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current aggregated statistics."""
        
        with self.lock:
            if not self.metrics_buffer:
                return {'status': 'no_data'}
            
            stats = {
                'total_samples': len(self.metrics_buffer),
                'time_range': {
                    'start': self.metrics_buffer[0]['timestamp'],
                    'end': self.metrics_buffer[-1]['timestamp'],
                    'duration_seconds': self.metrics_buffer[-1]['timestamp'] - self.metrics_buffer[0]['timestamp']
                },
                'metrics_summary': {}
            }
            
            # Compute summary statistics for each metric
            for metric_name, values in self.aggregated_stats.items():
                if values:
                    stats['metrics_summary'][metric_name] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            return stats
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics entries."""
        
        with self.lock:
            return list(self.metrics_buffer)[-count:]
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        
        anomalies = []
        
        with self.lock:
            for metric_name, values in self.aggregated_stats.items():
                if len(values) < 10:  # Need sufficient data
                    continue
                
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                
                if std_val == 0:
                    continue
                
                # Check recent values for anomalies
                recent_values = values[-10:]
                for i, value in enumerate(recent_values):
                    z_score = abs(value - mean_val) / std_val
                    
                    if z_score > self.config.anomaly_detection_sensitivity:
                        anomalies.append({
                            'metric': metric_name,
                            'value': value,
                            'expected_range': (mean_val - 2*std_val, mean_val + 2*std_val),
                            'z_score': z_score,
                            'timestamp': time.time() - (len(recent_values) - i),
                            'severity': 'high' if z_score > 3.0 else 'medium'
                        })
        
        return anomalies


class PerformanceAnalyzer:
    """Analyze VQ performance trends and patterns."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.analysis_history = deque(maxlen=config.trend_analysis_window)
        
    def analyze_performance_trend(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Analyze performance trends."""
        
        stats = metrics_collector.get_current_stats()
        
        if 'metrics_summary' not in stats:
            return {'status': 'insufficient_data'}
        
        trends = {}
        
        # Analyze each metric
        for metric_name, summary in stats['metrics_summary'].items():
            if summary['count'] < 20:  # Need sufficient data
                continue
            
            # Get recent vs older values
            values = metrics_collector.aggregated_stats[metric_name]
            if len(values) < 20:
                continue
            
            recent_mean = statistics.mean(values[-10:])
            older_mean = statistics.mean(values[-20:-10])
            
            # Determine trend
            if recent_mean > older_mean * 1.05:
                trend = "improving"
            elif recent_mean < older_mean * 0.95:
                trend = "degrading"
            else:
                trend = "stable"
            
            # Calculate trend strength
            trend_strength = abs(recent_mean - older_mean) / older_mean if older_mean != 0 else 0
            
            trends[metric_name] = {
                'trend': trend,
                'strength': trend_strength,
                'recent_mean': recent_mean,
                'older_mean': older_mean,
                'change_percent': ((recent_mean - older_mean) / older_mean * 100) if older_mean != 0 else 0
            }
        
        return {
            'timestamp': time.time(),
            'trends': trends,
            'overall_health': self._compute_overall_health(trends),
            'recommendations': self._generate_recommendations(trends)
        }
    
    def _compute_overall_health(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall system health score."""
        
        if not trends:
            return {'score': 0.0, 'status': 'unknown'}
        
        # Weight different metrics
        metric_weights = {
            'mse': 0.3,
            'codebook_usage': 0.2,
            'perplexity': 0.2,
            'commitment_loss': 0.15,
            'codebook_loss': 0.15
        }
        
        health_score = 0.0
        total_weight = 0.0
        
        for metric_name, trend_info in trends.items():
            weight = metric_weights.get(metric_name, 0.1)
            
            # Score based on trend
            if trend_info['trend'] == 'improving':
                metric_score = 1.0
            elif trend_info['trend'] == 'stable':
                metric_score = 0.8
            else:  # degrading
                metric_score = 0.4
            
            # Adjust by trend strength
            metric_score *= (1.0 - min(0.5, trend_info['strength']))
            
            health_score += metric_score * weight
            total_weight += weight
        
        final_score = health_score / total_weight if total_weight > 0 else 0.0
        
        # Determine status
        if final_score > 0.8:
            status = 'excellent'
        elif final_score > 0.6:
            status = 'good'
        elif final_score > 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'score': final_score,
            'status': status,
            'contributing_metrics': len(trends)
        }
    
    def _generate_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generates optimization recommendations based on trends."""
        
        recommendations = []
        
        for metric_name, trend_info in trends.items():
            if trend_info['trend'] == 'degrading' and trend_info['strength'] > 0.1:
                
                if metric_name == 'mse' and trend_info['change_percent'] > 10:
                    recommendations.append(
                        f"MSE increasing ({trend_info['change_percent']:.1f}%) - consider increasing codebook size or adjusting learning rate"
                    )
                
                elif metric_name == 'codebook_usage' and trend_info['recent_mean'] < 0.5:
                    recommendations.append(
                        f"Low codebook usage ({trend_info['recent_mean']:.2f}) - consider reducing codebook size"
                    )
                
                elif metric_name == 'perplexity' and trend_info['change_percent'] < -20:
                    recommendations.append(
                        f"Perplexity decreasing significantly - codebook may be underutilized"
                    )
                
                elif metric_name in ['commitment_loss', 'codebook_loss'] and trend_info['change_percent'] > 20:
                    recommendations.append(
                        f"{metric_name} increasing - check training stability and learning rates"
                    )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance is stable - no immediate optimizations needed")
        
        return recommendations


class VQMonitor:
    """
    Comprehensive VQ monitoring system.
    
    Features:
    - Real-time metrics collection
    - Performance trend analysis
    - Anomaly detection
    - Resource monitoring
    - Automated alerting
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        if config is None:
            config = MonitoringConfig()
        self.config = config
        
        # Core components
        self.metrics_collector = MetricsCollector(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks = []
        
        # Logging setup
        if config.enable_persistent_logging:
            self.log_directory = Path(config.log_directory)
            self.log_directory.mkdir(exist_ok=True)
            self.current_log_file = None
            self._setup_logging()
        
        logger.info("VQ Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        if self.config.enable_real_time_monitoring:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            self.monitoring_thread = None
        
        logger.info("Monitoring stopped")
    
    def record_vq_operation(self, 
                           vq_instance: Any,
                           inputs: Any,
                           outputs: Tuple[Any, Any, Dict[str, Any]],
                           operation_time: float,
                           memory_usage: Optional[float] = None):
        """
        Record a VQ operation for monitoring.
        
        Args:
            vq_instance: VQ instance that performed the operation
            inputs: Input data
            outputs: VQ outputs (quantized, indices, metrics)
            operation_time: Time taken for operation
            memory_usage: Memory usage in MB
        """
        
        quantized, indices, vq_metrics = outputs
        
        # Enhance metrics with monitoring information
        enhanced_metrics = vq_metrics.copy()
        enhanced_metrics.update({
            'operation_time_ms': operation_time * 1000,
            'memory_usage_mb': memory_usage,
            'input_shape': getattr(inputs, 'shape', None),
            'output_shape': getattr(quantized, 'shape', None),
            'vq_type': type(vq_instance).__name__,
            'timestamp': time.time()
        })
        
        # Record metrics
        self.metrics_collector.record_metrics(enhanced_metrics, source=type(vq_instance).__name__)
        
        # Check for alerts
        if self.config.enable_alerts:
            self._check_alerts(enhanced_metrics)
        
        # Log to file if enabled
        if self.config.enable_persistent_logging:
            self._log_to_file(enhanced_metrics)
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generates comprehensive monitoring report."""
        
        # Get current statistics
        current_stats = self.metrics_collector.get_current_stats()
        
        # Perform trend analysis
        trend_analysis = self.performance_analyzer.analyze_performance_trend(self.metrics_collector)
        
        # Detect anomalies
        anomalies = self.metrics_collector.detect_anomalies()
        
        # Resource utilization
        resource_stats = self._get_resource_stats()
        
        report = {
            'report_timestamp': time.time(),
            'monitoring_duration_hours': (time.time() - self.start_time) / 3600,
            'current_statistics': current_stats,
            'trend_analysis': trend_analysis,
            'anomalies': anomalies,
            'resource_utilization': resource_stats,
            'system_health': trend_analysis.get('overall_health', {}),
            'recommendations': trend_analysis.get('recommendations', []),
            'alert_summary': self._get_alert_summary()
        }
        
        return report
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get data for performance dashboard."""
        
        recent_metrics = self.metrics_collector.get_recent_metrics(50)
        
        if not recent_metrics:
            return {'status': 'no_data'}
        
        # Extract time series data
        time_series = {}
        for entry in recent_metrics:
            timestamp = entry['timestamp']
            for metric_name, value in entry['metrics'].items():
                if isinstance(value, (int, float)):
                    if metric_name not in time_series:
                        time_series[metric_name] = {'timestamps': [], 'values': []}
                    time_series[metric_name]['timestamps'].append(timestamp)
                    time_series[metric_name]['values'].append(value)
        
        # Current status
        latest_entry = recent_metrics[-1]
        current_status = {
            'timestamp': latest_entry['timestamp'],
            'vq_type': latest_entry['source'],
            'key_metrics': {
                'mse': latest_entry['metrics'].get('mse', 0),
                'codebook_usage': latest_entry['metrics'].get('codebook_usage', 0),
                'perplexity': latest_entry['metrics'].get('perplexity', 0),
                'operation_time_ms': latest_entry['metrics'].get('operation_time_ms', 0)
            }
        }
        
        return {
            'current_status': current_status,
            'time_series': time_series,
            'monitoring_active': self.monitoring_active,
            'total_operations': len(recent_metrics)
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time analysis."""
        
        while self.monitoring_active:
            try:
                # Perform periodic analysis
                if len(self.metrics_collector.metrics_buffer) > 10:
                    # Check for performance regressions
                    self._check_performance_regression()
                    
                    # Update trend analysis
                    if self.config.enable_trend_analysis:
                        self.performance_analyzer.analyze_performance_trend(self.metrics_collector)
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check if any alerts should be triggered."""
        
        alerts = []
        
        # Performance alerts
        if 'mse' in metrics and metrics['mse'] > 1.0:  # High reconstruction error
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"High MSE detected: {metrics['mse']:.4f}",
                'metric': 'mse',
                'value': metrics['mse']
            })
        
        # Memory alerts
        if 'memory_usage_mb' in metrics and metrics['memory_usage_mb']:
            if metrics['memory_usage_mb'] > self.config.memory_threshold * 1000:  # Convert to MB
                alerts.append({
                    'type': 'resource',
                    'severity': 'critical',
                    'message': f"High memory usage: {metrics['memory_usage_mb']:.1f}MB",
                    'metric': 'memory_usage_mb',
                    'value': metrics['memory_usage_mb']
                })
        
        # Latency alerts
        if 'operation_time_ms' in metrics:
            if metrics['operation_time_ms'] > self.config.latency_threshold_ms:
                alerts.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'message': f"High latency: {metrics['operation_time_ms']:.1f}ms",
                    'metric': 'operation_time_ms',
                    'value': metrics['operation_time_ms']
                })
        
        # Codebook utilization alerts
        if 'codebook_usage' in metrics:
            usage = metrics['codebook_usage']
            if usage < 0.1:
                alerts.append({
                    'type': 'efficiency',
                    'severity': 'info',
                    'message': f"Low codebook utilization: {usage:.1%}",
                    'metric': 'codebook_usage',
                    'value': usage
                })
            elif usage > 0.95:
                alerts.append({
                    'type': 'capacity',
                    'severity': 'warning',
                    'message': f"High codebook utilization: {usage:.1%} - consider increasing size",
                    'metric': 'codebook_usage',
                    'value': usage
                })
        
        # Trigger alert callbacks
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert callbacks."""
        
        alert['timestamp'] = time.time()
        
        # Log alert
        severity_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨'
        }
        
        emoji = severity_emoji.get(alert['severity'], '📊')
        logger.warning(f"{emoji} VQ Alert [{alert['severity'].upper()}]: {alert['message']}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _check_performance_regression(self):
        """Check for performance regressions."""
        
        # Simple regression detection based on recent vs historical performance
        recent_metrics = self.metrics_collector.get_recent_metrics(20)
        
        if len(recent_metrics) < 20:
            return
        
        # Check MSE trend
        recent_mse = [m['metrics'].get('mse', 0) for m in recent_metrics[-10:]]
        older_mse = [m['metrics'].get('mse', 0) for m in recent_metrics[-20:-10]]
        
        if recent_mse and older_mse:
            recent_avg = statistics.mean(recent_mse)
            older_avg = statistics.mean(older_mse)
            
            # Significant regression
            if recent_avg > older_avg * 1.2:
                self._trigger_alert({
                    'type': 'regression',
                    'severity': 'warning',
                    'message': f"Performance regression detected: MSE increased {((recent_avg/older_avg - 1) * 100):.1f}%",
                    'metric': 'mse_regression',
                    'value': recent_avg / older_avg
                })
    
    def _get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource utilization statistics."""
        
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent
                },
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'timestamp': time.time()
            }
        
        except ImportError:
            return {
                'status': 'psutil_not_available',
                'timestamp': time.time()
            }
    
    def _setup_logging(self):
        """Setup persistent logging."""
        
        log_filename = f"vq_monitoring_{int(time.time())}.jsonl"
        self.current_log_file = self.log_directory / log_filename
        
        logger.info(f"VQ monitoring logs: {self.current_log_file}")
    
    def _log_to_file(self, metrics: Dict[str, Any]):
        """Log metrics to file."""
        
        if not self.current_log_file:
            return
        
        try:
            log_entry = {
                'timestamp': time.time(),
                'metrics': metrics
            }
            
            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging to file: {e}")
    
    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        
        # This would typically integrate with an alerting system
        return {
            'total_alerts_today': 0,
            'critical_alerts': 0,
            'warning_alerts': 0,
            'info_alerts': 0,
            'last_alert_time': None
        }
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export collected metrics in specified format."""
        
        report = self.get_monitoring_report()
        
        if format == "json":
            return json.dumps(report, indent=2, default=str)
        elif format == "dict":
            return report
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Factory functions
def create_vq_monitor(enable_real_time: bool = True,
                     enable_logging: bool = True,
                     enable_alerts: bool = True,
                     **kwargs) -> VQMonitor:
    """
    Create a VQ monitor with specified configuration.
    
    Args:
        enable_real_time: Enable real-time monitoring
        enable_logging: Enable persistent logging
        enable_alerts: Enable alerting
        **kwargs: Additional configuration
        
    Returns:
        VQMonitor instance
    """
    
    config = MonitoringConfig(
        enable_real_time_monitoring=enable_real_time,
        enable_persistent_logging=enable_logging,
        enable_alerts=enable_alerts,
        **kwargs
    )
    
    return VQMonitor(config)


def create_production_monitor() -> VQMonitor:
    """Creates production-ready VQ monitor with optimized settings."""
    
    config = MonitoringConfig(
        enable_real_time_monitoring=True,
        enable_persistent_logging=True,
        enable_alerts=True,
        performance_threshold=0.9,
        memory_threshold=0.8,
        latency_threshold_ms=50.0,
        metrics_collection_interval=0.5,
        enable_trend_analysis=True,
        anomaly_detection_sensitivity=2.5
    )
    
    return VQMonitor(config)


# Global monitor instance
_global_monitor = None

def get_global_monitor() -> Optional[VQMonitor]:
    """Get global VQ monitor instance."""
    return _global_monitor

def set_global_monitor(monitor: VQMonitor):
    """Set global VQ monitor instance."""
    global _global_monitor
    _global_monitor = monitor

def monitor_vq_operation(vq_instance: Any, 
                        inputs: Any, 
                        outputs: Tuple[Any, Any, Dict[str, Any]],
                        operation_time: float):
    """Convenience function to monitor VQ operation using global monitor."""
    
    if _global_monitor:
        _global_monitor.record_vq_operation(vq_instance, inputs, outputs, operation_time)


logger.info("VQ Monitoring system initialized")

def main():
    # Main function for this module.
    logger.info("Module vq_monitoring.py starting")
    return True

if __name__ == "__main__":
    main()
