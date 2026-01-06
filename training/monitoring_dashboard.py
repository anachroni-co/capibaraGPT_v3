"""
Real-time Monitoring Dashboard and Analytics for Meta-Consensus System

This module provides comprehensive monitoring, analytics, and visualization capabilities:
- Real-time system metrics dashboard
- Expert performance analytics
- Cost and quality tracking
- Alert system for anomalies
- Historical trend analysis
- Interactive web dashboard
- API endpoints for external monitoring
"""

import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics being tracked."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    EXPERT_UTILIZATION = "expert_utilization"
    SYSTEM_HEALTH = "system_health"
    USER_SATISFACTION = "user_satisfaction"

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """System alert."""
    id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class DashboardConfig:
    """Configurestion for monitoring dashboard."""
    host: str = "0.0.0.0"
    port: int = 8080
    max_data_points: int = 1000
    refresh_interval_seconds: int = 5
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    enable_alerts: bool = True
    log_to_file: bool = True
    dashboard_title: str = "Meta-Consensus Monitoring Dashboard"

class MetricsCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.max_data_points)
        )
        self.alerts: List[Alert] = []
        self.alert_counter = 0
        
        # Default alert thresholds
        self.default_thresholds = {
            "response_time_ms": {"warning": 3000, "error": 5000, "critical": 10000},
            "quality_score": {"warning": 7.0, "error": 6.0, "critical": 5.0},
            "cost_per_query": {"warning": 0.05, "error": 0.10, "critical": 0.20},
            "success_rate": {"warning": 0.90, "error": 0.80, "critical": 0.70},
            "expert_failure_rate": {"warning": 0.10, "error": 0.20, "critical": 0.30}
        }
        
        # Update with custom thresholds
        for metric, thresholds in config.alert_thresholds.items():
            self.default_thresholds[metric].update(thresholds)
    
    def add_metric(self, metric_name: str, value: float, 
                   metric_type: MetricType = MetricType.PERFORMANCE,
                   metadata: Optional[Dict[str, Any]] = None):
        """Add a new metric data point."""
        
        point = MetricPoint(
            timestamp=datetime.now(),
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )
        
        self.metrics_data[metric_name].append(point)
        
        # Check for alerts
        if self.config.enable_alerts:
            self._check_alert_conditions(metric_name, value)
    
    def _check_alert_conditions(self, metric_name: str, value: float):
        """Check if metric value triggers any alerts."""
        
        if metric_name not in self.default_thresholds:
            return
        
        thresholds = self.default_thresholds[metric_name]
        alert_level = None
        threshold_value = None
        
        # Check thresholds in order of severity
        if "critical" in thresholds and self._exceeds_threshold(metric_name, value, thresholds["critical"]):
            alert_level = AlertLevel.CRITICAL
            threshold_value = thresholds["critical"]
        elif "error" in thresholds and self._exceeds_threshold(metric_name, value, thresholds["error"]):
            alert_level = AlertLevel.ERROR
            threshold_value = thresholds["error"]
        elif "warning" in thresholds and self._exceeds_threshold(metric_name, value, thresholds["warning"]):
            alert_level = AlertLevel.WARNING
            threshold_value = thresholds["warning"]
        
        if alert_level:
            self._create_alert(metric_name, value, threshold_value, alert_level)
    
    def _exceeds_threshold(self, metric_name: str, value: float, threshold: float) -> bool:
        """Check if value exceeds threshold based on metric type."""
        
        # Metrics where lower is better
        lower_is_better = ["quality_score", "success_rate"]
        
        if any(metric in metric_name.lower() for metric in lower_is_better):
            return value < threshold  # Alert if below threshold
        else:
            return value > threshold  # Alert if above threshold
    
    def _create_alert(self, metric_name: str, value: float, threshold: float, level: AlertLevel):
        """Creates a new alert."""
        
        self.alert_counter += 1
        alert = Alert(
            id=f"alert_{self.alert_counter}",
            timestamp=datetime.now(),
            level=level,
            title=f"{metric_name.replace('_', ' ').title()} {level.value.title()}",
            message=f"{metric_name} value {value:.2f} {'below' if 'quality' in metric_name or 'success' in metric_name else 'above'} {level.value} threshold {threshold:.2f}",
            metric_name=metric_name,
            current_value=value,
            threshold_value=threshold
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert.title} - {alert.message}")
    
    def get_recent_metrics(self, metric_name: str, hours: int = 1) -> List[MetricPoint]:
        """Get recent metrics for a specific metric."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            point for point in self.metrics_data[metric_name]
            if point.timestamp >= cutoff_time
        ]
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        
        if metric_name not in self.metrics_data or not self.metrics_data[metric_name]:
            return {}
        
        values = [point.value for point in self.metrics_data[metric_name]]
        
        return {
            "current": values[-1] if values else 0.0,
            "avg": np.mean(values),
            "min": np.min(values),
            "max": np.max(values),
            "std": np.std(values),
            "count": len(values)
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert by ID."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                logger.info(f"Alert resolved: {alert.title}")
                break

class DashboardGenerator:
    """Generatess dashboard visualizations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def create_performance_chart(self, hours: int = 24) -> str:
        """Creates performance metrics chart."""
        
        fig = go.Figure()
        
        performance_metrics = [
            "response_time_ms",
            "queries_per_second", 
            "success_rate"
        ]
        
        for metric in performance_metrics:
            data_points = self.metrics_collector.get_recent_metrics(metric, hours)
            if data_points:
                timestamps = [point.timestamp for point in data_points]
                values = [point.value for point in data_points]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_quality_chart(self, hours: int = 24) -> str:
        """Creates quality metrics chart."""
        
        fig = go.Figure()
        
        quality_metrics = [
            "avg_quality_score",
            "consensus_confidence",
            "expert_agreement"
        ]
        
        for metric in quality_metrics:
            data_points = self.metrics_collector.get_recent_metrics(metric, hours)
            if data_points:
                timestamps = [point.timestamp for point in data_points]
                values = [point.value for point in data_points]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Quality Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis=dict(range=[0, 10]),
            hovermode='x unified',
            height=400
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_cost_chart(self, hours: int = 24) -> str:
        """Creates cost analysis chart."""
        
        fig = go.Figure()
        
        cost_data = self.metrics_collector.get_recent_metrics("cost_per_query", hours)
        if cost_data:
            timestamps = [point.timestamp for point in cost_data]
            values = [point.value for point in cost_data]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name='Cost per Query',
                line=dict(color='red', width=2)
            ))
        
        # Add cumulative cost
        cumulative_data = self.metrics_collector.get_recent_metrics("cumulative_cost", hours)
        if cumulative_data:
            timestamps = [point.timestamp for point in cumulative_data]
            values = [point.value for point in cumulative_data]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name='Cumulative Cost',
                yaxis='y2',
                line=dict(color='orange', width=2)
            ))
        
        fig.update_layout(
            title="Cost Analysis",
            xaxis_title="Time",
            yaxis_title="Cost per Query ($)",
            yaxis2=dict(
                title="Cumulative Cost ($)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_expert_utilization_chart(self) -> str:
        """Creates expert utilization pie chart."""
        
        # Get expert usage data from recent metrics
        expert_data = defaultdict(int)
        
        for metric_name, data_points in self.metrics_collector.metrics_data.items():
            if "expert_usage_" in metric_name:
                expert_name = metric_name.replace("expert_usage_", "")
                if data_points:
                    expert_data[expert_name] = data_points[-1].value
        
        if expert_data:
            fig = go.Figure(data=[go.Pie(
                labels=list(expert_data.keys()),
                values=list(expert_data.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title="Expert Utilization Distribution",
                height=400
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                title="Expert Utilization Distribution",
                annotations=[dict(text="No data available", showarrow=False)],
                height=400
            )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_alerts_summary(self) -> Dict[str, Any]:
        """Creates alerts summary."""
        
        active_alerts = self.metrics_collector.get_active_alerts()
        
        alert_counts = {
            "total": len(active_alerts),
            "critical": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            "error": len([a for a in active_alerts if a.level == AlertLevel.ERROR]),
            "warning": len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
            "info": len([a for a in active_alerts if a.level == AlertLevel.INFO])
        }
        
        recent_alerts = sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
        
        return {
            "counts": alert_counts,
            "recent_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name
                }
                for alert in recent_alerts
            ]
        }

class MonitoringDashboard:
    """Main monitoring dashboard application."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.dashboard_generator = DashboardGenerator(self.metrics_collector)
        
        # FastAPI app
        self.app = FastAPI(title=config.dashboard_title)
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Monitoring Dashboard initialized on {config.host}:{config.port}")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._get_dashboard_html()
        
        @self.app.get("/api/metrics/{metric_name}")
        async def get_metric(metric_name: str, hours: int = 1):
            data_points = self.metrics_collector.get_recent_metrics(metric_name, hours)
            return {
                "metric_name": metric_name,
                "data_points": [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "metadata": point.metadata
                    }
                    for point in data_points
                ]
            }
        
        @self.app.get("/api/metrics/{metric_name}/summary")
        async def get_metric_summary(metric_name: str):
            return self.metrics_collector.get_metric_summary(metric_name)
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            return {
                "active_alerts": [asdict(alert) for alert in self.metrics_collector.get_active_alerts()],
                "summary": self.dashboard_generator.create_alerts_summary()
            }
        
        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            self.metrics_collector.resolve_alert(alert_id)
            return {"status": "resolved"}
        
        @self.app.get("/api/dashboard/performance")
        async def get_performance_chart(hours: int = 24):
            return {"chart_data": self.dashboard_generator.create_performance_chart(hours)}
        
        @self.app.get("/api/dashboard/quality")
        async def get_quality_chart(hours: int = 24):
            return {"chart_data": self.dashboard_generator.create_quality_chart(hours)}
        
        @self.app.get("/api/dashboard/cost")
        async def get_cost_chart(hours: int = 24):
            return {"chart_data": self.dashboard_generator.create_cost_chart(hours)}
        
        @self.app.get("/api/dashboard/experts")
        async def get_expert_chart():
            return {"chart_data": self.dashboard_generator.create_expert_utilization_chart()}
        
        @self.app.websocket("/ws/metrics")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send real-time updates every few seconds
                    await asyncio.sleep(self.config.refresh_interval_seconds)
                    
                    update_data = {
                        "timestamp": datetime.now().isoformat(),
                        "alerts": self.dashboard_generator.create_alerts_summary(),
                        "metrics": {
                            "response_time": self.metrics_collector.get_metric_summary("response_time_ms"),
                            "quality_score": self.metrics_collector.get_metric_summary("avg_quality_score"),
                            "cost": self.metrics_collector.get_metric_summary("cost_per_query")
                        }
                    }
                    
                    await websocket.send_json(update_data)
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.remove(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Generates dashboard HTML."""
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Meta-Consensus Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
                .alert-critical { background-color: #ffebee; border-left: 4px solid #f44336; }
                .alert-error { background-color: #fff3e0; border-left: 4px solid #ff9800; }
                .alert-warning { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                .status-good { background-color: #4caf50; }
                .status-warning { background-color: #ff9800; }
                .status-error { background-color: #f44336; }
                .refresh-button { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎭 Meta-Consensus Monitoring Dashboard</h1>
                <p>Real-time system monitoring and analytics</p>
                <button class="refresh-button" onclick="refreshDashboard()">🔄 Refresh</button>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>System Status</h3>
                    <div id="system-status">
                        <span class="status-indicator status-good"></span>
                        <span>System Operational</span>
                    </div>
                    <p>Uptime: <span id="uptime">--</span></p>
                    <p>Active Alerts: <span id="active-alerts">0</span></p>
                </div>
                
                <div class="metric-card">
                    <h3>Performance</h3>
                    <p>Avg Response Time: <span id="avg-response-time">--</span>ms</p>
                    <p>Queries/Second: <span id="queries-per-second">--</span></p>
                    <p>Success Rate: <span id="success-rate">--</span>%</p>
                </div>
                
                <div class="metric-card">
                    <h3>Quality</h3>
                    <p>Avg Quality Score: <span id="avg-quality">--</span>/10</p>
                    <p>Consensus Confidence: <span id="consensus-confidence">--</span>%</p>
                    <p>Expert Agreement: <span id="expert-agreement">--</span>%</p>
                </div>
                
                <div class="metric-card">
                    <h3>Cost Analysis</h3>
                    <p>Avg Cost/Query: $<span id="avg-cost">--</span></p>
                    <p>Total Cost Today: $<span id="total-cost">--</span></p>
                    <p>Cost Savings: <span id="cost-savings">--</span>%</p>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="performance-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="quality-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="cost-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="expert-chart"></div>
            </div>
            
            <div class="metric-card">
                <h3>Recent Alerts</h3>
                <div id="alerts-container">
                    <p>No active alerts</p>
                </div>
            </div>
            
            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket(`ws://${window.location.host}/ws/metrics`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    // Update metrics
                    if (data.metrics) {
                        if (data.metrics.response_time && data.metrics.response_time.current) {
                            document.getElementById('avg-response-time').textContent = Math.round(data.metrics.response_time.current);
                        }
                        if (data.metrics.quality_score && data.metrics.quality_score.current) {
                            document.getElementById('avg-quality').textContent = data.metrics.quality_score.current.toFixed(1);
                        }
                        if (data.metrics.cost && data.metrics.cost.current) {
                            document.getElementById('avg-cost').textContent = data.metrics.cost.current.toFixed(4);
                        }
                    }
                    
                    // Update alerts
                    if (data.alerts) {
                        document.getElementById('active-alerts').textContent = data.alerts.counts.total;
                        updateAlertsDisplay(data.alerts.recent_alerts);
                    }
                }
                
                function updateAlertsDisplay(alerts) {
                    const container = document.getElementById('alerts-container');
                    if (alerts.length === 0) {
                        container.innerHTML = '<p>No active alerts</p>';
                        return;
                    }
                    
                    container.innerHTML = alerts.map(alert => `
                        <div class="alert alert-${alert.level}">
                            <strong>${alert.title}</strong><br>
                            ${alert.message}<br>
                            <small>${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
                
                async function refreshDashboard() {
                    try {
                        // Fetch and update charts
                        const [performance, quality, cost, experts] = await Promise.all([
                            fetch('/api/dashboard/performance').then(r => r.json()),
                            fetch('/api/dashboard/quality').then(r => r.json()),
                            fetch('/api/dashboard/cost').then(r => r.json()),
                            fetch('/api/dashboard/experts').then(r => r.json())
                        ]);
                        
                        Plotly.newPlot('performance-chart', JSON.parse(performance.chart_data));
                        Plotly.newPlot('quality-chart', JSON.parse(quality.chart_data));
                        Plotly.newPlot('cost-chart', JSON.parse(cost.chart_data));
                        Plotly.newPlot('expert-chart', JSON.parse(experts.chart_data));
                        
                    } catch (error) {
                        console.error('Error refreshing dashboard:', error);
                    }
                }
                
                // Initial load
                refreshDashboard();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshDashboard, 30000);
            </script>
        </body>
        </html>
        """
    
    async def start_server(self):
        """Start the monitoring dashboard server."""
        
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def add_metric(self, metric_name: str, value: float, 
                   metric_type: MetricType = MetricType.PERFORMANCE,
                   metadata: Optional[Dict[str, Any]] = None):
        """Add a metric to the dashboard."""
        self.metrics_collector.add_metric(metric_name, value, metric_type, metadata)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        
        summary = {}
        for metric_name in self.metrics_collector.metrics_data.keys():
            summary[metric_name] = self.metrics_collector.get_metric_summary(metric_name)
        
        return {
            "metrics": summary,
            "alerts": self.dashboard_generator.create_alerts_summary(),
            "system_info": {
                "uptime_seconds": (datetime.now() - datetime.now()).total_seconds(),  # Placeholder
                "total_metrics": len(self.metrics_collector.metrics_data),
                "total_alerts": len(self.metrics_collector.alerts)
            }
        }

class MetaConsensusMonitor:
    """Integration class for monitoring Meta-Consensus System."""
    
    def __init__(self, dashboard_config: Optional[DashboardConfig] = None):
        self.config = dashboard_config or DashboardConfig()
        self.dashboard = MonitoringDashboard(self.config)
        self.start_time = datetime.now()
        
    async def monitor_consensus_query(self, query_id: str, result: Dict[str, Any]):
        """Monitor a consensus query result."""
        
        # Extract metrics from consensus result
        response_time = result.get('response_time_ms', 0)
        quality_score = result.get('quality_score', 0)
        cost = result.get('total_cost', 0)
        confidence = result.get('confidence', 0)
        
        # Add metrics to dashboard
        self.dashboard.add_metric("response_time_ms", response_time, MetricType.PERFORMANCE)
        self.dashboard.add_metric("avg_quality_score", quality_score, MetricType.QUALITY)
        self.dashboard.add_metric("cost_per_query", cost, MetricType.COST)
        self.dashboard.add_metric("consensus_confidence", confidence * 100, MetricType.QUALITY)
        
        # Track expert usage
        participating_experts = result.get('participating_experts', [])
        for expert in participating_experts:
            self.dashboard.add_metric(f"expert_usage_{expert}", 1, MetricType.EXPERT_UTILIZATION)
        
        # Calculate derived metrics
        success = 1 if quality_score > 6.0 else 0
        self.dashboard.add_metric("success_rate", success, MetricType.PERFORMANCE)
        
        # Update cumulative cost
        current_cumulative = self.dashboard.metrics_collector.get_metric_summary("cumulative_cost").get("current", 0)
        self.dashboard.add_metric("cumulative_cost", current_cumulative + cost, MetricType.COST)
    
    async def monitor_system_health(self, system_status: Dict[str, Any]):
        """Monitor overall system health."""
        
        # System performance metrics
        cpu_usage = system_status.get('cpu_percent', 0)
        memory_usage = system_status.get('memory_percent', 0)
        active_connections = system_status.get('active_connections', 0)
        
        self.dashboard.add_metric("cpu_usage", cpu_usage, MetricType.SYSTEM_HEALTH)
        self.dashboard.add_metric("memory_usage", memory_usage, MetricType.SYSTEM_HEALTH)
        self.dashboard.add_metric("active_connections", active_connections, MetricType.SYSTEM_HEALTH)
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        self.dashboard.add_metric("uptime_seconds", uptime_seconds, MetricType.SYSTEM_HEALTH)
    
    async def start_monitoring(self):
        """Start the monitoring dashboard."""
        logger.info("Starting Meta-Consensus monitoring dashboard...")
        await self.dashboard.start_server()
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.config.host}:{self.config.port}"


# Utility functions
def create_default_dashboard_config(**kwargs) -> DashboardConfig:
    """Creates default dashboard configuration."""
    
    default_thresholds = {
        "response_time_ms": {"warning": 3000, "error": 5000, "critical": 10000},
        "avg_quality_score": {"warning": 7.0, "error": 6.0, "critical": 5.0},
        "cost_per_query": {"warning": 0.05, "error": 0.10, "critical": 0.20},
        "success_rate": {"warning": 0.90, "error": 0.80, "critical": 0.70}
    }
    
    return DashboardConfig(
        alert_thresholds=default_thresholds,
        **kwargs
    )


# Export main components
__all__ = [
    'MonitoringDashboard',
    'MetaConsensusMonitor',
    'DashboardConfig',
    'MetricsCollector',
    'DashboardGenerator',
    'MetricType',
    'AlertLevel',
    'create_default_dashboard_config'
]


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create monitoring system
        config = create_default_dashboard_config(
            host="0.0.0.0",
            port=8080,
            dashboard_title="Meta-Consensus Monitor"
        )
        
        monitor = MetaConsensusMonitor(config)
        
        # Simulate some metrics
        sample_results = [
            {
                "query_id": "test_1",
                "response_time_ms": 1500,
                "quality_score": 8.5,
                "total_cost": 0.008,
                "confidence": 0.85,
                "participating_experts": ["math_expert", "general_expert"]
            },
            {
                "query_id": "test_2", 
                "response_time_ms": 2200,
                "quality_score": 9.1,
                "total_cost": 0.015,
                "confidence": 0.92,
                "participating_experts": ["premium_expert", "specialized_expert"]
            }
        ]
        
        # Add sample data
        for result in sample_results:
            await monitor.monitor_consensus_query(result["query_id"], result)
        
        # Add system health data
        await monitor.monitor_system_health({
            "cpu_percent": 45.2,
            "memory_percent": 67.8,
            "active_connections": 12
        })
        
        print(f"Dashboard available at: {monitor.get_dashboard_url()}")
        print("Starting monitoring dashboard server...")
        
        # Start dashboard (this will run indefinitely)
        await monitor.start_monitoring()
    
    asyncio.run(main())