"""
MCP (Model Context Protocol) integration for CapibaraGPT.

This implementation is an in-process message bus intended for local
testing and coordination. It does not provide a real network transport.
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
from enum import Enum

logger = logging.getLogger(__name__)

class MCPMessageType(Enum):
    """MCP message types."""
    HANDSHAKE = "handshake"
    CONTEXT_SYNC = "context_sync"
    TRAINING_UPDATE = "training_update"
    MODEL_STATE = "model_state"
    PERFORMANCE_REPORT = "performance_report"
    CONTROL_COMMAND = "control_command"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

@dataclass
class MCPMessage:
    """Standard MCP protocol message."""

    message_type: MCPMessageType
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary."""
        # Convert message_type string back to enum
        if isinstance(data.get('message_type'), str):
            data['message_type'] = MCPMessageType(data['message_type'])
        return cls(**data)

@dataclass
class MCPNode:
    """Node in the MCP network."""

    node_id: str
    node_type: str  # 'trainer', 'inference', 'coordinator', 'monitor'
    capabilities: List[str] = field(default_factory=list)
    status: str = "online"  # online, offline, busy, error
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPMessageHandler:
    """Base handler for MCP messages."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.handlers: Dict[MCPMessageType, List[Callable]] = {}

    def register_handler(self, message_type: MCPMessageType, handler: Callable):
        """Register a handler for a message type."""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        self.handlers[message_type].append(handler)
        logger.info(f" Registered handler for {message_type.value}")
    
    async def handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Handle an incoming message."""
        logger.debug(f" Handling message: {message.message_type.value} from {message.sender_id}")
        
        handlers = self.handlers.get(message.message_type, [])
        response = None
        
        for handler in handlers:
            try:
                result = await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
                if result and isinstance(result, MCPMessage):
                    response = result
            except Exception as e:
                logger.error(f" Handler error: {e}")
                response = MCPMessage(
                    message_type=MCPMessageType.ERROR,
                    sender_id=self.node_id,
                    recipient_id=message.sender_id,
                    payload={'error': str(e), 'original_message_id': message.message_id}
                )
        
        return response

class MCPClient:
    """Cliente MCP para comunicación con otros nodos."""
    
    def __init__(self, node_id: str, node_type: str = "trainer"):
        self.node_id = node_id
        self.node_type = node_type
        self.message_handler = MCPMessageHandler(node_id)
        self.connected_nodes: Dict[str, MCPNode] = {}
        self.message_queue: List[MCPMessage] = []
        self.running = False
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_start_time = time.time()
        
        logger.info(f" MCP Client initialized: {node_id} ({node_type})")
    
    async def start(self):
        """Start the MCP client."""
        self.running = True
        logger.info(f" MCP Client {self.node_id} started")

        # Send initial handshake
        await self.send_handshake()

        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop the MCP client."""
        self.running = False
        logger.info(f" MCP Client {self.node_id} stopped")

    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message."""
        try:
            # In real implementation, this would be sent over network
            # In-process queue (no network transport)
            self.message_queue.append(message)
            self.messages_sent += 1
            
            logger.debug(f" Sent message: {message.message_type.value} to {message.recipient_id or 'broadcast'}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to send message: {e}")
            return False
    
    async def send_handshake(self):
        """Send handshake message."""
        handshake_message = MCPMessage(
            message_type=MCPMessageType.HANDSHAKE,
            sender_id=self.node_id,
            payload={
                'node_type': self.node_type,
                'capabilities': ['training', 'inference', 'monitoring'],
                'version': '1.0.0',
                'status': 'online'
            }
        )
        await self.send_message(handshake_message)
    
    async def send_training_update(self, training_metrics: Dict[str, Any]):
        """Send training update."""
        update_message = MCPMessage(
            message_type=MCPMessageType.TRAINING_UPDATE,
            sender_id=self.node_id,
            payload={
                'metrics': training_metrics,
                'timestamp': datetime.now().isoformat(),
                'step': training_metrics.get('step', 0)
            },
            priority=2
        )
        await self.send_message(update_message)
    
    async def send_performance_report(self, performance_data: Dict[str, Any]):
        """Send performance report."""
        report_message = MCPMessage(
            message_type=MCPMessageType.PERFORMANCE_REPORT,
            sender_id=self.node_id,
            payload=performance_data,
            priority=2
        )
        await self.send_message(report_message)
    
    async def send_context_sync(self, context_data: Dict[str, Any]):
        """Send context synchronization."""
        sync_message = MCPMessage(
            message_type=MCPMessageType.CONTEXT_SYNC,
            sender_id=self.node_id,
            payload=context_data,
            priority=3
        )
        await self.send_message(sync_message)
    
    async def _heartbeat_loop(self):
        """Heartbeat loop to maintain connection."""
        while self.running:
            try:
                heartbeat_message = MCPMessage(
                    message_type=MCPMessageType.HEARTBEAT,
                    sender_id=self.node_id,
                    payload={
                        'status': 'online',
                        'uptime': time.time() - self.connection_start_time,
                        'messages_sent': self.messages_sent,
                        'messages_received': self.messages_received
                    }
                )
                await self.send_message(heartbeat_message)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f" Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    def register_handler(self, message_type: MCPMessageType, handler: Callable):
        """Register a message handler."""
        self.message_handler.register_handler(message_type, handler)

    def get_status(self) -> Dict[str, Any]:
        """Get the MCP client state."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'running': self.running,
            'connected_nodes': len(self.connected_nodes),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'uptime': time.time() - self.connection_start_time,
            'queue_size': len(self.message_queue)
        }

class MCPCoordinator:
    """MCP Coordinator to manage multiple nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, MCPNode] = {}
        self.message_history: List[MCPMessage] = []
        self.running = False
        
        logger.info(" MCP Coordinator initialized")
    
    async def start(self):
        """Start the coordinator."""
        self.running = True
        logger.info(" MCP Coordinator started")

        # Start maintenance tasks
        asyncio.create_task(self._maintenance_loop())

    async def stop(self):
        """Stop the coordinator."""
        self.running = False
        logger.info(" MCP Coordinator stopped")

    def register_node(self, node: MCPNode):
        """Register a new node."""
        self.nodes[node.node_id] = node
        logger.info(f" Node registered: {node.node_id} ({node.node_type})")

    def unregister_node(self, node_id: str):
        """Unregister a node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f" Node unregistered: {node_id}")

    async def broadcast_message(self, message: MCPMessage):
        """Send message to all nodes."""
        for node_id in self.nodes:
            message.recipient_id = node_id
            # In real implementation, send over network
            logger.debug(f" Broadcasting to {node_id}: {message.message_type.value}")

    async def route_message(self, message: MCPMessage):
        """Route a message to the appropriate recipient."""
        if message.recipient_id:
            if message.recipient_id in self.nodes:
                # Send to specific node
                logger.debug(f" Routing to {message.recipient_id}: {message.message_type.value}")
            else:
                logger.warning(f"WARNING: Unknown recipient: {message.recipient_id}")
        else:
            # Broadcast
            await self.broadcast_message(message)

    async def _maintenance_loop(self):
        """Coordinator maintenance loop."""
        while self.running:
            try:
                # Clean up inactive nodes
                current_time = datetime.now()
                inactive_nodes = []

                for node_id, node in self.nodes.items():
                    last_seen = datetime.fromisoformat(node.last_seen)
                    if (current_time - last_seen).total_seconds() > 300:  # 5 minutes
                        inactive_nodes.append(node_id)

                for node_id in inactive_nodes:
                    logger.warning(f"WARNING: Removing inactive node: {node_id}")
                    self.unregister_node(node_id)

                # Clean up old message history
                if len(self.message_history) > 1000:
                    self.message_history = self.message_history[-500:]

                await asyncio.sleep(60)  # Maintenance every minute

            except Exception as e:
                logger.error(f" Maintenance error: {e}")
                await asyncio.sleep(10)

    def get_network_status(self) -> Dict[str, Any]:
        """Get the MCP network state."""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        return {
            'total_nodes': len(self.nodes),
            'node_types': node_types,
            'message_history_size': len(self.message_history),
            'coordinator_running': self.running,
            'nodes': {node_id: {'type': node.node_type, 'status': node.status} 
                     for node_id, node in self.nodes.items()}
        }

class MCPIntegration:
    """
    Main MCP Integration for CapibaraGPT.

    This class provides a unified interface for all MCP functionality.
    """
    
    def __init__(self, node_id: Optional[str] = None, node_type: str = "trainer"):
        self.node_id = node_id or f"capibara_{int(time.time())}"
        self.node_type = node_type
        self.client = MCPClient(self.node_id, node_type)
        self.coordinator = None  # Only if acting as coordinator
        
        # Callbacks for external integration
        self.training_update_callback: Optional[Callable] = None
        self.performance_callback: Optional[Callable] = None
        self.context_sync_callback: Optional[Callable] = None
        
        # Register default handlers
        self._register_default_handlers()
        
        logger.info(f" MCP Integration initialized: {self.node_id}")
    
    def _register_default_handlers(self):
        """Register default handlers."""
        
        async def handle_handshake(message: MCPMessage) -> MCPMessage:
            """Handle handshake messages."""
            logger.info(f" Handshake from {message.sender_id}")
            return MCPMessage(
                message_type=MCPMessageType.HANDSHAKE,
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                payload={
                    'node_type': self.node_type,
                    'status': 'online',
                    'response_to': message.message_id
                }
            )
        
        async def handle_training_update(message: MCPMessage):
            """Handle training updates."""
            logger.info(f" Training update from {message.sender_id}")
            if self.training_update_callback:
                await self.training_update_callback(message.payload)
        
        async def handle_performance_report(message: MCPMessage):
            """Handle performance reports."""
            logger.info(f" Performance report from {message.sender_id}")
            if self.performance_callback:
                await self.performance_callback(message.payload)
        
        # Register handlers
        self.client.register_handler(MCPMessageType.HANDSHAKE, handle_handshake)
        self.client.register_handler(MCPMessageType.TRAINING_UPDATE, handle_training_update)
        self.client.register_handler(MCPMessageType.PERFORMANCE_REPORT, handle_performance_report)
    
    async def start(self, as_coordinator: bool = False):
        """Inicia la integración MCP."""
        await self.client.start()
        
        if as_coordinator:
            self.coordinator = MCPCoordinator()
            await self.coordinator.start()
            logger.info(" Started as MCP Coordinator")
    
    async def stop(self):
        """Detiene la integración MCP."""
        await self.client.stop()
        if self.coordinator:
            await self.coordinator.stop()
    
    async def report_training_progress(self, metrics: Dict[str, Any]):
        """Reporta progreso de entrenamiento."""
        await self.client.send_training_update(metrics)
    
    async def report_performance(self, performance_data: Dict[str, Any]):
        """Reporta datos de rendimiento."""
        await self.client.send_performance_report(performance_data)
    
    async def sync_context(self, context_data: Dict[str, Any]):
        """Sincroniza contexto con otros nodos."""
        await self.client.send_context_sync(context_data)
    
    def set_callbacks(self,
                     training_callback: Optional[Callable] = None,
                     performance_callback: Optional[Callable] = None,
                     context_callback: Optional[Callable] = None):
        """Establishes callbacks para eventos MCP."""
        self.training_update_callback = training_callback
        self.performance_callback = performance_callback
        self.context_sync_callback = context_callback
    
    def get_status(self) -> Dict[str, Any]:
        """Gets the state completo de MCP."""
        status = {
            'client': self.client.get_status(),
            'node_id': self.node_id,
            'node_type': self.node_type
        }
        
        if self.coordinator:
            status['coordinator'] = self.coordinator.get_network_status()
        
        return status

# Factory functions
def create_mcp_integration(node_id: Optional[str] = None, 
                          node_type: str = "trainer") -> MCPIntegration:
    """Create an MCP integration instance."""
    return MCPIntegration(node_id, node_type)

def create_mcp_coordinator() -> MCPCoordinator:
    """Create an MCP coordinator."""
    return MCPCoordinator()

# Global instance
_global_mcp: Optional[MCPIntegration] = None

def get_global_mcp() -> MCPIntegration:
    """Gets la instancia global de MCP."""
    global _global_mcp
    if _global_mcp is None:
        _global_mcp = create_mcp_integration()
    return _global_mcp

async def main():
    """Main function for testing."""
    logger.info(" MCP Integration - Testing Mode")
    
    # Create MCP integration
    mcp = create_mcp_integration("test_node", "trainer")
    await mcp.start()
    
    # Simular algunos eventos
    await mcp.report_training_progress({
        'step': 100,
        'loss': 0.5,
        'accuracy': 0.85
    })
    
    await mcp.report_performance({
        'throughput': 1000,
        'latency': 0.1,
        'memory_usage': 0.7
    })
    
    # Show state
    status = mcp.get_status()
    logger.info(f"MCP Status: {status}")
    
    await asyncio.sleep(2)
    await mcp.stop()

if __name__ == "__main__":
    asyncio.run(main())
