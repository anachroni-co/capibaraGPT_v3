"""
Federated Consensus System for Multi-Node Meta-Consensus
Implements distributed consensus across multiple nodes with fault tolerance
"""

import logging
import asyncio
import time
import json
import hashlib
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Network imports with fallbacks
try:
    import aiohttp
    from aiohttp import web
    NETWORK_AVAILABLE = True
except ImportError:
    logger.warning("Network libraries not available - HTTP services disabled")
    NETWORK_AVAILABLE = False
    aiohttp = None
    web = None


class _TextEmbedder:
    """Lazy text embedder using Hugging Face transformers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except Exception as exc:
                raise RuntimeError(f"Transformers backend not available: {exc}") from exc

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            if torch.cuda.is_available():
                self._device = "cuda"
                self._model.to(self._device)

    def encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        import torch

        with torch.no_grad():
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                pooled = token_embeddings.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            return pooled.cpu().numpy().astype(np.float32)


class NodeRole(Enum):
    """Roles that nodes can take in the federated consensus."""
    COORDINATOR = "coordinator"      # Primary coordinator node
    PARTICIPANT = "participant"      # Regular consensus participant
    OBSERVER = "observer"           # Read-only observer
    VALIDATOR = "validator"         # Validation-only node
    BACKUP_COORDINATOR = "backup_coordinator"  # Backup coordinator


class ConsensusProtocol(Enum):
    """Consensus protocols supported."""
    RAFT = "raft"                  # Raft consensus algorithm
    PBFT = "pbft"                  # Practical Byzantine Fault Tolerance
    TENDERMINT = "tendermint"      # Tendermint consensus
    HYBRID = "hybrid"              # Hybrid protocol combining multiple approaches


class NodeStatus(Enum):
    """Status of federated nodes."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


@dataclass
class NodeInfo:
    """Information about a federated node."""
    node_id: str
    role: NodeRole
    status: NodeStatus
    endpoint: str
    capabilities: List[str]
    last_heartbeat: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    trust_score: float = 1.0
    version: str = "1.0.0"


@dataclass
class ConsensusProposal:
    """A consensus proposal from a node."""
    proposal_id: str
    node_id: str
    timestamp: datetime
    query_id: str
    expert_responses: List[Dict[str, Any]]
    confidence_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result of federated consensus."""
    consensus_id: str
    query_id: str
    final_response: Dict[str, Any]
    participating_nodes: List[str]
    consensus_confidence: float
    consensus_time: float
    agreement_level: float
    minority_opinions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FederatedConsensusConfig:
    """Configurestion for federated consensus system."""
    
    # Network configuration
    coordinator_endpoint: str = "ws://localhost:8765"
    node_endpoints: List[str] = field(default_factory=list)
    heartbeat_interval: int = 30  # seconds
    connection_timeout: int = 10  # seconds
    
    # Consensus configuration
    consensus_protocol: ConsensusProtocol = ConsensusProtocol.RAFT
    min_participants: int = 3
    consensus_threshold: float = 0.67  # Minimum agreement percentage
    max_consensus_time: int = 30  # seconds
    
    # Fault tolerance
    byzantine_fault_tolerance: bool = True
    max_byzantine_nodes: int = 1
    node_failure_timeout: int = 60  # seconds
    auto_recovery: bool = True
    
    # Performance optimization
    enable_caching: bool = True
    cache_size: int = 1000
    enable_compression: bool = True
    batch_size: int = 10
    
    # Security
    enable_encryption: bool = True
    require_signatures: bool = True
    trust_threshold: float = 0.7
    
    # Quality assurance
    enable_validation: bool = True
    validator_nodes: List[str] = field(default_factory=list)
    quality_threshold: float = 0.8


class FederatedConsensusNode:
    """A node in the federated consensus system."""
    
    def __init__(self, node_id: str, role: NodeRole, config: FederatedConsensusConfig):
        self.node_id = node_id
        self.role = role
        self.config = config
        self.status = NodeStatus.INITIALIZING
        
        # Node state
        self.peer_nodes: Dict[str, NodeInfo] = {}
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.consensus_history: List[ConsensusResult] = []
        
        # Performance tracking
        self.performance_metrics = {
            'proposals_processed': 0,
            'consensus_participated': 0,
            'average_response_time': 0.0,
            'success_rate': 1.0,
            'uptime': 0.0
        }
        
        # Network connections
        self.connections: Dict[str, Any] = {}
        self.websocket_server = None
        self.coordinator_connection = None
        self.http_runner = None
        self.http_site = None
        self._embedder = None
        
        # Consensus state
        self.current_term = 0
        self.voted_for = None
        self.log_entries = []
        self.commit_index = 0
        
        logger.info(f" Federated consensus node {node_id} initialized (role: {role.value})")

    def _get_embedder(self) -> _TextEmbedder:
        if self._embedder is None:
            self._embedder = _TextEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        return self._embedder

    async def _encode_texts(self, texts: List[str]) -> np.ndarray:
        embedder = self._get_embedder()
        return await asyncio.to_thread(embedder.encode, texts)
    
    async def start(self):
        """Start the federated consensus node."""
        logger.info(f" Starting federated consensus node {self.node_id}")
        
        try:
            # Start network services
            if NETWORK_AVAILABLE:
                await self._start_network_services()
            else:
                raise RuntimeError("Network services unavailable: aiohttp not installed")
            
            # Connect to coordinator (if not coordinator)
            if self.role != NodeRole.COORDINATOR:
                await self._connect_to_coordinator()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._consensus_loop())
            asyncio.create_task(self._maintenance_loop())
            
            self.status = NodeStatus.ACTIVE
            logger.info(f" Node {self.node_id} is now active")
            
        except Exception as e:
            logger.error(f" Failed to start node {self.node_id}: {e}")
            self.status = NodeStatus.FAILED
            raise
    
    async def propose_consensus(self, query_id: str, expert_responses: List[Dict[str, Any]], 
                              confidence_scores: List[float]) -> str:
        """Propose a new consensus for expert responses."""
        proposal_id = f"proposal_{self.node_id}_{int(time.time())}"
        
        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            node_id=self.node_id,
            timestamp=datetime.now(),
            query_id=query_id,
            expert_responses=expert_responses,
            confidence_scores=confidence_scores,
            metadata={
                'node_role': self.role.value,
                'protocol': self.config.consensus_protocol.value
            }
        )
        
        # Sign proposal if required
        if self.config.require_signatures:
            proposal.signature = self._sign_proposal(proposal)
        
        # Store proposal
        self.active_proposals[proposal_id] = proposal
        
        # Broadcast to network
        await self._broadcast_proposal(proposal)
        
        logger.info(f" Proposed consensus {proposal_id} for query {query_id}")
        return proposal_id
    
    async def participate_in_consensus(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """Participate in consensus for a received proposal."""
        logger.info(f" Participating in consensus {proposal.proposal_id}")
        
        # Validate proposal
        if not await self._validate_proposal(proposal):
            logger.warning(f" Invalid proposal {proposal.proposal_id}")
            return {'vote': 'reject', 'reason': 'invalid_proposal'}
        
        # Generate local consensus response
        local_response = await self._generate_local_consensus(
            proposal.expert_responses, 
            proposal.confidence_scores
        )
        
        # Calculate agreement with proposal
        agreement_score = await self._calculate_agreement(
            local_response, 
            proposal.expert_responses
        )
        
        # Determine vote
        vote = 'accept' if agreement_score >= self.config.consensus_threshold else 'reject'
        
        response = {
            'node_id': self.node_id,
            'proposal_id': proposal.proposal_id,
            'vote': vote,
            'agreement_score': agreement_score,
            'local_response': local_response,
            'timestamp': datetime.now().isoformat(),
            'confidence': np.mean(proposal.confidence_scores)
        }
        
        # Sign response if required
        if self.config.require_signatures:
            response['signature'] = self._sign_response(response)
        
        return response
    
    async def _start_network_services(self):
        """Start network services for the node."""
        if not NETWORK_AVAILABLE:
            return

        app = web.Application()
        app.add_routes([
            web.post("/proposal", self._http_handle_proposal),
            web.post("/consensus/participate", self._http_handle_consensus_participation),
            web.post("/heartbeat", self._http_handle_heartbeat),
            web.post("/register", self._http_handle_register),
        ])

        self.http_runner = web.AppRunner(app)
        await self.http_runner.setup()

        port = 8765
        if hasattr(self, "endpoint"):
            try:
                port = int(self.endpoint.split(":")[-1])
            except Exception as exc:
                logger.debug("Could not parse port from %r, using default %d: %s", self.endpoint, port, exc)

        self.http_site = web.TCPSite(self.http_runner, "0.0.0.0", port)
        await self.http_site.start()
        logger.info(f" HTTP server started on port {port}")
    
    async def _connect_to_coordinator(self):
        """Connect to the coordinator node."""
        try:
            # Send registration message
            registration = {
                'type': 'register',
                'node_id': self.node_id,
                'role': self.role.value,
                'capabilities': ['consensus', 'validation'],
                'endpoint': getattr(self, 'endpoint', 'http://localhost:8765')
            }
            await self._post_json(self.config.coordinator_endpoint.rstrip("/") + "/register", registration)
            logger.info(" Connected to coordinator")
            
        except Exception as e:
            logger.error(f" Failed to connect to coordinator: {e}")
            raise
    
    async def _process_network_message(self, data: Dict[str, Any]):
        """Process incoming network messages."""
        message_type = data.get('type')
        
        if message_type == 'proposal':
            proposal = ConsensusProposal(**data['proposal'])
            response = await self.participate_in_consensus(proposal)
            return {'type': 'consensus_response', 'response': response}
        
        elif message_type == 'heartbeat':
            return {
                'type': 'heartbeat_ack',
                'node_id': self.node_id,
                'status': self.status.value,
                'metrics': self.performance_metrics
            }
        
        elif message_type == 'register':
            node_info = NodeInfo(
                node_id=data['node_id'],
                role=NodeRole(data['role']),
                status=NodeStatus.ACTIVE,
                endpoint=data['endpoint'],
                capabilities=data['capabilities'],
                last_heartbeat=datetime.now()
            )
            self.peer_nodes[data['node_id']] = node_info
            logger.info(f" Registered peer node {data['node_id']}")
            return {'type': 'register_ack', 'node_id': self.node_id}

        return {'type': 'unknown'}

    async def _http_handle_proposal(self, request: web.Request) -> web.Response:
        data = await request.json()
        response = await self._process_network_message(data)
        return web.json_response(response)

    async def _http_handle_consensus_participation(self, request: web.Request) -> web.Response:
        data = await request.json()
        proposal = ConsensusProposal(**data['proposal'])
        response = await self.participate_in_consensus(proposal)
        return web.json_response(response)

    async def _http_handle_heartbeat(self, request: web.Request) -> web.Response:
        data = await request.json()
        response = await self._process_network_message(data)
        return web.json_response(response)

    async def _http_handle_register(self, request: web.Request) -> web.Response:
        data = await request.json()
        response = await self._process_network_message(data)
        return web.json_response(response)

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not NETWORK_AVAILABLE:
            raise RuntimeError("HTTP client unavailable")
        timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
    
    async def _broadcast_proposal(self, proposal: ConsensusProposal):
        """Broadcast proposal to all peer nodes."""
        message = {
            'type': 'proposal',
            'proposal': {
                'proposal_id': proposal.proposal_id,
                'node_id': proposal.node_id,
                'timestamp': proposal.timestamp.isoformat(),
                'query_id': proposal.query_id,
                'expert_responses': proposal.expert_responses,
                'confidence_scores': proposal.confidence_scores,
                'metadata': proposal.metadata,
                'signature': proposal.signature
            }
        }
        
        if NETWORK_AVAILABLE:
            targets = self.config.node_endpoints or []
            if self.role != NodeRole.COORDINATOR and self.config.coordinator_endpoint:
                targets = [self.config.coordinator_endpoint] + targets
            for endpoint in targets:
                try:
                    await self._post_json(endpoint.rstrip("/") + "/proposal", message)
                except Exception as e:
                    logger.error(f" Failed to broadcast proposal to {endpoint}: {e}")
        else:
            raise RuntimeError("Network unavailable for broadcasting")
    
    async def _validate_proposal(self, proposal: ConsensusProposal) -> bool:
        """Validates a consensus proposal."""
        # Check timestamp (not too old or too far in future)
        now = datetime.now()
        time_diff = abs((now - proposal.timestamp).total_seconds())
        if time_diff > 300:  # 5 minutes
            return False
        
        # Check node trust score
        node_info = self.peer_nodes.get(proposal.node_id)
        if node_info and node_info.trust_score < self.config.trust_threshold:
            return False
        
        # Validate signature if required
        if self.config.require_signatures and proposal.signature:
            if not self._verify_signature(proposal):
                return False
        
        # Check data integrity
        if not proposal.expert_responses or not proposal.confidence_scores:
            return False
        
        if len(proposal.expert_responses) != len(proposal.confidence_scores):
            return False
        
        return True
    
    async def _generate_local_consensus(self, expert_responses: List[Dict[str, Any]], 
                                      confidence_scores: List[float]) -> Dict[str, Any]:
        """Generates local consensus from expert responses."""
        # Simple weighted average consensus
        if not expert_responses:
            return {}

        response_texts = []
        for response in expert_responses:
            text = response.get("response") or response.get("text") or json.dumps(response)
            response_texts.append(text)

        response_embeddings = await self._encode_texts(response_texts)
        confidence_scores = np.array(confidence_scores)
        
        # Weighted average
        weights = confidence_scores / np.sum(confidence_scores)
        consensus_embedding = np.average(response_embeddings, axis=0, weights=weights)
        
        # Generate consensus response
        consensus_response = {
            'consensus_embedding': consensus_embedding.tolist(),
            'consensus_confidence': float(np.mean(confidence_scores)),
            'num_responses': len(expert_responses),
            'consensus_method': 'weighted_average',
            'timestamp': datetime.now().isoformat()
        }
        
        return consensus_response
    
    async def _calculate_agreement(self, local_response: Dict[str, Any], 
                                 expert_responses: List[Dict[str, Any]]) -> float:
        """Calculate agreement score between local and expert responses."""
        if not local_response or not expert_responses:
            return 0.0

        local_embedding = np.array(local_response.get("consensus_embedding") or [], dtype=np.float32)
        if local_embedding.size == 0:
            return 0.0

        response_texts = []
        for response in expert_responses:
            text = response.get("response") or response.get("text") or json.dumps(response)
            response_texts.append(text)

        embeddings = await self._encode_texts(response_texts)
        mean_embedding = embeddings.mean(axis=0)
        denom = (np.linalg.norm(local_embedding) * np.linalg.norm(mean_embedding)) + 1e-8
        similarity = float(np.dot(local_embedding, mean_embedding) / denom)
        return max(0.0, min(1.0, similarity))
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connectivity."""
        while self.status in [NodeStatus.ACTIVE, NodeStatus.DEGRADED]:
            try:
                # Send heartbeat to coordinator
                if NETWORK_AVAILABLE and self.config.coordinator_endpoint:
                    heartbeat = {
                        'type': 'heartbeat',
                        'node_id': self.node_id,
                        'status': self.status.value,
                        'metrics': self.performance_metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    await self._post_json(self.config.coordinator_endpoint.rstrip("/") + "/heartbeat", heartbeat)
                
                # Update peer node status
                await self._update_peer_status()
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f" Heartbeat error: {e}")
                self.status = NodeStatus.DEGRADED
                await asyncio.sleep(5)  # Retry after 5 seconds
    
    async def _consensus_loop(self):
        """Main consensus processing loop."""
        while self.status in [NodeStatus.ACTIVE, NodeStatus.DEGRADED]:
            try:
                # Process pending consensus proposals
                await self._process_pending_consensus()
                
                # Clean up old proposals
                await self._cleanup_old_proposals()
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f" Consensus loop error: {e}")
                await asyncio.sleep(5)
    
    async def _maintenance_loop(self):
        """Background maintenance tasks."""
        while self.status != NodeStatus.FAILED:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check node health
                await self._health_check()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f" Maintenance error: {e}")
                await asyncio.sleep(30)
    
    async def _update_peer_status(self):
        """Update status of peer nodes."""
        current_time = datetime.now()
        
        for node_id, node_info in self.peer_nodes.items():
            time_since_heartbeat = (current_time - node_info.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.config.node_failure_timeout:
                if node_info.status != NodeStatus.FAILED:
                    logger.warning(f"️ Node {node_id} marked as failed (no heartbeat)")
                    node_info.status = NodeStatus.FAILED
            elif time_since_heartbeat > self.config.heartbeat_interval * 2:
                if node_info.status == NodeStatus.ACTIVE:
                    logger.warning(f"️ Node {node_id} marked as degraded")
                    node_info.status = NodeStatus.DEGRADED
    
    async def _process_pending_consensus(self):
        """Process pending consensus proposals."""
        # Mock implementation
        if self.active_proposals:
            logger.debug(f"Processing {len(self.active_proposals)} pending proposals")
    
    async def _cleanup_old_proposals(self):
        """Clean up old consensus proposals."""
        current_time = datetime.now()
        expired_proposals = []
        
        for proposal_id, proposal in self.active_proposals.items():
            age = (current_time - proposal.timestamp).total_seconds()
            if age > self.config.max_consensus_time:
                expired_proposals.append(proposal_id)
        
        for proposal_id in expired_proposals:
            del self.active_proposals[proposal_id]
            logger.info(f" Cleaned up expired proposal {proposal_id}")
    
    def _update_performance_metrics(self):
        """Update node performance metrics."""
        # Update uptime
        self.performance_metrics['uptime'] += 60  # seconds
        
        # Calculate success rate
        total_proposals = self.performance_metrics['proposals_processed']
        if total_proposals > 0:
            self.performance_metrics['success_rate'] = (
                self.performance_metrics['consensus_participated'] / total_proposals
            )
    
    async def _health_check(self):
        """Perform health check on the node."""
        # Check memory usage, CPU, network connectivity, etc.
        # For now, just ensure we're not in failed state
        if self.status == NodeStatus.FAILED and self.config.auto_recovery:
            logger.info(" Attempting auto-recovery...")
            try:
                await self.start()
            except Exception as e:
                logger.error(f" Auto-recovery failed: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old consensus history and cached data."""
        # Keep only last 1000 consensus results
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-1000:]
    
    def _sign_proposal(self, proposal: ConsensusProposal) -> str:
        """Sign a consensus proposal."""
        # Mock signature implementation
        proposal_data = f"{proposal.proposal_id}{proposal.node_id}{proposal.timestamp}"
        return hashlib.sha256(proposal_data.encode()).hexdigest()
    
    def _sign_response(self, response: Dict[str, Any]) -> str:
        """Sign a consensus response."""
        # Mock signature implementation
        response_data = f"{response['node_id']}{response['proposal_id']}{response['vote']}"
        return hashlib.sha256(response_data.encode()).hexdigest()
    
    def _verify_signature(self, proposal: ConsensusProposal) -> bool:
        """Verify proposal signature."""
        # Mock signature verification
        expected_signature = self._sign_proposal(proposal)
        return proposal.signature == expected_signature
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive node status."""
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'status': self.status.value,
            'peer_nodes': len(self.peer_nodes),
            'active_proposals': len(self.active_proposals),
            'consensus_history': len(self.consensus_history),
            'performance_metrics': self.performance_metrics,
            'uptime_hours': self.performance_metrics['uptime'] / 3600
        }


class FederatedConsensusCoordinator:
    """Coordinator for federated consensus system."""
    
    def __init__(self, config: FederatedConsensusConfig):
        self.config = config
        self.nodes: Dict[str, NodeInfo] = {}
        self.active_consensus: Dict[str, Dict[str, Any]] = {}
        self.consensus_results: List[ConsensusResult] = []
        
        # Create coordinator node
        self.coordinator_node = FederatedConsensusNode(
            "coordinator", NodeRole.COORDINATOR, config
        )
    
    async def start(self):
        """Start the federated consensus coordinator."""
        logger.info(" Starting federated consensus coordinator")
        await self.coordinator_node.start()
    
    async def coordinate_consensus(self, query_id: str, proposals: List[ConsensusProposal]) -> ConsensusResult:
        """Coordinate consensus across multiple nodes."""
        consensus_id = f"consensus_{query_id}_{int(time.time())}"
        
        logger.info(f" Coordinating consensus {consensus_id} with {len(proposals)} proposals")
        
        # Collect responses from participating nodes
        responses = []
        for proposal in proposals:
            for node_id in self.nodes:
                if self.nodes[node_id].status == NodeStatus.ACTIVE:
                    try:
                        response = await self._request_consensus_participation(node_id, proposal)
                        responses.append(response)
                    except Exception as e:
                        logger.warning(f"️ Node {node_id} failed to participate: {e}")
        
        # Calculate consensus result
        consensus_result = await self._calculate_final_consensus(
            consensus_id, query_id, proposals, responses
        )
        
        self.consensus_results.append(consensus_result)
        
        logger.info(f" Consensus {consensus_id} completed with {consensus_result.agreement_level:.2%} agreement")
        
        return consensus_result
    
    async def _request_consensus_participation(self, node_id: str, proposal: ConsensusProposal) -> Dict[str, Any]:
        """Request consensus participation from a node."""
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node {node_id}")
        payload = {
            "proposal": {
                'proposal_id': proposal.proposal_id,
                'node_id': proposal.node_id,
                'timestamp': proposal.timestamp.isoformat(),
                'query_id': proposal.query_id,
                'expert_responses': proposal.expert_responses,
                'confidence_scores': proposal.confidence_scores,
                'metadata': proposal.metadata,
                'signature': proposal.signature
            }
        }
        return await self._post_json(node.endpoint.rstrip("/") + "/consensus/participate", payload)
    
    async def _calculate_final_consensus(self, consensus_id: str, query_id: str,
                                       proposals: List[ConsensusProposal], 
                                       responses: List[Dict[str, Any]]) -> ConsensusResult:
        """Calculate final consensus result."""
        # Count votes
        accept_votes = sum(1 for r in responses if r.get('vote') == 'accept')
        total_votes = len(responses)
        
        agreement_level = accept_votes / total_votes if total_votes > 0 else 0.0
        
        # Determine final response (simplified)
        if agreement_level >= self.config.consensus_threshold:
            # Use the proposal with highest average confidence
            best_proposal = max(proposals, key=lambda p: np.mean(p.confidence_scores))
            final_response = best_proposal.expert_responses[0]  # Simplified
            consensus_confidence = agreement_level
        else:
            # No consensus reached
            final_response = {'error': 'No consensus reached'}
            consensus_confidence = 0.0
        
        participating_nodes = [r['node_id'] for r in responses]
        
        return ConsensusResult(
            consensus_id=consensus_id,
            query_id=query_id,
            final_response=final_response,
            participating_nodes=participating_nodes,
            consensus_confidence=consensus_confidence,
            consensus_time=time.time(),
            agreement_level=agreement_level
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_nodes = sum(1 for node in self.nodes.values() if node.status == NodeStatus.ACTIVE)
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'active_consensus': len(self.active_consensus),
            'completed_consensus': len(self.consensus_results),
            'system_health': active_nodes / len(self.nodes) if self.nodes else 0.0
        }


def create_federated_consensus_system(
    node_id: str,
    role: NodeRole = NodeRole.PARTICIPANT,
    coordinator_endpoint: str = "ws://localhost:8765",
    min_participants: int = 3
) -> FederatedConsensusNode:
    """Creates a federated consensus node with optimal configuration."""
    
    config = FederatedConsensusConfig(
        coordinator_endpoint=coordinator_endpoint,
        min_participants=min_participants,
        consensus_protocol=ConsensusProtocol.RAFT,
        byzantine_fault_tolerance=True,
        enable_caching=True,
        enable_validation=True
    )
    
    return FederatedConsensusNode(node_id, role, config)
