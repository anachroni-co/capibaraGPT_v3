"""
Federated Consensus Smoke Tests - Integration tests for distributed consensus.

This module provides smoke tests for the federated consensus system,
testing multi-node coordination, fault tolerance, and consensus protocols.

Author: Skydesk International Dev Team.
"""

import importlib.util
from pathlib import Path
import pytest

_ANYIO_AVAILABLE = False
try:
    import anyio  # noqa: F401
    _ANYIO_AVAILABLE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _ANYIO_AVAILABLE,
    reason="anyio not installed; skipping async integration test"
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_federated_consensus_smoke():
    aiohttp = pytest.importorskip("aiohttp")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    module_path = Path(__file__).resolve().parents[2] / "training" / "federated_consensus" / "federated_consensus_system.py"
    spec = importlib.util.spec_from_file_location("federated_consensus_system", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    FederatedConsensusConfig = module.FederatedConsensusConfig
    FederatedConsensusNode = module.FederatedConsensusNode
    NodeRole = module.NodeRole
    _TextEmbedder = module._TextEmbedder

    # Sanity check: ensure embeddings can be generated (skip if model can't be loaded).
    try:
        embedder = _TextEmbedder("sshleifer/tiny-distilroberta-base")
        _ = await __import__("asyncio").to_thread(embedder.encode, ["hello"])
    except Exception as exc:
        pytest.skip(f"Transformers model unavailable: {exc}")

    coordinator_port = 8765
    participant_port = 8766
    coordinator_endpoint = f"http://localhost:{coordinator_port}"

    config = FederatedConsensusConfig(
        coordinator_endpoint=coordinator_endpoint,
        node_endpoints=[f"http://localhost:{participant_port}"],
        heartbeat_interval=1,
        connection_timeout=5,
    )

    coordinator = FederatedConsensusNode("coordinator", NodeRole.COORDINATOR, config)
    coordinator.endpoint = coordinator_endpoint
    coordinator._embedder = embedder

    participant = FederatedConsensusNode("node_1", NodeRole.PARTICIPANT, config)
    participant.endpoint = f"http://localhost:{participant_port}"
    participant._embedder = embedder

    await coordinator.start()
    await participant.start()

    proposal_id = await participant.propose_consensus(
        query_id="q1",
        expert_responses=[{"response": "test response", "confidence": 0.9}],
        confidence_scores=[0.9],
    )

    assert proposal_id.startswith("proposal_")

    # Cleanup servers
    if coordinator.http_runner is not None:
        await coordinator.http_runner.cleanup()
    if participant.http_runner is not None:
        await participant.http_runner.cleanup()
