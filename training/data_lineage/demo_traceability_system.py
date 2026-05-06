#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Complete Data Traceability System Demo - CapibaraGPT v3
==========================================================

Comprehensive demonstration of the data lineage and traceability system:

1. Blockchain-like audit logging
2. Parameter influence mapping  
3. Dataset parameter control (enable/disable)
4. Compliance reporting
5. Real-time traceability during training

This demo shows how to track every dataset's influence on model parameters
and control them granularly for compliance and debugging.
"""

import asyncio
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import jax.numpy as jnp

# BACKLOG-005: this file is a demo-only script. Do NOT configure the root
# logger at import time - that side-effect leaks into any process that
# imports the data_lineage package. Logging is configured only inside the
# CLI entrypoint below.
logger = logging.getLogger(__name__)


def _configure_demo_logging() -> None:
    """Configure root logging for the demo CLI only."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

try:
    from .blockchain_audit_log import (
        BlockchainAuditLog, create_blockchain_audit_log,
        DataProvenanceHash, ImmutableLogEntry
    )
    from .dataset_parameter_controller import (
        DatasetParameterController, create_dataset_parameter_controller,
        ParameterMask, DatasetControlPolicy
    )
    FULL_LINEAGE_AVAILABLE = True
except ImportError:
    logger.warning("️ Full lineage system not available - running mock demo")
    FULL_LINEAGE_AVAILABLE = False

class _DemoMockModel:
    """Private demo mock model (BACKLOG-005: not exported)."""
    
    def __init__(self, size: str = "300M"):
        self.size = size
        self.parameters = self._create_mock_parameters()
        
    def _create_mock_parameters(self) -> Dict[str, jnp.ndarray]:
        """Create mock model parameters."""
        if self.size == "300M":
            params = {
                "embedding.weight": jnp.ones((50000, 768)),
                "transformer.layer_0.attention.weight": jnp.ones((768, 768)),
                "transformer.layer_0.ffn.weight": jnp.ones((768, 3072)),
                "transformer.layer_1.attention.weight": jnp.ones((768, 768)),
                "transformer.layer_1.ffn.weight": jnp.ones((768, 3072)),
                "output.weight": jnp.ones((768, 50000))
            }
        else:
            # Larger model
            params = {
                "embedding.weight": jnp.ones((50000, 1024)),
                "transformer.layer_0.attention.weight": jnp.ones((1024, 1024)),
                "transformer.layer_0.ffn.weight": jnp.ones((1024, 4096)),
                "output.weight": jnp.ones((1024, 50000))
            }
        
        return params

class TraceabilitySystemDemo:
    """
    Complete demonstration of the data traceability system.
    
    Shows how to:
    1. Track training data influences with blockchain audit
    2. Map parameters to datasets
    3. Enable/disable parameters by dataset
    4. Generate compliance reports
    """
    
    def __init__(self, demo_dir: str = "traceability_demo"):
        self.demo_dir = Path(demo_dir)
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.audit_log = None
        self.parameter_controller = None
        self.mock_model = _DemoMockModel("300M")
        
        # Demo datasets
        self.demo_datasets = {
            "linux_kernel": {
                "size_gb": 25,
                "description": "Linux kernel source code and documentation",
                "compliance_level": "open_source"
            },
            "medical_papers": {
                "size_gb": 15,
                "description": "Medical research papers and clinical data",
                "compliance_level": "restricted"
            },
            "robotics_data": {
                "size_gb": 35,
                "description": "Unitree robotics datasets and manipulation data",
                "compliance_level": "commercial"
            },
            "physics_research": {
                "size_gb": 50,
                "description": "CERN particle physics experimental data",
                "compliance_level": "research_only"
            }
        }
        
        logger.info(f" TraceabilitySystemDemo initialized")
        logger.info(f" Demo directory: {self.demo_dir}")
    
    async def run_complete_demo(self):
        """Run the complete traceability system demonstration."""
        logger.debug(" COMPLETE DATA TRACEABILITY SYSTEM DEMO")
        logger.info("=" * 60)
        
        if not FULL_LINEAGE_AVAILABLE:
            await self._run_mock_demo()
            return
        
        # Step 1: Initialize blockchain audit log
        logger.info("\n STEP 1: Initialize Blockchain Audit Log")
        await self._demo_blockchain_audit()
        
        # Step 2: Simulate training with data tracking
        logger.info("\n STEP 2: Simulate Training with Data Tracking")
        await self._demo_training_run()
        
        # Step 3: Initialize parameter controller
        logger.info("\n️ STEP 3: Initialize Parameter Controller")
        await self._demo_parameter_controller()
        
        # Step 4: Demonstrate dataset control
        logger.info("\n️ STEP 4: Demonstrate Dataset Parameter Control")
        await self._demo_dataset_control()
        
        # Step 5: Generate compliance reports
        logger.info("\n STEP 5: Generate Compliance Reports")
        await self._demo_compliance_reporting()
        
        # Step 6: Show real-time verification
        logger.info("\n STEP 6: Verify Blockchain Integrity")
        await self._demo_integrity_verification()
        
        logger.info("\n COMPLETE DEMO FINISHED!")
        logger.debug(" All traceability components demonstrated successfully")
    
    async def _demo_blockchain_audit(self):
        """Demonstrate blockchain audit log initialization."""
        audit_dir = self.demo_dir / "audit_logs"
        self.audit_log = create_blockchain_audit_log(str(audit_dir))
        
        logger.info(f" Blockchain audit log initialized")
        logger.info(f" Genesis block created and mined")
        logger.info(f" Cryptographic integrity enabled")
    
    async def _demo_training_run(self):
        """Simulate training steps with audit logging."""
        logger.info(" Simulating training steps with audit tracking...")
        
        training_steps = [
            {
                "dataset_id": "linux_kernel",
                "step": 1000,
                "affected_params": ["embedding.weight", "transformer.layer_0.attention.weight"],
                "param_deltas": {"embedding.weight": 0.001, "transformer.layer_0.attention.weight": 0.0008},
                "gradient_norms": {"embedding.weight": 0.15, "transformer.layer_0.attention.weight": 0.12}
            },
            {
                "dataset_id": "medical_papers", 
                "step": 2000,
                "affected_params": ["transformer.layer_1.ffn.weight", "output.weight"],
                "param_deltas": {"transformer.layer_1.ffn.weight": 0.0012, "output.weight": 0.0015},
                "gradient_norms": {"transformer.layer_1.ffn.weight": 0.18, "output.weight": 0.22}
            },
            {
                "dataset_id": "robotics_data",
                "step": 3000, 
                "affected_params": ["transformer.layer_0.ffn.weight", "transformer.layer_1.attention.weight"],
                "param_deltas": {"transformer.layer_0.ffn.weight": 0.0009, "transformer.layer_1.attention.weight": 0.0011},
                "gradient_norms": {"transformer.layer_0.ffn.weight": 0.14, "transformer.layer_1.attention.weight": 0.16}
            },
            {
                "dataset_id": "physics_research",
                "step": 4000,
                "affected_params": ["embedding.weight", "output.weight"],
                "param_deltas": {"embedding.weight": 0.0007, "output.weight": 0.0013},
                "gradient_norms": {"embedding.weight": 0.11, "output.weight": 0.19}
            }
        ]
        
        for step_data in training_steps:
            # Add training event to blockchain audit log
            entry_id = self.audit_log.add_training_event(
                dataset_id=step_data["dataset_id"],
                affected_parameters=step_data["affected_params"],
                parameter_deltas=step_data["param_deltas"],
                gradient_norms=step_data["gradient_norms"],
                data_hash=f"hash_{step_data['dataset_id']}_{step_data['step']}",
                data_size=self.demo_datasets[step_data["dataset_id"]]["size_gb"] * 1024**3,
                metadata={
                    "training_step": step_data["step"],
                    "compliance_level": self.demo_datasets[step_data["dataset_id"]]["compliance_level"]
                }
            )
            
            logger.info(f"    Step {step_data['step']}: {step_data['dataset_id']} → {len(step_data['affected_params'])} parameters")
            await asyncio.sleep(0.1)  # Small delay for demo
        
        logger.info(f" {len(training_steps)} training steps logged to blockchain")
    
    async def _demo_parameter_controller(self):
        """Demonstrate parameter controller initialization."""
        # Export audit report for parameter controller
        audit_report_path = self.demo_dir / "audit_report.json"
        audit_report = self.audit_log.export_audit_report(audit_report_path)
        
        # Initialize parameter controller with lineage data
        self.parameter_controller = create_dataset_parameter_controller(
            model_parameters=self.mock_model.parameters,
            lineage_file=str(audit_report_path)
        )
        
        logger.info(f" Parameter controller initialized")
        logger.info(f" Loaded lineage for {len(audit_report['parameter_summary'])} parameters")
        logger.info(f"️ Tracking {len(audit_report['dataset_summary'])} datasets")
    
    async def _demo_dataset_control(self):
        """Demonstrate dataset parameter control capabilities."""
        logger.info("️ Testing dataset parameter control...")
        
        # Get initial state
        initial_summary = self.parameter_controller.get_global_control_summary()
        logger.info(f" Initial state: {initial_summary['control_summary']['active_parameters']} active parameters")
        
        # Test 1: Disable medical dataset for compliance
        logger.info("\n TEST 1: Disable medical dataset (compliance requirement)")
        success = self.parameter_controller.disable_dataset_parameters("medical_papers")
        if success:
            medical_report = self.parameter_controller.get_dataset_influence_report("medical_papers")
            logger.info(f"    Medical dataset disabled")
            logger.info(f"    Affected parameters: {medical_report['total_parameters']}")
            logger.info(f"    Active ratio: {medical_report['active_ratio']:.2%}")
        
        # Test 2: Create compliance policy
        logger.info("\n TEST 2: Create compliance policy")
        policy = self.parameter_controller.create_control_policy(
            policy_name="gdpr_compliance",
            enabled_datasets=["linux_kernel", "physics_research"],
            disabled_datasets=["medical_papers"],
            compliance_mode=True
        )
        logger.info(f"    Created policy: {policy.policy_name}")
        logger.info(f"    Enabled datasets: {len(policy.enabled_datasets)}")
        logger.info(f"    Disabled datasets: {len(policy.disabled_datasets)}")
        
        # Test 3: Apply compliance policy
        logger.info("\n️ TEST 3: Apply compliance policy")
        success = self.parameter_controller.apply_control_policy("gdpr_compliance")
        if success:
            final_summary = self.parameter_controller.get_global_control_summary()
            logger.info(f"    Policy applied successfully")
            logger.info(f"    Active parameters: {final_summary['control_summary']['active_parameters']}")
            logger.info(f"    Disabled parameters: {final_summary['control_summary']['disabled_parameters']}")
        
        # Test 4: Export control state
        logger.info("\n TEST 4: Export control state")
        control_export_path = self.demo_dir / "control_state.json"
        export_data = self.parameter_controller.export_control_state(control_export_path)
        logger.info(f"    Control state exported to {control_export_path}")
        logger.info(f"    Export includes {len(export_data['policies'])} policies")
    
    async def _demo_compliance_reporting(self):
        """Demonstrate compliance reporting capabilities."""
        logger.info(" Generating comprehensive compliance reports...")
        
        # Generate audit report
        audit_report_path = self.demo_dir / "full_audit_report.json"
        audit_report = self.audit_log.export_audit_report(audit_report_path)
        
        logger.info(f" Audit report generated")
        logger.info(f" Total entries: {audit_report['audit_metadata']['total_entries']}")
        logger.info(f" Total blocks: {audit_report['audit_metadata']['total_blocks']}")
        logger.info(f" Blockchain valid: {audit_report['audit_metadata']['blockchain_valid']}")
        
        # Dataset influence summary
        logger.info("\n DATASET INFLUENCE SUMMARY:")
        for dataset_id, dataset_info in audit_report["dataset_summary"].items():
            compliance_level = self.demo_datasets[dataset_id]["compliance_level"]
            logger.info(f"   • {dataset_id}: {dataset_info['total_events']} events, "
                  f"{len(dataset_info['parameters_affected'])} params, "
                  f"compliance: {compliance_level}")
        
        # Parameter influence summary
        logger.info("\n PARAMETER INFLUENCE SUMMARY:")
        for param_name, param_info in list(audit_report["parameter_summary"].items())[:3]:
            logger.info(f"   • {param_name}: {param_info['total_updates']} updates, "
                  f"{len(param_info['datasets_affecting'])} datasets")
    
    async def _demo_integrity_verification(self):
        """Demonstrate blockchain integrity verification."""
        logger.info(" Verifying blockchain integrity...")
        
        is_valid, issues = self.audit_log.verify_blockchain_integrity()
        
        if is_valid:
            logger.info(" Blockchain integrity verified - no issues found")
            logger.info(" All audit entries are tamper-evident")
            logger.info("️ Chain of custody is complete")
        else:
            logger.info(f" Blockchain integrity issues found: {issues}")
        
        # Show some blockchain statistics
        total_blocks = len(self.audit_log.blocks)
        total_entries = sum(len(block.entries) for block in self.audit_log.blocks)
        
        logger.info(f"\n BLOCKCHAIN STATISTICS:")
        logger.info(f"   • Total blocks: {total_blocks}")
        logger.info(f"   • Total entries: {total_entries}")
        logger.info(f"   • Average entries per block: {total_entries/total_blocks:.1f}")
        logger.info(f"   • Cryptographic difficulty: {self.audit_log.difficulty}")
    
    async def _run_mock_demo(self):
        """Run a simplified mock demo when full system isn't available."""
        logger.warning(" Running mock demonstration (full system not available)")
        logger.info("-" * 50)
        
        logger.info(" Mock blockchain audit log:")
        logger.info("    Genesis block created")
        logger.info("    4 training events logged")
        logger.info("    Cryptographic hashing enabled")
        
        logger.info("\n️ Mock parameter controller:")
        logger.info("    6 parameter groups tracked")
        logger.info("    4 datasets mapped to parameters")
        logger.info("    Binary masking available")
        
        logger.info("\n️ Mock dataset control:")
        logger.info("    medical_papers dataset disabled")
        logger.info("    GDPR compliance policy created")
        logger.info("    Active parameters: 85% (reduced from 100%)")
        
        logger.info("\n Mock compliance report:")
        logger.info("    Audit trail generated")
        logger.info("    4 blockchain blocks verified")
        logger.info("    Complete parameter lineage tracked")
        
        logger.info("\n Mock demo completed!")
        logger.info(" Install full dependencies to see real implementation")

def main():
    """Main demo function."""
    demo = TraceabilitySystemDemo()
    asyncio.run(demo.run_complete_demo())

if __name__ == "__main__":
    import os
    import sys

    if os.environ.get("CAPIBARA_DATA_LINEAGE_DEMO") != "1":
        sys.stderr.write(
            "demo_traceability_system.py is a demo-only script.\n"
            "Set CAPIBARA_DATA_LINEAGE_DEMO=1 to opt in (BACKLOG-005).\n"
        )
        sys.exit(2)
    _configure_demo_logging()
    main()
