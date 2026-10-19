"""
Unit tests for the HPC (High Performance Computing) module.

Tests distributed training infrastructure for EuroHPC systems.
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import HPC modules
try:
    from training.hpc.distributed_init import (
        DistributedConfig,
        DistributedBackend,
    )
    from training.hpc.slurm_launcher import (
        SlurmConfig,
        SlurmLauncher,
        detect_slurm_environment,
        get_slurm_job_info,
    )
    from training.hpc.deepspeed_config import (
        DeepSpeedConfig,
        ZeROConfig,
        ZeROStage,
        OffloadDevice,
        create_deepspeed_config,
    )
    from training.hpc.fsdp_wrapper import (
        FSDPConfig,
        ShardingStrategy,
        create_fsdp_config,
    )
    from training.hpc.checkpoint_manager import (
        CheckpointConfig,
        CheckpointMetadata,
        DistributedCheckpointManager,
    )
    from training.hpc.eurohpc_config import (
        EuroHPCConfig,
        EuroHPCSystem,
        detect_eurohpc_system,
        SYSTEM_SPECS,
    )
    HPC_AVAILABLE = True
except ImportError as e:
    HPC_AVAILABLE = False
    IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not HPC_AVAILABLE,
    reason=f"HPC module not importable: {IMPORT_ERROR if not HPC_AVAILABLE else ''}"
)


class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedConfig()

        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.master_addr == "localhost"
        assert config.master_port == 29500
        assert config.backend == DistributedBackend.NCCL

    def test_slurm_detection(self):
        """Test SLURM environment detection."""
        # Mock SLURM environment
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_NNODES": "4",
            "SLURM_NTASKS": "16",
            "SLURM_PROCID": "5",
            "SLURM_LOCALID": "1",
            "SLURM_NODELIST": "node[001-004]",
        }

        with patch.dict(os.environ, slurm_env, clear=False):
            config = DistributedConfig()

            assert config.world_size == 16
            assert config.rank == 5
            assert config.local_rank == 1

    def test_torchrun_detection(self):
        """Test torchrun environment detection."""
        torchrun_env = {
            "RANK": "3",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "3",
            "MASTER_ADDR": "192.168.1.1",
            "MASTER_PORT": "29501",
        }

        with patch.dict(os.environ, torchrun_env, clear=False):
            config = DistributedConfig()

            assert config.world_size == 8
            assert config.rank == 3
            assert config.local_rank == 3
            assert config.master_addr == "192.168.1.1"
            assert config.master_port == 29501


class TestSlurmLauncher:
    """Tests for SLURM launcher."""

    def test_slurm_config_defaults(self):
        """Test default SLURM configuration."""
        config = SlurmConfig()

        assert config.job_name == "capibara_train"
        assert config.nodes == 1
        assert config.gpus_per_node == 4
        assert config.time_limit == "24:00:00"

    def test_generate_script(self):
        """Test SLURM script generation."""
        config = SlurmConfig(
            job_name="test_job",
            nodes=2,
            gpus_per_node=4,
            time_limit="12:00:00",
            partition="gpu",
            modules=["cuda/12.1", "python/3.11"],
            training_script="train.py",
            training_args=["--lr", "1e-4"],
        )

        launcher = SlurmLauncher(config)
        script = launcher.generate_script()

        # Check required SBATCH directives
        assert "#SBATCH --job-name=test_job" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --gpus-per-node=4" in script
        assert "#SBATCH --time=12:00:00" in script
        assert "#SBATCH --partition=gpu" in script

        # Check module loading
        assert "module load cuda/12.1" in script
        assert "module load python/3.11" in script

        # Check environment setup
        assert "MASTER_ADDR" in script
        assert "MASTER_PORT" in script
        assert "NCCL" in script

        # Check training command
        assert "train.py" in script
        assert "--lr" in script

    def test_generate_script_with_deepspeed(self):
        """Test SLURM script with DeepSpeed."""
        config = SlurmConfig(
            job_name="ds_job",
            nodes=4,
            use_deepspeed=True,
            deepspeed_config="ds_config.json",
        )

        launcher = SlurmLauncher(config)
        script = launcher.generate_script()

        assert "deepspeed" in script
        assert "ds_config.json" in script

    def test_detect_slurm_environment(self):
        """Test SLURM environment detection."""
        # Not in SLURM
        assert detect_slurm_environment() == ("SLURM_JOB_ID" in os.environ)

        # Mock SLURM environment
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            assert detect_slurm_environment() == True

    def test_get_slurm_job_info(self):
        """Test SLURM job info retrieval."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NAME": "test_job",
            "SLURM_NNODES": "2",
            "SLURM_NTASKS": "8",
        }

        with patch.dict(os.environ, slurm_env, clear=False):
            info = get_slurm_job_info()

            assert info["job_id"] == "12345"
            assert info["job_name"] == "test_job"
            assert info["num_nodes"] == 2
            assert info["num_tasks"] == 8


class TestDeepSpeedConfig:
    """Tests for DeepSpeed configuration."""

    def test_zero_config(self):
        """Test ZeRO configuration."""
        zero = ZeROConfig(stage=ZeROStage.STAGE_2)
        config_dict = zero.to_dict()

        assert config_dict["stage"] == 2
        assert config_dict["overlap_comm"] == True
        assert config_dict["contiguous_gradients"] == True

    def test_zero_stage3_config(self):
        """Test ZeRO Stage 3 with offloading."""
        zero = ZeROConfig(
            stage=ZeROStage.STAGE_3,
            offload_optimizer=OffloadDevice.CPU,
            offload_param=OffloadDevice.CPU,
        )
        config_dict = zero.to_dict()

        assert config_dict["stage"] == 3
        assert "offload_optimizer" in config_dict
        assert config_dict["offload_optimizer"]["device"] == "cpu"
        assert "offload_param" in config_dict

    def test_deepspeed_config(self):
        """Test full DeepSpeed configuration."""
        config = DeepSpeedConfig(
            train_batch_size=256,
            train_micro_batch_size_per_gpu=8,
            gradient_accumulation_steps=8,
        )

        config_dict = config.to_dict()

        assert config_dict["train_batch_size"] == 256
        assert config_dict["train_micro_batch_size_per_gpu"] == 8
        assert config_dict["gradient_accumulation_steps"] == 8
        assert "zero_optimization" in config_dict
        assert "bf16" in config_dict

    def test_deepspeed_json_export(self):
        """Test JSON export."""
        config = DeepSpeedConfig()
        json_str = config.to_json()

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert "train_batch_size" in parsed
        assert "zero_optimization" in parsed

    def test_deepspeed_save_load(self):
        """Test saving config to file."""
        config = DeepSpeedConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ds_config.json"
            config.save(path)

            assert path.exists()

            with open(path) as f:
                loaded = json.load(f)

            assert loaded["train_batch_size"] == config.train_batch_size

    def test_create_deepspeed_config_factory(self):
        """Test DeepSpeed config factory function."""
        # Small model - should use ZeRO Stage 1 or 2
        config = create_deepspeed_config(
            model_size="3B",
            num_gpus=8,
            num_nodes=1,
            gpu_memory_gb=80,
        )

        assert config.zero.stage in [ZeROStage.STAGE_1, ZeROStage.STAGE_2]

        # Large model - should use ZeRO Stage 3
        config = create_deepspeed_config(
            model_size="70B",
            num_gpus=8,
            num_nodes=1,
            gpu_memory_gb=80,
        )

        assert config.zero.stage == ZeROStage.STAGE_3


class TestFSDPConfig:
    """Tests for FSDP configuration."""

    def test_fsdp_config_defaults(self):
        """Test default FSDP configuration."""
        config = FSDPConfig()

        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert config.use_mixed_precision == True
        assert config.param_dtype == "bfloat16"
        assert config.activation_checkpointing == True

    def test_create_fsdp_config_factory(self):
        """Test FSDP config factory function."""
        config = create_fsdp_config(
            model_size="7B",
            num_gpus=8,
            gpu_memory_gb=80,
        )

        assert config.sharding_strategy in [
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.HYBRID_SHARD,
        ]


class TestCheckpointManager:
    """Tests for checkpoint manager."""

    def test_checkpoint_config(self):
        """Test checkpoint configuration."""
        config = CheckpointConfig(
            checkpoint_dir="/tmp/checkpoints",
            save_every_n_steps=500,
            max_checkpoints=3,
        )

        assert config.save_every_n_steps == 500
        assert config.max_checkpoints == 3

    def test_checkpoint_metadata(self):
        """Test checkpoint metadata."""
        metadata = CheckpointMetadata(
            step=1000,
            epoch=2,
            timestamp="2024-01-01T00:00:00",
            metrics={"loss": 0.5, "perplexity": 10.0},
        )

        data = metadata.to_dict()

        assert data["step"] == 1000
        assert data["epoch"] == 2
        assert data["metrics"]["loss"] == 0.5

        # Test round-trip
        loaded = CheckpointMetadata.from_dict(data)
        assert loaded.step == metadata.step
        assert loaded.metrics == metadata.metrics

    def test_checkpoint_manager_init(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir)
            manager = DistributedCheckpointManager(config, rank=0, world_size=1)

            assert manager.rank == 0
            assert manager.world_size == 1
            assert manager.checkpoint_dir == Path(tmpdir)

    def test_should_save(self):
        """Test should_save logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=tmpdir,
                save_every_n_steps=100,
            )
            manager = DistributedCheckpointManager(config, rank=0, world_size=1)

            # Should not save immediately
            assert manager.should_save(50) == False

            # Should save after 100 steps
            assert manager.should_save(100) == True


class TestEuroHPCConfig:
    """Tests for EuroHPC configuration."""

    def test_system_specs(self):
        """Test system specifications."""
        assert EuroHPCSystem.LEONARDO in SYSTEM_SPECS
        assert EuroHPCSystem.JUWELS in SYSTEM_SPECS
        assert EuroHPCSystem.LUMI in SYSTEM_SPECS

        leonardo_specs = SYSTEM_SPECS[EuroHPCSystem.LEONARDO]
        assert leonardo_specs.gpu_type == "NVIDIA A100"
        assert leonardo_specs.gpu_memory_gb == 64
        assert leonardo_specs.gpus_per_node == 4

    def test_eurohpc_config(self):
        """Test EuroHPC configuration."""
        config = EuroHPCConfig(
            system=EuroHPCSystem.LEONARDO,
            nodes=4,
            model_size="7B",
        )

        assert config.system == EuroHPCSystem.LEONARDO
        assert config.nodes == 4
        assert config.total_gpus == 16  # 4 nodes * 4 GPUs
        assert config.specs.gpu_type == "NVIDIA A100"

    def test_global_batch_size(self):
        """Test global batch size calculation."""
        config = EuroHPCConfig(
            system=EuroHPCSystem.JUWELS,
            nodes=2,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
        )

        # 2 nodes * 4 GPUs * 4 batch * 8 accum = 256
        assert config.global_batch_size == 256

    def test_get_slurm_config(self):
        """Test SLURM config generation."""
        config = EuroHPCConfig(
            system=EuroHPCSystem.LEONARDO,
            nodes=4,
            time_limit="48:00:00",
            partition="gpu",
        )

        slurm_config = config.get_slurm_config()

        assert slurm_config.nodes == 4
        assert slurm_config.gpus_per_node == 4
        assert slurm_config.time_limit == "48:00:00"

    def test_get_deepspeed_config(self):
        """Test DeepSpeed config generation."""
        config = EuroHPCConfig(
            system=EuroHPCSystem.JUWELS,
            model_size="13B",
            nodes=4,
        )

        ds_config = config.get_deepspeed_config()

        assert ds_config.train_batch_size > 0
        assert ds_config.zero.stage in [ZeROStage.STAGE_1, ZeROStage.STAGE_2, ZeROStage.STAGE_3]

    def test_detect_system(self):
        """Test system auto-detection."""
        # Default should be GENERIC when not on HPC
        system = detect_eurohpc_system()
        assert system == EuroHPCSystem.GENERIC

        # Test Leonardo detection
        with patch.dict(os.environ, {"HOSTNAME": "leonardo-gpu001"}):
            system = detect_eurohpc_system()
            assert system == EuroHPCSystem.LEONARDO

        # Test LUMI detection
        with patch.dict(os.environ, {"LMOD_SYSTEM_NAME": "LUMI"}):
            system = detect_eurohpc_system()
            assert system == EuroHPCSystem.LUMI


class TestIntegration:
    """Integration tests for HPC module."""

    def test_full_config_pipeline(self):
        """Test complete configuration pipeline."""
        # Create EuroHPC config
        config = EuroHPCConfig(
            system=EuroHPCSystem.JUWELS,
            nodes=8,
            model_size="30B",
            backend=DistributedBackend.DEEPSPEED,
            zero_stage=3,
            cpu_offload=True,
        )

        # Generate all sub-configs
        slurm_config = config.get_slurm_config()
        ds_config = config.get_deepspeed_config()
        checkpoint_config = config.get_checkpoint_config()

        # Verify consistency
        assert slurm_config.nodes == config.nodes
        assert ds_config.zero.stage == ZeROStage.STAGE_3
        assert ds_config.zero.offload_optimizer != OffloadDevice.NONE

        # Generate SLURM script
        launcher = SlurmLauncher(slurm_config)
        script = launcher.generate_script()

        assert "deepspeed" in script
        assert f"--nodes={config.nodes}" in script


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
