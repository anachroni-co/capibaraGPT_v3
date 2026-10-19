"""
SLURM Job Scheduler Integration for EuroHPC

Provides utilities for:
- Generating SLURM job scripts
- Detecting SLURM environment
- Job management and monitoring
- Multi-node job coordination
"""

import os
import subprocess
import logging
import socket
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class Partition(Enum):
    """Common EuroHPC partition types."""
    GPU = "gpu"
    GPU_A100 = "gpu-a100"
    GPU_H100 = "gpu-h100"
    STANDARD = "standard"
    LARGE = "large"
    DEBUG = "debug"


class QOS(Enum):
    """Quality of Service levels."""
    NORMAL = "normal"
    HIGH = "high"
    DEBUG = "debug"
    LONG = "long"


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission."""

    # Job identification
    job_name: str = "capibara_train"

    # Resource allocation
    nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 8
    memory_gb: int = 256

    # Time allocation (format: D-HH:MM:SS or HH:MM:SS)
    time_limit: str = "24:00:00"

    # Partition and QOS
    partition: str = "gpu"
    qos: Optional[str] = None
    account: Optional[str] = None

    # Output configuration
    output_dir: str = "slurm_logs"
    output_pattern: str = "%x_%j_%N.out"
    error_pattern: str = "%x_%j_%N.err"

    # Environment
    modules: List[str] = field(default_factory=lambda: [
        "cuda/12.1",
        "cudnn/8.9",
        "nccl/2.18",
        "python/3.11",
    ])
    conda_env: Optional[str] = None
    venv_path: Optional[str] = None

    # GPU configuration
    gpu_type: Optional[str] = None  # e.g., "a100", "h100"
    exclusive: bool = True

    # Networking
    network: str = "sharp"  # InfiniBand SHARP for collective operations

    # Job dependencies
    dependency: Optional[str] = None  # e.g., "afterok:12345"

    # Email notifications
    mail_user: Optional[str] = None
    mail_type: str = "END,FAIL"

    # Extra SLURM options
    extra_sbatch_options: Dict[str, str] = field(default_factory=dict)

    # Training script configuration
    training_script: str = "train.py"
    training_args: List[str] = field(default_factory=list)

    # DeepSpeed/torchrun configuration
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    use_torchrun: bool = True


class SlurmLauncher:
    """Generates and submits SLURM jobs for distributed training."""

    def __init__(self, config: SlurmConfig):
        self.config = config
        self.job_id: Optional[int] = None

    def generate_script(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate SLURM batch script."""
        cfg = self.config

        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={cfg.job_name}",
            f"#SBATCH --nodes={cfg.nodes}",
            f"#SBATCH --ntasks-per-node={cfg.gpus_per_node}",
            f"#SBATCH --gpus-per-node={cfg.gpus_per_node}",
            f"#SBATCH --cpus-per-task={cfg.cpus_per_task}",
            f"#SBATCH --mem={cfg.memory_gb}G",
            f"#SBATCH --time={cfg.time_limit}",
            f"#SBATCH --partition={cfg.partition}",
        ]

        # Optional SLURM directives
        if cfg.qos:
            script_lines.append(f"#SBATCH --qos={cfg.qos}")
        if cfg.account:
            script_lines.append(f"#SBATCH --account={cfg.account}")
        if cfg.exclusive:
            script_lines.append("#SBATCH --exclusive")
        if cfg.gpu_type:
            script_lines.append(f"#SBATCH --gres=gpu:{cfg.gpu_type}:{cfg.gpus_per_node}")
        if cfg.dependency:
            script_lines.append(f"#SBATCH --dependency={cfg.dependency}")
        if cfg.mail_user:
            script_lines.append(f"#SBATCH --mail-user={cfg.mail_user}")
            script_lines.append(f"#SBATCH --mail-type={cfg.mail_type}")

        # Output files
        output_dir = Path(cfg.output_dir)
        script_lines.extend([
            f"#SBATCH --output={output_dir / cfg.output_pattern}",
            f"#SBATCH --error={output_dir / cfg.error_pattern}",
        ])

        # Extra options
        for key, value in cfg.extra_sbatch_options.items():
            script_lines.append(f"#SBATCH --{key}={value}")

        script_lines.append("")
        script_lines.append("# ============================================")
        script_lines.append("# CapibaraGPT EuroHPC Training Job")
        script_lines.append("# ============================================")
        script_lines.append("")

        # Print job info
        script_lines.extend([
            'echo "Job ID: $SLURM_JOB_ID"',
            'echo "Job Name: $SLURM_JOB_NAME"',
            'echo "Nodes: $SLURM_NNODES"',
            'echo "Tasks: $SLURM_NTASKS"',
            'echo "GPUs per node: $SLURM_GPUS_PER_NODE"',
            'echo "Node list: $SLURM_NODELIST"',
            'echo "Master node: $(scontrol show hostnames $SLURM_NODELIST | head -n 1)"',
            'echo ""',
        ])

        # Load modules
        if cfg.modules:
            script_lines.append("# Load required modules")
            script_lines.append("module purge")
            for module in cfg.modules:
                script_lines.append(f"module load {module}")
            script_lines.append("")

        # Activate environment
        if cfg.conda_env:
            script_lines.extend([
                "# Activate conda environment",
                f"source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {cfg.conda_env}",
                "",
            ])
        elif cfg.venv_path:
            script_lines.extend([
                "# Activate virtual environment",
                f"source {cfg.venv_path}/bin/activate",
                "",
            ])

        # Set up distributed environment
        script_lines.extend([
            "# Configure distributed training environment",
            'export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)',
            "export MASTER_PORT=29500",
            "export WORLD_SIZE=$SLURM_NTASKS",
            "export RANK=$SLURM_PROCID",
            "export LOCAL_RANK=$SLURM_LOCALID",
            "",
            "# NCCL configuration for InfiniBand",
            "export NCCL_DEBUG=WARN",
            "export NCCL_IB_DISABLE=0",
            "export NCCL_NET_GDR_LEVEL=5",
            "export NCCL_IB_GID_INDEX=3",
            "export NCCL_SOCKET_IFNAME=ib0",
            "export NCCL_ASYNC_ERROR_HANDLING=1",
            "",
            "# CUDA configuration",
            "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID",
            "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
            "",
        ])

        # Launch command
        script_lines.append("# Launch distributed training")

        total_gpus = cfg.nodes * cfg.gpus_per_node
        training_args_str = " ".join(cfg.training_args)

        if cfg.use_deepspeed:
            deepspeed_cfg = f"--deepspeed_config {cfg.deepspeed_config}" if cfg.deepspeed_config else ""
            script_lines.extend([
                f"srun --ntasks={total_gpus} --ntasks-per-node={cfg.gpus_per_node} \\",
                f"    deepspeed --num_gpus={cfg.gpus_per_node} --num_nodes={cfg.nodes} \\",
                f"    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\",
                f"    {cfg.training_script} {deepspeed_cfg} {training_args_str}",
            ])
        elif cfg.use_torchrun:
            script_lines.extend([
                f"srun --ntasks={total_gpus} --ntasks-per-node={cfg.gpus_per_node} \\",
                f"    torchrun --nnodes={cfg.nodes} --nproc_per_node={cfg.gpus_per_node} \\",
                f"    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \\",
                f"    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\",
                f"    {cfg.training_script} {training_args_str}",
            ])
        else:
            script_lines.extend([
                f"srun --ntasks={total_gpus} --ntasks-per-node={cfg.gpus_per_node} \\",
                f"    python {cfg.training_script} {training_args_str}",
            ])

        script_lines.extend([
            "",
            "# Capture exit status",
            "EXIT_STATUS=$?",
            "",
            'echo ""',
            'echo "Job completed with exit status: $EXIT_STATUS"',
            'echo "End time: $(date)"',
            "",
            "exit $EXIT_STATUS",
        ])

        script = "\n".join(script_lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(script)
            output_path.chmod(0o755)
            logger.info(f"SLURM script written to: {output_path}")

        return script

    def submit(self, script_path: Optional[Union[str, Path]] = None) -> int:
        """Submit job to SLURM scheduler."""
        if script_path is None:
            script_path = Path(f"/tmp/slurm_{self.config.job_name}_{int(time.time())}.sh")
            self.generate_script(script_path)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse job ID from output: "Submitted batch job 12345"
            output = result.stdout.strip()
            self.job_id = int(output.split()[-1])

            logger.info(f"Submitted SLURM job {self.job_id}")
            return self.job_id

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SLURM job: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("sbatch command not found. Is SLURM installed?")
            raise

    def get_status(self, job_id: Optional[int] = None) -> Dict[str, Any]:
        """Get status of a SLURM job."""
        job_id = job_id or self.job_id
        if job_id is None:
            raise ValueError("No job ID specified")

        try:
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "--format=%i|%j|%T|%M|%N|%P|%R", "--noheader"],
                capture_output=True,
                text=True,
                check=True
            )

            if not result.stdout.strip():
                return {"status": "COMPLETED", "job_id": job_id}

            parts = result.stdout.strip().split("|")
            return {
                "job_id": int(parts[0]),
                "name": parts[1],
                "state": parts[2],
                "time": parts[3],
                "nodes": parts[4],
                "partition": parts[5],
                "reason": parts[6] if len(parts) > 6 else "",
            }

        except subprocess.CalledProcessError:
            return {"status": "UNKNOWN", "job_id": job_id}

    def cancel(self, job_id: Optional[int] = None) -> bool:
        """Cancel a SLURM job."""
        job_id = job_id or self.job_id
        if job_id is None:
            raise ValueError("No job ID specified")

        try:
            subprocess.run(["scancel", str(job_id)], check=True)
            logger.info(f"Cancelled SLURM job {job_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False


def detect_slurm_environment() -> bool:
    """Check if running inside a SLURM job."""
    return "SLURM_JOB_ID" in os.environ


def get_slurm_job_info() -> Dict[str, Any]:
    """Get information about the current SLURM job."""
    if not detect_slurm_environment():
        return {}

    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "num_nodes": int(os.environ.get("SLURM_NNODES", "1")),
        "num_tasks": int(os.environ.get("SLURM_NTASKS", "1")),
        "gpus_per_node": os.environ.get("SLURM_GPUS_PER_NODE", "0"),
        "nodelist": os.environ.get("SLURM_NODELIST", ""),
        "procid": int(os.environ.get("SLURM_PROCID", "0")),
        "localid": int(os.environ.get("SLURM_LOCALID", "0")),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "account": os.environ.get("SLURM_JOB_ACCOUNT", ""),
        "submit_dir": os.environ.get("SLURM_SUBMIT_DIR", ""),
    }


def get_master_addr() -> str:
    """Get master node address from SLURM environment."""
    if not detect_slurm_environment():
        return "localhost"

    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        return "localhost"

    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            check=True
        )
        nodes = result.stdout.strip().split("\n")
        if nodes:
            master_node = nodes[0]
            try:
                return socket.gethostbyname(master_node)
            except socket.gaierror:
                return master_node
    except Exception as e:
        logger.warning(f"Failed to get master address: {e}")

    return "localhost"


def create_checkpoint_signal_handler(checkpoint_func):
    """
    Create signal handler for SLURM job preemption.

    SLURM sends SIGTERM before preempting/timing out.
    This handler saves a checkpoint before exit.
    """
    import signal

    def handler(signum, frame):
        logger.warning(f"Received signal {signum}, saving checkpoint...")
        try:
            checkpoint_func()
            logger.info("Checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
        finally:
            # Re-raise to allow normal termination
            signal.default_int_handler(signum, frame)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGUSR1, handler)  # SLURM requeue signal

    logger.info("Checkpoint signal handlers installed")
