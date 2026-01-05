#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Training - Sistema de entrenamiento distribuido para fine-tuning.
"""

import logging
import json
import os
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class TrainingBackend(Enum):
    """Backends de entrenamiento distribuido."""
    DEEPSPEED = "deepspeed"
    FAIRSCALE = "fairscale"
    HUGGINGFACE = "huggingface"
    PYTORCH_DDP = "pytorch_ddp"
    TORCHRUN = "torchrun"


class InfrastructureType(Enum):
    """Tipos de infraestructura."""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    MULTI_NODE = "multi_node"
    TPU = "tpu"
    GOOGLE_CLOUD = "google_cloud"


@dataclass
class NodeConfig:
    """Configuración de nodo."""
    node_id: str
    hostname: str
    ip_address: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_count: int
    ram_gb: int
    is_master: bool = False
    port: int = 29500


@dataclass
class DistributedTrainingConfig:
    """Configuración de entrenamiento distribuido."""
    backend: TrainingBackend
    infrastructure: InfrastructureType
    nodes: List[NodeConfig]
    world_size: int
    local_rank: int
    global_rank: int
    master_addr: str
    master_port: int
    backend_config: Dict[str, Any]
    checkpointing_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]


@dataclass
class TrainingJob:
    """Job de entrenamiento."""
    job_id: str
    config_name: str
    distributed_config: DistributedTrainingConfig
    start_time: datetime
    status: str  # pending, running, completed, failed, cancelled
    progress: float
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    loss: float
    learning_rate: float
    gpu_utilization: Dict[str, float]
    memory_usage: Dict[str, float]
    logs: List[str]
    checkpoints: List[str]
    metrics: Dict[str, Any]


class DistributedTrainingManager:
    """Gestor de entrenamiento distribuido."""
    
    def __init__(self, 
                 jobs_dir: str = "backend/data/training_jobs",
                 checkpoints_dir: str = "backend/models/checkpoints",
                 logs_dir: str = "backend/logs/training"):
        self.jobs_dir = jobs_dir
        self.checkpoints_dir = checkpoints_dir
        self.logs_dir = logs_dir
        
        # Configuración de infraestructura
        self.infrastructure_config = self._load_infrastructure_config()
        
        # Jobs activos
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        # Estadísticas
        self.training_stats = {
            'total_jobs_created': 0,
            'total_jobs_completed': 0,
            'total_jobs_failed': 0,
            'total_training_hours': 0.0,
            'total_gpu_hours': 0.0,
            'average_training_time_hours': 0.0
        }
        
        # Asegurar directorios
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        logger.info(f"DistributedTrainingManager inicializado: jobs_dir={jobs_dir}")
    
    def _load_infrastructure_config(self) -> Dict[str, Any]:
        """Carga configuración de infraestructura."""
        try:
            # Configuración para Google Cloud
            config = {
                'google_cloud': {
                    'project_id': 'mamba-001',
                    'zone': 'europe-southwest1-b',
                    'vm_20b': {
                        'name': 'gpt-oss-20b',
                        'machine_type': 'c2-standard-32',  # ARM Axion equivalent
                        'gpu_count': 0,
                        'cpu_count': 32,
                        'ram_gb': 64
                    },
                    'vm_120b': {
                        'name': 'gpt-oss-120b',
                        'machine_type': 'a2-highgpu-2g',  # 2x H100
                        'gpu_count': 2,
                        'gpu_type': 'nvidia-h100',
                        'gpu_memory_gb': 80,
                        'cpu_count': 24,
                        'ram_gb': 170
                    },
                    'tpu_training': {
                        'name': 'tpu-v5e-64',
                        'tpu_type': 'v5e-64',
                        'tpu_cores': 64,
                        'zone': 'europe-southwest1-b'
                    }
                },
                'local': {
                    'single_gpu': {
                        'gpu_count': 1,
                        'gpu_memory_gb': 24,
                        'cpu_count': 8,
                        'ram_gb': 32
                    }
                }
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración de infraestructura: {e}")
            return {}
    
    def create_distributed_config(self, 
                                model_size: str,
                                infrastructure: InfrastructureType,
                                backend: TrainingBackend = TrainingBackend.DEEPSPEED) -> DistributedTrainingConfig:
        """Crea configuración distribuida."""
        try:
            if infrastructure == InfrastructureType.GOOGLE_CLOUD:
                return self._create_google_cloud_config(model_size, backend)
            elif infrastructure == InfrastructureType.MULTI_GPU:
                return self._create_multi_gpu_config(model_size, backend)
            elif infrastructure == InfrastructureType.SINGLE_GPU:
                return self._create_single_gpu_config(model_size, backend)
            elif infrastructure == InfrastructureType.TPU:
                return self._create_tpu_config(model_size, backend)
            else:
                raise ValueError(f"Infraestructura no soportada: {infrastructure}")
                
        except Exception as e:
            logger.error(f"Error creando configuración distribuida: {e}")
            raise
    
    def _create_google_cloud_config(self, model_size: str, backend: TrainingBackend) -> DistributedTrainingConfig:
        """Crea configuración para Google Cloud."""
        if model_size == "20b":
            vm_config = self.infrastructure_config['google_cloud']['vm_20b']
            nodes = [
                NodeConfig(
                    node_id="gcp-20b-001",
                    hostname=vm_config['name'],
                    ip_address="10.0.0.2",  # IP interna
                    gpu_count=vm_config['gpu_count'],
                    gpu_memory_gb=0,
                    cpu_count=vm_config['cpu_count'],
                    ram_gb=vm_config['ram_gb'],
                    is_master=True
                )
            ]
            world_size = 1
        elif model_size == "120b":
            vm_config = self.infrastructure_config['google_cloud']['vm_120b']
            nodes = [
                NodeConfig(
                    node_id="gcp-120b-001",
                    hostname=vm_config['name'],
                    ip_address="10.0.0.3",  # IP interna
                    gpu_count=vm_config['gpu_count'],
                    gpu_memory_gb=vm_config['gpu_memory_gb'],
                    cpu_count=vm_config['cpu_count'],
                    ram_gb=vm_config['ram_gb'],
                    is_master=True
                )
            ]
            world_size = vm_config['gpu_count']
        else:
            raise ValueError(f"Tamaño de modelo no soportado: {model_size}")
        
        # Configuración específica del backend
        if backend == TrainingBackend.DEEPSPEED:
            backend_config = {
                "deepspeed_config": {
                    "train_batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "optimizer": {
                        "type": "AdamW",
                        "params": {
                            "lr": 2e-4,
                            "betas": [0.9, 0.999],
                            "eps": 1e-8,
                            "weight_decay": 0.01
                        }
                    },
                    "scheduler": {
                        "type": "WarmupLR",
                        "params": {
                            "warmup_min_lr": 0,
                            "warmup_max_lr": 2e-4,
                            "warmup_num_steps": 100
                        }
                    },
                    "fp16": {
                        "enabled": True,
                        "auto_cast": False,
                        "loss_scale": 0,
                        "initial_scale_power": 16,
                        "loss_scale_window": 1000,
                        "hysteresis": 2,
                        "min_loss_scale": 1
                    },
                    "zero_optimization": {
                        "stage": 2,
                        "allgather_partitions": True,
                        "allgather_bucket_size": 2e8,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 2e8,
                        "contiguous_gradients": True
                    },
                    "activation_checkpointing": {
                        "partition_activations": True,
                        "cpu_checkpointing": True,
                        "contiguous_memory_optimization": False,
                        "number_checkpoints": 4,
                        "synchronize_checkpoint_boundary": False,
                        "profile": False
                    }
                }
            }
        else:
            backend_config = {}
        
        return DistributedTrainingConfig(
            backend=backend,
            infrastructure=InfrastructureType.GOOGLE_CLOUD,
            nodes=nodes,
            world_size=world_size,
            local_rank=0,
            global_rank=0,
            master_addr=nodes[0].ip_address,
            master_port=29500,
            backend_config=backend_config,
            checkpointing_config=self._create_checkpointing_config(),
            monitoring_config=self._create_monitoring_config()
        )
    
    def _create_multi_gpu_config(self, model_size: str, backend: TrainingBackend) -> DistributedTrainingConfig:
        """Crea configuración para múltiples GPUs."""
        # Configuración para 2 GPUs locales
        nodes = [
            NodeConfig(
                node_id="local-gpu-001",
                hostname="localhost",
                ip_address="127.0.0.1",
                gpu_count=2,
                gpu_memory_gb=24,
                cpu_count=16,
                ram_gb=64,
                is_master=True
            )
        ]
        
        backend_config = {
            "ddp_config": {
                "backend": "nccl",
                "init_method": "env://",
                "world_size": 2,
                "rank": 0
            }
        }
        
        return DistributedTrainingConfig(
            backend=backend,
            infrastructure=InfrastructureType.MULTI_GPU,
            nodes=nodes,
            world_size=2,
            local_rank=0,
            global_rank=0,
            master_addr="127.0.0.1",
            master_port=29500,
            backend_config=backend_config,
            checkpointing_config=self._create_checkpointing_config(),
            monitoring_config=self._create_monitoring_config()
        )
    
    def _create_single_gpu_config(self, model_size: str, backend: TrainingBackend) -> DistributedTrainingConfig:
        """Crea configuración para GPU única."""
        nodes = [
            NodeConfig(
                node_id="local-single-001",
                hostname="localhost",
                ip_address="127.0.0.1",
                gpu_count=1,
                gpu_memory_gb=24,
                cpu_count=8,
                ram_gb=32,
                is_master=True
            )
        ]
        
        backend_config = {}
        
        return DistributedTrainingConfig(
            backend=backend,
            infrastructure=InfrastructureType.SINGLE_GPU,
            nodes=nodes,
            world_size=1,
            local_rank=0,
            global_rank=0,
            master_addr="127.0.0.1",
            master_port=29500,
            backend_config=backend_config,
            checkpointing_config=self._create_checkpointing_config(),
            monitoring_config=self._create_monitoring_config()
        )
    
    def _create_tpu_config(self, model_size: str, backend: TrainingBackend) -> DistributedTrainingConfig:
        """Crea configuración para TPU."""
        tpu_config = self.infrastructure_config['google_cloud']['tpu_training']
        
        nodes = [
            NodeConfig(
                node_id="tpu-v5e-001",
                hostname=tpu_config['name'],
                ip_address="10.0.0.4",
                gpu_count=0,  # TPU no tiene GPUs
                gpu_memory_gb=0,
                cpu_count=tpu_config['tpu_cores'],
                ram_gb=256,  # TPU tiene mucha RAM
                is_master=True
            )
        ]
        
        backend_config = {
            "tpu_config": {
                "tpu_cores": tpu_config['tpu_cores'],
                "tpu_type": tpu_config['tpu_type'],
                "zone": tpu_config['zone']
            }
        }
        
        return DistributedTrainingConfig(
            backend=backend,
            infrastructure=InfrastructureType.TPU,
            nodes=nodes,
            world_size=tpu_config['tpu_cores'],
            local_rank=0,
            global_rank=0,
            master_addr=nodes[0].ip_address,
            master_port=29500,
            backend_config=backend_config,
            checkpointing_config=self._create_checkpointing_config(),
            monitoring_config=self._create_monitoring_config()
        )
    
    def _create_checkpointing_config(self) -> Dict[str, Any]:
        """Crea configuración de checkpointing."""
        return {
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "checkpointing_backend": "local",
            "checkpoint_compression": True,
            "checkpoint_encryption": False
        }
    
    def _create_monitoring_config(self) -> Dict[str, Any]:
        """Crea configuración de monitoreo."""
        return {
            "enable_wandb": True,
            "wandb_project": "capibara6-finetuning",
            "log_level": "INFO",
            "log_steps": 10,
            "eval_steps": 500,
            "save_steps": 500,
            "monitoring_interval": 30,  # segundos
            "metrics_to_track": [
                "loss", "learning_rate", "epoch", "step",
                "gpu_utilization", "memory_usage", "throughput"
            ]
        }
    
    def create_training_job(self, 
                          config_name: str,
                          distributed_config: DistributedTrainingConfig,
                          training_script_path: str) -> TrainingJob:
        """Crea job de entrenamiento."""
        job_id = f"job_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = TrainingJob(
            job_id=job_id,
            config_name=config_name,
            distributed_config=distributed_config,
            start_time=datetime.now(),
            status="pending",
            progress=0.0,
            current_epoch=0,
            total_epochs=0,
            current_step=0,
            total_steps=0,
            loss=0.0,
            learning_rate=0.0,
            gpu_utilization={},
            memory_usage={},
            logs=[],
            checkpoints=[],
            metrics={}
        )
        
        # Guardar job
        self._save_job(job)
        
        # Agregar a jobs activos
        self.active_jobs[job_id] = job
        
        self.training_stats['total_jobs_created'] += 1
        
        logger.info(f"Job de entrenamiento creado: {job_id}")
        return job
    
    def start_training_job(self, job_id: str) -> bool:
        """Inicia job de entrenamiento."""
        try:
            if job_id not in self.active_jobs:
                logger.error(f"Job no encontrado: {job_id}")
                return False
            
            job = self.active_jobs[job_id]
            
            if job.status != "pending":
                logger.error(f"Job no está en estado pending: {job.status}")
                return False
            
            # Actualizar estado
            job.status = "running"
            job.start_time = datetime.now()
            
            # Generar comando de entrenamiento
            training_command = self._generate_training_command(job)
            
            # Ejecutar comando (en background)
            self._execute_training_command(job, training_command)
            
            # Guardar job actualizado
            self._save_job(job)
            
            logger.info(f"Job de entrenamiento iniciado: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando job {job_id}: {e}")
            return False
    
    def _generate_training_command(self, job: TrainingJob) -> str:
        """Genera comando de entrenamiento."""
        config = job.distributed_config
        
        if config.backend == TrainingBackend.DEEPSPEED:
            return self._generate_deepspeed_command(job)
        elif config.backend == TrainingBackend.TORCHRUN:
            return self._generate_torchrun_command(job)
        elif config.backend == TrainingBackend.HUGGINGFACE:
            return self._generate_huggingface_command(job)
        else:
            raise ValueError(f"Backend no soportado: {config.backend}")
    
    def _generate_deepspeed_command(self, job: TrainingJob) -> str:
        """Genera comando DeepSpeed."""
        config = job.distributed_config
        
        # Crear archivo de configuración DeepSpeed
        deepspeed_config_path = os.path.join(self.jobs_dir, f"{job.job_id}_deepspeed_config.json")
        with open(deepspeed_config_path, 'w') as f:
            json.dump(config.backend_config['deepspeed_config'], f, indent=2)
        
        # Comando DeepSpeed
        command = f"""
        deepspeed --num_gpus={config.world_size} \\
            --master_addr={config.master_addr} \\
            --master_port={config.master_port} \\
            --deepspeed_config={deepspeed_config_path} \\
            backend/scripts/train_{job.config_name}.py
        """
        
        return command.strip()
    
    def _generate_torchrun_command(self, job: TrainingJob) -> str:
        """Genera comando torchrun."""
        config = job.distributed_config
        
        command = f"""
        torchrun --nproc_per_node={config.world_size} \\
            --nnodes=1 \\
            --node_rank=0 \\
            --master_addr={config.master_addr} \\
            --master_port={config.master_port} \\
            backend/scripts/train_{job.config_name}.py
        """
        
        return command.strip()
    
    def _generate_huggingface_command(self, job: TrainingJob) -> str:
        """Genera comando HuggingFace."""
        config = job.distributed_config
        
        command = f"""
        python -m torch.distributed.launch \\
            --nproc_per_node={config.world_size} \\
            --master_addr={config.master_addr} \\
            --master_port={config.master_port} \\
            backend/scripts/train_{job.config_name}.py
        """
        
        return command.strip()
    
    def _execute_training_command(self, job: TrainingJob, command: str):
        """Ejecuta comando de entrenamiento."""
        try:
            # Crear archivo de log
            log_file = os.path.join(self.logs_dir, f"{job.job_id}.log")
            
            # Ejecutar comando en background
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
            
            # Guardar PID del proceso
            job.metrics['process_id'] = process.pid
            
            logger.info(f"Comando de entrenamiento ejecutado para job {job.job_id}: PID {process.pid}")
            
        except Exception as e:
            logger.error(f"Error ejecutando comando de entrenamiento: {e}")
            job.status = "failed"
            job.logs.append(f"Error ejecutando comando: {e}")
    
    def monitor_training_job(self, job_id: str) -> Dict[str, Any]:
        """Monitorea job de entrenamiento."""
        try:
            if job_id not in self.active_jobs:
                return {'error': f'Job no encontrado: {job_id}'}
            
            job = self.active_jobs[job_id]
            
            # Leer logs recientes
            log_file = os.path.join(self.logs_dir, f"{job.job_id}.log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    recent_logs = f.readlines()[-50:]  # Últimas 50 líneas
                    job.logs.extend(recent_logs)
            
            # Simular métricas (en implementación real, leer de GPU/monitoring)
            if job.status == "running":
                job.progress = min(job.progress + 0.01, 1.0)  # Simular progreso
                job.loss = max(job.loss - 0.001, 0.1)  # Simular mejora de loss
                job.gpu_utilization = {"gpu_0": 85.0, "gpu_1": 82.0}  # Simular utilización
                job.memory_usage = {"gpu_0": 18.5, "gpu_1": 19.2}  # Simular memoria
            
            # Verificar si el job terminó
            if job.progress >= 1.0 and job.status == "running":
                job.status = "completed"
                job.end_time = datetime.now()
                self.training_stats['total_jobs_completed'] += 1
                
                # Calcular tiempo de entrenamiento
                training_time = (job.end_time - job.start_time).total_seconds() / 3600
                self.training_stats['total_training_hours'] += training_time
            
            # Guardar job actualizado
            self._save_job(job)
            
            return {
                'job_id': job.job_id,
                'status': job.status,
                'progress': job.progress,
                'current_epoch': job.current_epoch,
                'total_epochs': job.total_epochs,
                'current_step': job.current_step,
                'total_steps': job.total_steps,
                'loss': job.loss,
                'learning_rate': job.learning_rate,
                'gpu_utilization': job.gpu_utilization,
                'memory_usage': job.memory_usage,
                'recent_logs': job.logs[-10:] if job.logs else []
            }
            
        except Exception as e:
            logger.error(f"Error monitoreando job {job_id}: {e}")
            return {'error': str(e)}
    
    def stop_training_job(self, job_id: str) -> bool:
        """Detiene job de entrenamiento."""
        try:
            if job_id not in self.active_jobs:
                logger.error(f"Job no encontrado: {job_id}")
                return False
            
            job = self.active_jobs[job_id]
            
            if job.status not in ["running", "pending"]:
                logger.error(f"Job no está en estado ejecutable: {job.status}")
                return False
            
            # Terminar proceso si existe
            if 'process_id' in job.metrics:
                try:
                    os.kill(job.metrics['process_id'], 9)  # SIGKILL
                    logger.info(f"Proceso terminado: {job.metrics['process_id']}")
                except:
                    pass
            
            # Actualizar estado
            job.status = "cancelled"
            job.end_time = datetime.now()
            
            # Guardar job
            self._save_job(job)
            
            logger.info(f"Job de entrenamiento detenido: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deteniendo job {job_id}: {e}")
            return False
    
    def _save_job(self, job: TrainingJob):
        """Guarda job en archivo."""
        try:
            job_file = os.path.join(self.jobs_dir, f"{job.job_id}.json")
            
            # Convertir a diccionario
            job_dict = asdict(job)
            
            # Convertir datetime a string
            job_dict['start_time'] = job.start_time.isoformat()
            if hasattr(job, 'end_time') and job.end_time:
                job_dict['end_time'] = job.end_time.isoformat()
            
            # Convertir enums a string
            job_dict['distributed_config']['backend'] = job.distributed_config.backend.value
            job_dict['distributed_config']['infrastructure'] = job.distributed_config.infrastructure.value
            
            with open(job_file, 'w', encoding='utf-8') as f:
                json.dump(job_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error guardando job {job.job_id}: {e}")
    
    def load_job(self, job_id: str) -> Optional[TrainingJob]:
        """Carga job desde archivo."""
        try:
            job_file = os.path.join(self.jobs_dir, f"{job_id}.json")
            
            if not os.path.exists(job_file):
                return None
            
            with open(job_file, 'r', encoding='utf-8') as f:
                job_dict = json.load(f)
            
            # Reconstruir job
            job = self._dict_to_job(job_dict)
            
            return job
            
        except Exception as e:
            logger.error(f"Error cargando job {job_id}: {e}")
            return None
    
    def _dict_to_job(self, job_dict: Dict[str, Any]) -> TrainingJob:
        """Convierte diccionario a TrainingJob."""
        # Reconstruir DistributedTrainingConfig
        dist_config_dict = job_dict['distributed_config']
        nodes = [NodeConfig(**node) for node in dist_config_dict['nodes']]
        
        distributed_config = DistributedTrainingConfig(
            backend=TrainingBackend(dist_config_dict['backend']),
            infrastructure=InfrastructureType(dist_config_dict['infrastructure']),
            nodes=nodes,
            world_size=dist_config_dict['world_size'],
            local_rank=dist_config_dict['local_rank'],
            global_rank=dist_config_dict['global_rank'],
            master_addr=dist_config_dict['master_addr'],
            master_port=dist_config_dict['master_port'],
            backend_config=dist_config_dict['backend_config'],
            checkpointing_config=dist_config_dict['checkpointing_config'],
            monitoring_config=dist_config_dict['monitoring_config']
        )
        
        # Reconstruir TrainingJob
        job = TrainingJob(
            job_id=job_dict['job_id'],
            config_name=job_dict['config_name'],
            distributed_config=distributed_config,
            start_time=datetime.fromisoformat(job_dict['start_time']),
            status=job_dict['status'],
            progress=job_dict['progress'],
            current_epoch=job_dict['current_epoch'],
            total_epochs=job_dict['total_epochs'],
            current_step=job_dict['current_step'],
            total_steps=job_dict['total_steps'],
            loss=job_dict['loss'],
            learning_rate=job_dict['learning_rate'],
            gpu_utilization=job_dict['gpu_utilization'],
            memory_usage=job_dict['memory_usage'],
            logs=job_dict['logs'],
            checkpoints=job_dict['checkpoints'],
            metrics=job_dict['metrics']
        )
        
        if 'end_time' in job_dict and job_dict['end_time']:
            job.end_time = datetime.fromisoformat(job_dict['end_time'])
        
        return job
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de entrenamiento."""
        return self.training_stats.copy()
    
    def list_active_jobs(self) -> List[str]:
        """Lista jobs activos."""
        return list(self.active_jobs.keys())
    
    def cleanup_completed_jobs(self, days_old: int = 7) -> int:
        """Limpia jobs completados antiguos."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_count = 0
            
            for job_id, job in list(self.active_jobs.items()):
                if (job.status in ["completed", "failed", "cancelled"] and 
                    hasattr(job, 'end_time') and 
                    job.end_time and 
                    job.end_time < cutoff_date):
                    
                    # Remover de jobs activos
                    del self.active_jobs[job_id]
                    
                    # Opcional: eliminar archivos de log y checkpoint
                    # (comentado para seguridad)
                    # os.remove(os.path.join(self.jobs_dir, f"{job_id}.json"))
                    
                    cleaned_count += 1
            
            logger.info(f"Limpieza completada: {cleaned_count} jobs removidos")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error en limpieza de jobs: {e}")
            return 0


if __name__ == "__main__":
    # Test del DistributedTrainingManager
    logging.basicConfig(level=logging.INFO)
    
    manager = DistributedTrainingManager()
    
    # Crear configuración distribuida para modelo 20B
    dist_config_20b = manager.create_distributed_config(
        model_size="20b",
        infrastructure=InfrastructureType.GOOGLE_CLOUD,
        backend=TrainingBackend.DEEPSPEED
    )
    
    print(f"Configuración 20B creada: {dist_config_20b.world_size} GPUs")
    print(f"Backend: {dist_config_20b.backend.value}")
    print(f"Master: {dist_config_20b.master_addr}:{dist_config_20b.master_port}")
    
    # Crear job de entrenamiento
    job = manager.create_training_job(
        config_name="20b_qlora",
        distributed_config=dist_config_20b,
        training_script_path="backend/scripts/train_20b_qlora.py"
    )
    
    print(f"Job creado: {job.job_id}")
    print(f"Estado: {job.status}")
    
    # Simular monitoreo
    for i in range(3):
        status = manager.monitor_training_job(job.job_id)
        print(f"Monitoreo {i+1}: {status['status']}, progreso: {status['progress']:.2f}")
        time.sleep(1)
    
    # Mostrar estadísticas
    stats = manager.get_training_stats()
    print(f"Estadísticas: {stats}")
