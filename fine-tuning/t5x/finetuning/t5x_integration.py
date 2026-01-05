#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5X Integration - Integración con T5X para entrenamiento optimizado en TPU.
"""

import logging
import json
import os
import subprocess
import yaml
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile

logger = logging.getLogger(__name__)


class T5XBackend(Enum):
    """Backends de T5X."""
    XMANAGER = "xmanager"
    VERTEX_AI = "vertex_ai"
    GCE_TPU = "gce_tpu"
    LOCAL_TPU = "local_tpu"


class T5XModelSize(Enum):
    """Tamaños de modelo T5X."""
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    XL = "xl"
    XXL = "xxl"
    CUSTOM = "custom"


@dataclass
class T5XConfig:
    """Configuración T5X."""
    model_size: T5XModelSize
    backend: T5XBackend
    tpu_type: str
    tpu_cores: int
    project_id: str
    zone: str
    gin_configs: List[str]
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    checkpoint_config: Dict[str, Any]


@dataclass
class T5XJob:
    """Job de entrenamiento T5X."""
    job_id: str
    config: T5XConfig
    status: str  # pending, running, completed, failed
    start_time: datetime
    end_time: Optional[datetime]
    logs: List[str]
    checkpoints: List[str]
    metrics: Dict[str, Any]


class T5XManager:
    """Gestor de entrenamiento T5X."""
    
    def __init__(self, 
                 t5x_dir: str = "t5x",
                 configs_dir: str = "backend/data/t5x_configs",
                 jobs_dir: str = "backend/data/t5x_jobs"):
        self.t5x_dir = t5x_dir
        self.configs_dir = configs_dir
        self.jobs_dir = jobs_dir
        
        # Configuración de infraestructura
        self.infrastructure_config = self._load_infrastructure_config()
        
        # Jobs activos
        self.active_jobs: Dict[str, T5XJob] = {}
        
        # Estadísticas
        self.t5x_stats = {
            'total_jobs_created': 0,
            'total_jobs_completed': 0,
            'total_tpu_hours': 0.0,
            'average_training_time_hours': 0.0
        }
        
        # Asegurar directorios
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)
        
        logger.info(f"T5XManager inicializado: t5x_dir={t5x_dir}")
    
    def _load_infrastructure_config(self) -> Dict[str, Any]:
        """Carga configuración de infraestructura T5X."""
        return {
            'google_cloud': {
                'project_id': 'mamba-001',
                'zone': 'europe-southwest1-b',
                'tpu_v5e_64': {
                    'name': 'tpu-v5e-64',
                    'tpu_type': 'v5e-64',
                    'tpu_cores': 64,
                    'zone': 'europe-southwest1-b',
                    'accelerator_type': 'TPU_V5E'
                },
                'tpu_v4_32': {
                    'name': 'tpu-v4-32',
                    'tpu_type': 'v4-32',
                    'tpu_cores': 32,
                    'zone': 'europe-southwest1-b',
                    'accelerator_type': 'TPU_V4'
                }
            },
            't5x_models': {
                'small': {
                    'gin_file': 't5x/examples/t5/t5_1_1/small.gin',
                    'tpu_cores_required': 8,
                    'memory_gb': 16
                },
                'base': {
                    'gin_file': 't5x/examples/t5/t5_1_1/base.gin',
                    'tpu_cores_required': 16,
                    'memory_gb': 32
                },
                'large': {
                    'gin_file': 't5x/examples/t5/t5_1_1/large.gin',
                    'tpu_cores_required': 32,
                    'memory_gb': 64
                },
                'xl': {
                    'gin_file': 't5x/examples/t5/t5_1_1/xl.gin',
                    'tpu_cores_required': 64,
                    'memory_gb': 128
                },
                'xxl': {
                    'gin_file': 't5x/examples/t5/t5_1_1/xxl.gin',
                    'tpu_cores_required': 128,
                    'memory_gb': 256
                }
            }
        }
    
    def create_t5x_config(self, 
                         model_size: T5XModelSize,
                         backend: T5XBackend = T5XBackend.XMANAGER,
                         custom_gin_configs: Optional[List[str]] = None) -> T5XConfig:
        """Crea configuración T5X."""
        try:
            # Configuración base
            tpu_config = self.infrastructure_config['google_cloud']['tpu_v5e_64']
            model_config = self.infrastructure_config['t5x_models'][model_size.value]
            
            # Configurar gin files
            gin_configs = custom_gin_configs or [model_config['gin_file']]
            
            # Configuración de dataset
            dataset_config = {
                'mixture_or_task_name': 'capibara6_moe',
                'tfds_data_dir': 'gs://capibara6-data/tfds',
                'sequence_length': {'inputs': 2048, 'targets': 2048},
                'batch_size': 8,
                'shuffle_buffer_size': 10000
            }
            
            # Configuración de entrenamiento
            training_config = {
                'num_train_steps': 1000000,
                'num_eval_steps': 1000,
                'save_checkpoint_freq': 10000,
                'eval_freq': 5000,
                'learning_rate': 0.001,
                'warmup_steps': 10000,
                'weight_decay': 0.01,
                'gradient_clipping': 1.0,
                'dtype': 'bfloat16',
                'optimizer': 'adafactor'
            }
            
            # Configuración de checkpointing
            checkpoint_config = {
                'save_dir': 'gs://capibara6-models/checkpoints',
                'restore_dir': None,
                'restore_checkpoint_cfg': {
                    'mode': 'specific',
                    'path': None
                },
                'checkpoint_period': 10000,
                'keep_checkpoint_max': 5
            }
            
            return T5XConfig(
                model_size=model_size,
                backend=backend,
                tpu_type=tpu_config['tpu_type'],
                tpu_cores=tpu_config['tpu_cores'],
                project_id=self.infrastructure_config['google_cloud']['project_id'],
                zone=tpu_config['zone'],
                gin_configs=gin_configs,
                dataset_config=dataset_config,
                training_config=training_config,
                checkpoint_config=checkpoint_config
            )
            
        except Exception as e:
            logger.error(f"Error creando configuración T5X: {e}")
            raise
    
    def create_t5x_job(self, 
                      config: T5XConfig,
                      job_name: Optional[str] = None) -> T5XJob:
        """Crea job de entrenamiento T5X."""
        job_id = job_name or f"t5x_{config.model_size.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = T5XJob(
            job_id=job_id,
            config=config,
            status="pending",
            start_time=datetime.now(),
            end_time=None,
            logs=[],
            checkpoints=[],
            metrics={}
        )
        
        # Guardar job
        self._save_job(job)
        
        # Agregar a jobs activos
        self.active_jobs[job_id] = job
        
        self.t5x_stats['total_jobs_created'] += 1
        
        logger.info(f"Job T5X creado: {job_id}")
        return job
    
    def start_t5x_job(self, job_id: str) -> bool:
        """Inicia job de entrenamiento T5X."""
        try:
            if job_id not in self.active_jobs:
                logger.error(f"Job T5X no encontrado: {job_id}")
                return False
            
            job = self.active_jobs[job_id]
            
            if job.status != "pending":
                logger.error(f"Job T5X no está en estado pending: {job.status}")
                return False
            
            # Generar comando de entrenamiento
            training_command = self._generate_t5x_command(job)
            
            # Ejecutar comando
            self._execute_t5x_command(job, training_command)
            
            # Actualizar estado
            job.status = "running"
            job.start_time = datetime.now()
            
            # Guardar job actualizado
            self._save_job(job)
            
            logger.info(f"Job T5X iniciado: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando job T5X {job_id}: {e}")
            return False
    
    def _generate_t5x_command(self, job: T5XJob) -> str:
        """Genera comando de entrenamiento T5X."""
        config = job.config
        
        if config.backend == T5XBackend.XMANAGER:
            return self._generate_xmanager_command(job)
        elif config.backend == T5XBackend.VERTEX_AI:
            return self._generate_vertex_ai_command(job)
        elif config.backend == T5XBackend.GCE_TPU:
            return self._generate_gce_tpu_command(job)
        else:
            raise ValueError(f"Backend T5X no soportado: {config.backend}")
    
    def _generate_xmanager_command(self, job: T5XJob) -> str:
        """Genera comando XManager."""
        config = job.config
        
        # Crear archivo de configuración XManager
        xmanager_config = {
            'experiment': {
                'name': job.job_id,
                'owner': 'capibara6',
                'description': f'T5X training for {config.model_size.value} model'
            },
            'platform': {
                'type': 'vertex_ai',
                'project_id': config.project_id,
                'zone': config.zone
            },
            'tpu': {
                'type': config.tpu_type,
                'cores': config.tpu_cores,
                'accelerator_type': 'TPU_V5E'
            },
            'training': {
                'gin_configs': config.gin_configs,
                'dataset_config': config.dataset_config,
                'training_config': config.training_config,
                'checkpoint_config': config.checkpoint_config
            }
        }
        
        # Guardar configuración XManager
        xmanager_config_path = os.path.join(self.configs_dir, f"{job.job_id}_xmanager.yaml")
        with open(xmanager_config_path, 'w') as f:
            yaml.dump(xmanager_config, f, default_flow_style=False)
        
        # Comando XManager
        command = f"""
        xmanager launch {xmanager_config_path} \\
            --xm_wrap_late_binding \\
            --xm_resource_allocator_group=vertex_ai \\
            --xm_platform=vertex_ai \\
            --xm_resource_allocator_project_id={config.project_id} \\
            --xm_resource_allocator_zone={config.zone}
        """
        
        return command.strip()
    
    def _generate_vertex_ai_command(self, job: T5XJob) -> str:
        """Genera comando Vertex AI."""
        config = job.config
        
        # Crear archivo de configuración Vertex AI
        vertex_config = {
            'display_name': job.job_id,
            'project': config.project_id,
            'location': config.zone,
            'worker_pool_specs': [{
                'machine_spec': {
                    'machine_type': 'cloud-tpu',
                    'accelerator_type': 'TPU_V5E',
                    'accelerator_count': config.tpu_cores
                },
                'replica_count': 1,
                'container_spec': {
                    'image_uri': 'gcr.io/t5x/t5x:latest',
                    'command': self._generate_t5x_training_command(job)
                }
            }]
        }
        
        # Guardar configuración Vertex AI
        vertex_config_path = os.path.join(self.configs_dir, f"{job.job_id}_vertex.yaml")
        with open(vertex_config_path, 'w') as f:
            yaml.dump(vertex_config, f, default_flow_style=False)
        
        # Comando Vertex AI
        command = f"""
        gcloud ai custom-jobs create \\
            --region={config.zone} \\
            --config={vertex_config_path}
        """
        
        return command.strip()
    
    def _generate_gce_tpu_command(self, job: T5XJob) -> str:
        """Genera comando GCE TPU."""
        config = job.config
        
        # Comando GCE TPU
        command = f"""
        gcloud compute tpus tpu-vm create {job.job_id} \\
            --zone={config.zone} \\
            --accelerator-type={config.tpu_type} \\
            --version=tpu-vm-base \\
            --project={config.project_id}
        
        gcloud compute tpus tpu-vm ssh {job.job_id} \\
            --zone={config.zone} \\
            --project={config.project_id} \\
            --command="{self._generate_t5x_training_command(job)}"
        """
        
        return command.strip()
    
    def _generate_t5x_training_command(self, job: T5XJob) -> str:
        """Genera comando de entrenamiento T5X."""
        config = job.config
        
        # Construir argumentos gin
        gin_args = []
        for gin_file in config.gin_configs:
            gin_args.append(f"--gin_file={gin_file}")
        
        # Agregar configuraciones específicas
        gin_args.extend([
            f"--gin.MIXTURE_OR_TASK_NAME=\"{config.dataset_config['mixture_or_task_name']}\"",
            f"--gin.TFDS_DATA_DIR=\"{config.dataset_config['tfds_data_dir']}\"",
            f"--gin.MODEL_DIR=\"{config.checkpoint_config['save_dir']}/{job.job_id}\"",
            f"--gin.NUM_TRAIN_STEPS={config.training_config['num_train_steps']}",
            f"--gin.NUM_EVAL_STEPS={config.training_config['num_eval_steps']}",
            f"--gin.SAVE_CHECKPOINT_FREQ={config.training_config['save_checkpoint_freq']}",
            f"--gin.EVAL_FREQ={config.training_config['eval_freq']}",
            f"--gin.LEARNING_RATE={config.training_config['learning_rate']}",
            f"--gin.WARMUP_STEPS={config.training_config['warmup_steps']}",
            f"--gin.WEIGHT_DECAY={config.training_config['weight_decay']}",
            f"--gin.GRADIENT_CLIPPING={config.training_config['gradient_clipping']}",
            f"--gin.DTYPE=\"{config.training_config['dtype']}\"",
            f"--gin.OPTIMIZER=\"{config.training_config['optimizer']}\""
        ])
        
        # Comando T5X
        command = f"""
        python3 {self.t5x_dir}/t5x/train.py \\
            {' '.join(gin_args)}
        """
        
        return command.strip()
    
    def _execute_t5x_command(self, job: T5XJob, command: str):
        """Ejecuta comando T5X."""
        try:
            # Crear archivo de log
            log_file = os.path.join(self.jobs_dir, f"{job.job_id}.log")
            
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
            
            logger.info(f"Comando T5X ejecutado para job {job.job_id}: PID {process.pid}")
            
        except Exception as e:
            logger.error(f"Error ejecutando comando T5X: {e}")
            job.status = "failed"
            job.logs.append(f"Error ejecutando comando: {e}")
    
    def monitor_t5x_job(self, job_id: str) -> Dict[str, Any]:
        """Monitorea job T5X."""
        try:
            if job_id not in self.active_jobs:
                return {'error': f'Job T5X no encontrado: {job_id}'}
            
            job = self.active_jobs[job_id]
            
            # Leer logs recientes
            log_file = os.path.join(self.jobs_dir, f"{job.job_id}.log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    recent_logs = f.readlines()[-50:]  # Últimas 50 líneas
                    job.logs.extend(recent_logs)
            
            # Simular métricas (en implementación real, leer de TPU/monitoring)
            if job.status == "running":
                # Simular progreso de entrenamiento
                job.metrics['current_step'] = job.metrics.get('current_step', 0) + 100
                job.metrics['loss'] = max(job.metrics.get('loss', 2.0) - 0.001, 0.1)
                job.metrics['learning_rate'] = 0.001
                job.metrics['tpu_utilization'] = 85.0
            
            # Verificar si el job terminó
            if job.metrics.get('current_step', 0) >= job.config.training_config['num_train_steps']:
                job.status = "completed"
                job.end_time = datetime.now()
                self.t5x_stats['total_jobs_completed'] += 1
                
                # Calcular tiempo de entrenamiento
                training_time = (job.end_time - job.start_time).total_seconds() / 3600
                self.t5x_stats['total_tpu_hours'] += training_time
            
            # Guardar job actualizado
            self._save_job(job)
            
            return {
                'job_id': job.job_id,
                'status': job.status,
                'current_step': job.metrics.get('current_step', 0),
                'total_steps': job.config.training_config['num_train_steps'],
                'loss': job.metrics.get('loss', 0.0),
                'learning_rate': job.metrics.get('learning_rate', 0.0),
                'tpu_utilization': job.metrics.get('tpu_utilization', 0.0),
                'recent_logs': job.logs[-10:] if job.logs else []
            }
            
        except Exception as e:
            logger.error(f"Error monitoreando job T5X {job_id}: {e}")
            return {'error': str(e)}
    
    def stop_t5x_job(self, job_id: str) -> bool:
        """Detiene job T5X."""
        try:
            if job_id not in self.active_jobs:
                logger.error(f"Job T5X no encontrado: {job_id}")
                return False
            
            job = self.active_jobs[job_id]
            
            if job.status not in ["running", "pending"]:
                logger.error(f"Job T5X no está en estado ejecutable: {job.status}")
                return False
            
            # Terminar proceso si existe
            if 'process_id' in job.metrics:
                try:
                    os.kill(job.metrics['process_id'], 9)  # SIGKILL
                    logger.info(f"Proceso T5X terminado: {job.metrics['process_id']}")
                except:
                    pass
            
            # Actualizar estado
            job.status = "cancelled"
            job.end_time = datetime.now()
            
            # Guardar job
            self._save_job(job)
            
            logger.info(f"Job T5X detenido: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deteniendo job T5X {job_id}: {e}")
            return False
    
    def _save_job(self, job: T5XJob):
        """Guarda job T5X en archivo."""
        try:
            job_file = os.path.join(self.jobs_dir, f"{job.job_id}.json")
            
            # Convertir a diccionario
            job_dict = asdict(job)
            
            # Convertir datetime a string
            job_dict['start_time'] = job.start_time.isoformat()
            if job.end_time:
                job_dict['end_time'] = job.end_time.isoformat()
            
            # Convertir enums a string
            job_dict['config']['model_size'] = job.config.model_size.value
            job_dict['config']['backend'] = job.config.backend.value
            
            with open(job_file, 'w', encoding='utf-8') as f:
                json.dump(job_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error guardando job T5X {job.job_id}: {e}")
    
    def get_t5x_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas T5X."""
        return self.t5x_stats.copy()
    
    def list_active_jobs(self) -> List[str]:
        """Lista jobs T5X activos."""
        return list(self.active_jobs.keys())
    
    def create_custom_gin_config(self, 
                               job_id: str,
                               model_size: T5XModelSize,
                               custom_config: Dict[str, Any]) -> str:
        """Crea configuración gin personalizada."""
        try:
            # Configuración base
            base_config = self.infrastructure_config['t5x_models'][model_size.value]
            
            # Crear configuración gin personalizada
            gin_config = f"""
# Configuración T5X personalizada para {job_id}
# Generada el: {datetime.now().isoformat()}

include "{base_config['gin_file']}"

# Configuraciones personalizadas
MIXTURE_OR_TASK_NAME = "{custom_config.get('mixture_or_task_name', 'capibara6_moe')}"
TFDS_DATA_DIR = "{custom_config.get('tfds_data_dir', 'gs://capibara6-data/tfds')}"
MODEL_DIR = "{custom_config.get('model_dir', 'gs://capibara6-models')}"

# Configuración de entrenamiento
NUM_TRAIN_STEPS = {custom_config.get('num_train_steps', 1000000)}
NUM_EVAL_STEPS = {custom_config.get('num_eval_steps', 1000)}
SAVE_CHECKPOINT_FREQ = {custom_config.get('save_checkpoint_freq', 10000)}
EVAL_FREQ = {custom_config.get('eval_freq', 5000)}

# Configuración de optimización
LEARNING_RATE = {custom_config.get('learning_rate', 0.001)}
WARMUP_STEPS = {custom_config.get('warmup_steps', 10000)}
WEIGHT_DECAY = {custom_config.get('weight_decay', 0.01)}
GRADIENT_CLIPPING = {custom_config.get('gradient_clipping', 1.0)}

# Configuración de datos
SEQUENCE_LENGTH = {custom_config.get('sequence_length', {'inputs': 2048, 'targets': 2048})}
BATCH_SIZE = {custom_config.get('batch_size', 8)}
SHUFFLE_BUFFER_SIZE = {custom_config.get('shuffle_buffer_size', 10000)}

# Configuración de modelo
DTYPE = "{custom_config.get('dtype', 'bfloat16')}"
OPTIMIZER = "{custom_config.get('optimizer', 'adafactor')}"
"""
            
            # Guardar configuración gin
            gin_file_path = os.path.join(self.configs_dir, f"{job_id}_custom.gin")
            with open(gin_file_path, 'w') as f:
                f.write(gin_config)
            
            logger.info(f"Configuración gin personalizada creada: {gin_file_path}")
            return gin_file_path
            
        except Exception as e:
            logger.error(f"Error creando configuración gin personalizada: {e}")
            return ""


if __name__ == "__main__":
    # Test del T5XManager
    logging.basicConfig(level=logging.INFO)
    
    manager = T5XManager()
    
    # Crear configuración T5X para modelo base
    t5x_config = manager.create_t5x_config(
        model_size=T5XModelSize.BASE,
        backend=T5XBackend.XMANAGER
    )
    
    print(f"Configuración T5X creada: {t5x_config.model_size.value}")
    print(f"Backend: {t5x_config.backend.value}")
    print(f"TPU: {t5x_config.tpu_type} ({t5x_config.tpu_cores} cores)")
    
    # Crear job T5X
    job = manager.create_t5x_job(t5x_config, "test_t5x_base")
    
    print(f"Job T5X creado: {job.job_id}")
    print(f"Estado: {job.status}")
    
    # Crear configuración gin personalizada
    custom_config = {
        'mixture_or_task_name': 'capibara6_python_expert',
        'num_train_steps': 500000,
        'learning_rate': 0.0005,
        'batch_size': 16
    }
    
    gin_file = manager.create_custom_gin_config(
        job.job_id,
        T5XModelSize.BASE,
        custom_config
    )
    
    print(f"Configuración gin personalizada: {gin_file}")
    
    # Mostrar estadísticas
    stats = manager.get_t5x_stats()
    print(f"Estadísticas T5X: {stats}")
