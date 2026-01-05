#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Checkpointing - Sistema de checkpointing mejorado para T5X y fine-tuning.
"""

import logging
import json
import os
import shutil
import hashlib
import gzip
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CheckpointType(Enum):
    """Tipos de checkpoint."""
    FULL_MODEL = "full_model"
    LORA_ADAPTER = "lora_adapter"
    OPTIMIZER_STATE = "optimizer_state"
    TRAINING_STATE = "training_state"
    METADATA = "metadata"
    INCREMENTAL = "incremental"


class CheckpointFormat(Enum):
    """Formatos de checkpoint."""
    T5X_NATIVE = "t5x_native"
    HUGGINGFACE = "huggingface"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUSTOM = "custom"


@dataclass
class CheckpointMetadata:
    """Metadata de checkpoint."""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    format: CheckpointFormat
    model_name: str
    step: int
    epoch: int
    loss: float
    learning_rate: float
    timestamp: datetime
    file_size_bytes: int
    checksum: str
    dependencies: List[str]
    tags: List[str]
    metrics: Dict[str, Any]


@dataclass
class CheckpointConfig:
    """Configuración de checkpointing."""
    save_dir: str
    max_checkpoints: int
    save_frequency: int  # steps
    compression: bool
    encryption: bool
    incremental: bool
    formats: List[CheckpointFormat]
    metadata_backup: bool
    cloud_sync: bool
    retention_days: int


class AdvancedCheckpointManager:
    """Gestor avanzado de checkpoints."""
    
    def __init__(self, 
                 checkpoint_dir: str = "backend/models/checkpoints",
                 config: Optional[CheckpointConfig] = None):
        self.checkpoint_dir = checkpoint_dir
        
        # Configuración por defecto
        if config is None:
            config = CheckpointConfig(
                save_dir=checkpoint_dir,
                max_checkpoints=10,
                save_frequency=1000,
                compression=True,
                encryption=False,
                incremental=True,
                formats=[CheckpointFormat.T5X_NATIVE, CheckpointFormat.HUGGINGFACE],
                metadata_backup=True,
                cloud_sync=True,
                retention_days=30
            )
        
        self.config = config
        
        # Directorios
        self.metadata_dir = os.path.join(checkpoint_dir, "metadata")
        self.backup_dir = os.path.join(checkpoint_dir, "backups")
        self.temp_dir = os.path.join(checkpoint_dir, "temp")
        
        # Asegurar directorios
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Cache de metadatos
        self.metadata_cache: Dict[str, CheckpointMetadata] = {}
        
        # Estadísticas
        self.checkpoint_stats = {
            'total_checkpoints_created': 0,
            'total_checkpoints_restored': 0,
            'total_storage_used_bytes': 0,
            'compression_ratio': 0.0,
            'average_checkpoint_size_bytes': 0
        }
        
        logger.info(f"AdvancedCheckpointManager inicializado: checkpoint_dir={checkpoint_dir}")
    
    def save_checkpoint(self, 
                       model_state: Dict[str, Any],
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       training_state: Optional[Dict[str, Any]] = None,
                       step: int = 0,
                       epoch: int = 0,
                       loss: float = 0.0,
                       learning_rate: float = 0.0,
                       model_name: str = "model",
                       tags: Optional[List[str]] = None) -> str:
        """Guarda checkpoint avanzado."""
        try:
            checkpoint_id = self._generate_checkpoint_id(model_name, step, epoch)
            
            # Crear metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                checkpoint_type=CheckpointType.FULL_MODEL,
                format=CheckpointFormat.T5X_NATIVE,
                model_name=model_name,
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                timestamp=datetime.now(),
                file_size_bytes=0,  # Se calculará después
                checksum="",  # Se calculará después
                dependencies=[],
                tags=tags or [],
                metrics={}
            )
            
            # Guardar en múltiples formatos
            saved_files = []
            
            for format_type in self.config.formats:
                file_path = self._save_checkpoint_format(
                    checkpoint_id, model_state, optimizer_state, 
                    training_state, format_type
                )
                if file_path:
                    saved_files.append(file_path)
            
            # Calcular tamaño total y checksum
            total_size = sum(os.path.getsize(f) for f in saved_files)
            checksum = self._calculate_checksum(saved_files)
            
            metadata.file_size_bytes = total_size
            metadata.checksum = checksum
            
            # Guardar metadata
            self._save_metadata(metadata)
            
            # Actualizar estadísticas
            self._update_stats(metadata)
            
            # Limpiar checkpoints antiguos
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint guardado: {checkpoint_id} ({total_size} bytes)")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Error guardando checkpoint: {e}")
            return ""
    
    def _save_checkpoint_format(self, 
                              checkpoint_id: str,
                              model_state: Dict[str, Any],
                              optimizer_state: Optional[Dict[str, Any]],
                              training_state: Optional[Dict[str, Any]],
                              format_type: CheckpointFormat) -> Optional[str]:
        """Guarda checkpoint en formato específico."""
        try:
            if format_type == CheckpointFormat.T5X_NATIVE:
                return self._save_t5x_checkpoint(checkpoint_id, model_state, optimizer_state, training_state)
            elif format_type == CheckpointFormat.HUGGINGFACE:
                return self._save_huggingface_checkpoint(checkpoint_id, model_state, optimizer_state, training_state)
            elif format_type == CheckpointFormat.PYTORCH:
                return self._save_pytorch_checkpoint(checkpoint_id, model_state, optimizer_state, training_state)
            else:
                logger.warning(f"Formato de checkpoint no soportado: {format_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error guardando checkpoint en formato {format_type}: {e}")
            return None
    
    def _save_t5x_checkpoint(self, 
                           checkpoint_id: str,
                           model_state: Dict[str, Any],
                           optimizer_state: Optional[Dict[str, Any]],
                           training_state: Optional[Dict[str, Any]]) -> str:
        """Guarda checkpoint en formato T5X nativo."""
        checkpoint_dir = os.path.join(self.checkpoint_dir, checkpoint_id, "t5x")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Guardar estado del modelo
        model_file = os.path.join(checkpoint_dir, "model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model_state, f)
        
        # Guardar estado del optimizador si existe
        if optimizer_state:
            optimizer_file = os.path.join(checkpoint_dir, "optimizer.pkl")
            with open(optimizer_file, 'wb') as f:
                pickle.dump(optimizer_state, f)
        
        # Guardar estado de entrenamiento si existe
        if training_state:
            training_file = os.path.join(checkpoint_dir, "training.pkl")
            with open(training_file, 'wb') as f:
                pickle.dump(training_state, f)
        
        # Comprimir si está habilitado
        if self.config.compression:
            compressed_file = f"{checkpoint_dir}.tar.gz"
            shutil.make_archive(checkpoint_dir, 'gztar', checkpoint_dir)
            shutil.rmtree(checkpoint_dir)
            return compressed_file
        
        return checkpoint_dir
    
    def _save_huggingface_checkpoint(self, 
                                   checkpoint_id: str,
                                   model_state: Dict[str, Any],
                                   optimizer_state: Optional[Dict[str, Any]],
                                   training_state: Optional[Dict[str, Any]]) -> str:
        """Guarda checkpoint en formato HuggingFace."""
        checkpoint_dir = os.path.join(self.checkpoint_dir, checkpoint_id, "huggingface")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Convertir estado del modelo a formato HuggingFace
        hf_state = self._convert_to_huggingface_format(model_state)
        
        # Guardar modelo
        model_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        with open(model_file, 'wb') as f:
            pickle.dump(hf_state, f)
        
        # Guardar configuración
        config_file = os.path.join(checkpoint_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self._extract_model_config(model_state), f, indent=2)
        
        # Guardar tokenizer si existe
        if 'tokenizer' in model_state:
            tokenizer_file = os.path.join(checkpoint_dir, "tokenizer.json")
            with open(tokenizer_file, 'w') as f:
                json.dump(model_state['tokenizer'], f, indent=2)
        
        return checkpoint_dir
    
    def _save_pytorch_checkpoint(self, 
                               checkpoint_id: str,
                               model_state: Dict[str, Any],
                               optimizer_state: Optional[Dict[str, Any]],
                               training_state: Optional[Dict[str, Any]]) -> str:
        """Guarda checkpoint en formato PyTorch."""
        checkpoint_dir = os.path.join(self.checkpoint_dir, checkpoint_id, "pytorch")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Guardar checkpoint completo
        checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'training_state': training_state,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pth")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        return checkpoint_dir
    
    def load_checkpoint(self, 
                       checkpoint_id: str,
                       format_type: CheckpointFormat = CheckpointFormat.T5X_NATIVE) -> Optional[Dict[str, Any]]:
        """Carga checkpoint."""
        try:
            # Verificar que el checkpoint existe
            if not self._checkpoint_exists(checkpoint_id):
                logger.error(f"Checkpoint no encontrado: {checkpoint_id}")
                return None
            
            # Cargar metadata
            metadata = self._load_metadata(checkpoint_id)
            if not metadata:
                logger.error(f"Metadata no encontrada para checkpoint: {checkpoint_id}")
                return None
            
            # Cargar checkpoint en formato específico
            checkpoint_data = self._load_checkpoint_format(checkpoint_id, format_type)
            
            if checkpoint_data:
                self.checkpoint_stats['total_checkpoints_restored'] += 1
                logger.info(f"Checkpoint cargado: {checkpoint_id}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error cargando checkpoint {checkpoint_id}: {e}")
            return None
    
    def _load_checkpoint_format(self, 
                              checkpoint_id: str,
                              format_type: CheckpointFormat) -> Optional[Dict[str, Any]]:
        """Carga checkpoint en formato específico."""
        try:
            if format_type == CheckpointFormat.T5X_NATIVE:
                return self._load_t5x_checkpoint(checkpoint_id)
            elif format_type == CheckpointFormat.HUGGINGFACE:
                return self._load_huggingface_checkpoint(checkpoint_id)
            elif format_type == CheckpointFormat.PYTORCH:
                return self._load_pytorch_checkpoint(checkpoint_id)
            else:
                logger.warning(f"Formato de checkpoint no soportado: {format_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error cargando checkpoint en formato {format_type}: {e}")
            return None
    
    def _load_t5x_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Carga checkpoint T5X."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id, "t5x")
        
        # Verificar si está comprimido
        compressed_path = f"{checkpoint_path}.tar.gz"
        if os.path.exists(compressed_path):
            # Descomprimir
            shutil.unpack_archive(compressed_path, self.temp_dir)
            checkpoint_path = os.path.join(self.temp_dir, checkpoint_id, "t5x")
        
        if not os.path.exists(checkpoint_path):
            return None
        
        # Cargar estado del modelo
        model_file = os.path.join(checkpoint_path, "model.pkl")
        if not os.path.exists(model_file):
            return None
        
        with open(model_file, 'rb') as f:
            model_state = pickle.load(f)
        
        # Cargar estado del optimizador si existe
        optimizer_file = os.path.join(checkpoint_path, "optimizer.pkl")
        if os.path.exists(optimizer_file):
            with open(optimizer_file, 'rb') as f:
                optimizer_state = pickle.load(f)
            model_state['optimizer_state'] = optimizer_state
        
        # Cargar estado de entrenamiento si existe
        training_file = os.path.join(checkpoint_path, "training.pkl")
        if os.path.exists(training_file):
            with open(training_file, 'rb') as f:
                training_state = pickle.load(f)
            model_state['training_state'] = training_state
        
        return model_state
    
    def _load_huggingface_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Carga checkpoint HuggingFace."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id, "huggingface")
        
        if not os.path.exists(checkpoint_path):
            return None
        
        # Cargar modelo
        model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(model_file):
            return None
        
        with open(model_file, 'rb') as f:
            model_state = pickle.load(f)
        
        # Cargar configuración
        config_file = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            model_state['config'] = config
        
        return model_state
    
    def _load_pytorch_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Carga checkpoint PyTorch."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id, "pytorch")
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pth")
        
        if not os.path.exists(checkpoint_file):
            return None
        
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def list_checkpoints(self, 
                        model_name: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        limit: int = 10) -> List[CheckpointMetadata]:
        """Lista checkpoints disponibles."""
        try:
            checkpoints = []
            
            # Cargar todos los metadatos
            for filename in os.listdir(self.metadata_dir):
                if filename.endswith('.json'):
                    metadata = self._load_metadata_from_file(os.path.join(self.metadata_dir, filename))
                    if metadata:
                        checkpoints.append(metadata)
            
            # Filtrar por modelo
            if model_name:
                checkpoints = [c for c in checkpoints if c.model_name == model_name]
            
            # Filtrar por tags
            if tags:
                checkpoints = [c for c in checkpoints if any(tag in c.tags for tag in tags)]
            
            # Ordenar por timestamp (más reciente primero)
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            return checkpoints[:limit]
            
        except Exception as e:
            logger.error(f"Error listando checkpoints: {e}")
            return []
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Elimina checkpoint."""
        try:
            # Eliminar directorio del checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            
            # Eliminar archivo comprimido si existe
            compressed_path = f"{checkpoint_path}.tar.gz"
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            
            # Eliminar metadata
            metadata_file = os.path.join(self.metadata_dir, f"{checkpoint_id}.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            # Remover del cache
            if checkpoint_id in self.metadata_cache:
                del self.metadata_cache[checkpoint_id]
            
            logger.info(f"Checkpoint eliminado: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando checkpoint {checkpoint_id}: {e}")
            return False
    
    def _generate_checkpoint_id(self, model_name: str, step: int, epoch: int) -> str:
        """Genera ID único para checkpoint."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{model_name}_step_{step}_epoch_{epoch}_{timestamp}"
    
    def _calculate_checksum(self, file_paths: List[str]) -> str:
        """Calcula checksum de archivos."""
        hasher = hashlib.md5()
        
        for file_path in sorted(file_paths):
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            elif os.path.isdir(file_path):
                for root, dirs, files in os.walk(file_path):
                    for file in sorted(files):
                        file_full_path = os.path.join(root, file)
                        with open(file_full_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _save_metadata(self, metadata: CheckpointMetadata):
        """Guarda metadata de checkpoint."""
        try:
            metadata_file = os.path.join(self.metadata_dir, f"{metadata.checkpoint_id}.json")
            
            # Convertir a diccionario
            metadata_dict = asdict(metadata)
            metadata_dict['timestamp'] = metadata.timestamp.isoformat()
            metadata_dict['checkpoint_type'] = metadata.checkpoint_type.value
            metadata_dict['format'] = metadata.format.value
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            # Actualizar cache
            self.metadata_cache[metadata.checkpoint_id] = metadata
            
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")
    
    def _load_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Carga metadata de checkpoint."""
        try:
            # Verificar cache primero
            if checkpoint_id in self.metadata_cache:
                return self.metadata_cache[checkpoint_id]
            
            # Cargar desde archivo
            metadata_file = os.path.join(self.metadata_dir, f"{checkpoint_id}.json")
            return self._load_metadata_from_file(metadata_file)
            
        except Exception as e:
            logger.error(f"Error cargando metadata: {e}")
            return None
    
    def _load_metadata_from_file(self, metadata_file: str) -> Optional[CheckpointMetadata]:
        """Carga metadata desde archivo."""
        try:
            if not os.path.exists(metadata_file):
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Reconstruir metadata
            metadata = CheckpointMetadata(
                checkpoint_id=metadata_dict['checkpoint_id'],
                checkpoint_type=CheckpointType(metadata_dict['checkpoint_type']),
                format=CheckpointFormat(metadata_dict['format']),
                model_name=metadata_dict['model_name'],
                step=metadata_dict['step'],
                epoch=metadata_dict['epoch'],
                loss=metadata_dict['loss'],
                learning_rate=metadata_dict['learning_rate'],
                timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                file_size_bytes=metadata_dict['file_size_bytes'],
                checksum=metadata_dict['checksum'],
                dependencies=metadata_dict['dependencies'],
                tags=metadata_dict['tags'],
                metrics=metadata_dict['metrics']
            )
            
            # Actualizar cache
            self.metadata_cache[metadata.checkpoint_id] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error cargando metadata desde archivo {metadata_file}: {e}")
            return None
    
    def _checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Verifica si checkpoint existe."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        compressed_path = f"{checkpoint_path}.tar.gz"
        
        return os.path.exists(checkpoint_path) or os.path.exists(compressed_path)
    
    def _cleanup_old_checkpoints(self):
        """Limpia checkpoints antiguos."""
        try:
            # Obtener todos los checkpoints
            checkpoints = self.list_checkpoints(limit=1000)
            
            # Ordenar por timestamp
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Eliminar checkpoints excedentes
            if len(checkpoints) > self.config.max_checkpoints:
                for checkpoint in checkpoints[self.config.max_checkpoints:]:
                    self.delete_checkpoint(checkpoint.checkpoint_id)
            
            # Eliminar checkpoints antiguos
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            for checkpoint in checkpoints:
                if checkpoint.timestamp < cutoff_date:
                    self.delete_checkpoint(checkpoint.checkpoint_id)
                    
        except Exception as e:
            logger.error(f"Error limpiando checkpoints antiguos: {e}")
    
    def _update_stats(self, metadata: CheckpointMetadata):
        """Actualiza estadísticas."""
        self.checkpoint_stats['total_checkpoints_created'] += 1
        self.checkpoint_stats['total_storage_used_bytes'] += metadata.file_size_bytes
        
        # Calcular tamaño promedio
        total_checkpoints = self.checkpoint_stats['total_checkpoints_created']
        total_storage = self.checkpoint_stats['total_storage_used_bytes']
        self.checkpoint_stats['average_checkpoint_size_bytes'] = total_storage / total_checkpoints
    
    def _convert_to_huggingface_format(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte estado del modelo a formato HuggingFace."""
        # Implementación simplificada - en producción sería más compleja
        return {
            'state_dict': model_state,
            'model_config': self._extract_model_config(model_state)
        }
    
    def _extract_model_config(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae configuración del modelo."""
        # Implementación simplificada
        return {
            'model_type': 't5',
            'vocab_size': 32000,
            'd_model': 768,
            'd_ff': 3072,
            'num_layers': 12,
            'num_heads': 12
        }
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de checkpointing."""
        return self.checkpoint_stats.copy()
    
    def backup_checkpoints(self, backup_dir: Optional[str] = None) -> bool:
        """Hace backup de checkpoints."""
        try:
            if backup_dir is None:
                backup_dir = self.backup_dir
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"checkpoints_backup_{timestamp}")
            
            # Copiar directorio de checkpoints
            shutil.copytree(self.checkpoint_dir, backup_path)
            
            logger.info(f"Backup de checkpoints creado: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creando backup de checkpoints: {e}")
            return False


if __name__ == "__main__":
    # Test del AdvancedCheckpointManager
    logging.basicConfig(level=logging.INFO)
    
    manager = AdvancedCheckpointManager()
    
    # Crear checkpoint de prueba
    model_state = {
        'weights': np.random.randn(100, 100),
        'biases': np.random.randn(100),
        'config': {'hidden_size': 100, 'num_layers': 4}
    }
    
    optimizer_state = {
        'lr': 0.001,
        'momentum': 0.9,
        'step': 1000
    }
    
    training_state = {
        'epoch': 5,
        'best_loss': 0.5,
        'patience': 3
    }
    
    checkpoint_id = manager.save_checkpoint(
        model_state=model_state,
        optimizer_state=optimizer_state,
        training_state=training_state,
        step=1000,
        epoch=5,
        loss=0.5,
        learning_rate=0.001,
        model_name="test_model",
        tags=["test", "demo"]
    )
    
    print(f"Checkpoint guardado: {checkpoint_id}")
    
    # Listar checkpoints
    checkpoints = manager.list_checkpoints(limit=5)
    print(f"Checkpoints disponibles: {len(checkpoints)}")
    
    # Cargar checkpoint
    loaded_data = manager.load_checkpoint(checkpoint_id)
    if loaded_data:
        print(f"Checkpoint cargado exitosamente")
        print(f"Keys: {list(loaded_data.keys())}")
    
    # Mostrar estadísticas
    stats = manager.get_checkpoint_stats()
    print(f"Estadísticas: {stats}")
