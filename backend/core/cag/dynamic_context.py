#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynamicContext - Contexto dinámico que se llenará con ACE.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import threading
from collections import deque

logger = logging.getLogger(__name__)


class DynamicContext:
    """
    Contexto dinámico que se llenará con ACE.
    Gestiona contexto evolutivo basado en conversaciones y patrones.
    """
    
    def __init__(self, max_context_size: int = 1000, 
                 context_ttl_hours: int = 24):
        """
        Inicializa el DynamicContext.
        
        Args:
            max_context_size: Tamaño máximo del contexto
            context_ttl_hours: TTL del contexto en horas
        """
        self.max_context_size = max_context_size
        self.context_ttl = timedelta(hours=context_ttl_hours)
        
        # Almacenamiento de contexto
        self.context_entries = deque(maxlen=max_context_size)
        self.context_providers = {}
        self.context_filters = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Métricas
        self.stats = {
            'total_entries': 0,
            'active_entries': 0,
            'expired_entries': 0,
            'provider_calls': 0
        }
        
        logger.info("DynamicContext inicializado")
    
    def register_provider(self, name: str, provider_func: Callable, 
                         priority: int = 0, enabled: bool = True):
        """
        Registra un proveedor de contexto.
        
        Args:
            name: Nombre del proveedor
            provider_func: Función que genera contexto
            priority: Prioridad (mayor = más importante)
            enabled: Si está habilitado
        """
        try:
            with self._lock:
                self.context_providers[name] = {
                    'function': provider_func,
                    'priority': priority,
                    'enabled': enabled,
                    'last_called': None,
                    'call_count': 0
                }
                
                logger.info(f"Proveedor de contexto '{name}' registrado con prioridad {priority}")
                
        except Exception as e:
            logger.error(f"Error registrando proveedor '{name}': {e}")
    
    def unregister_provider(self, name: str):
        """Desregistra un proveedor de contexto."""
        try:
            with self._lock:
                if name in self.context_providers:
                    del self.context_providers[name]
                    logger.info(f"Proveedor '{name}' desregistrado")
                else:
                    logger.warning(f"Proveedor '{name}' no encontrado")
                    
        except Exception as e:
            logger.error(f"Error desregistrando proveedor '{name}': {e}")
    
    def register_filter(self, name: str, filter_func: Callable):
        """
        Registra un filtro de contexto.
        
        Args:
            name: Nombre del filtro
            filter_func: Función que filtra contexto
        """
        try:
            with self._lock:
                self.context_filters[name] = filter_func
                logger.info(f"Filtro de contexto '{name}' registrado")
                
        except Exception as e:
            logger.error(f"Error registrando filtro '{name}': {e}")
    
    def add_context(self, content: str, source: str = "manual", 
                   metadata: Dict[str, Any] = None, ttl_hours: int = None):
        """
        Agrega contexto manual.
        
        Args:
            content: Contenido del contexto
            source: Fuente del contexto
            metadata: Metadata adicional
            ttl_hours: TTL personalizado en horas
        """
        try:
            with self._lock:
                if metadata is None:
                    metadata = {}
                
                ttl = timedelta(hours=ttl_hours) if ttl_hours else self.context_ttl
                expires_at = datetime.now() + ttl
                
                entry = {
                    'id': self._generate_entry_id(),
                    'content': content,
                    'source': source,
                    'metadata': metadata,
                    'created_at': datetime.now(),
                    'expires_at': expires_at,
                    'access_count': 0,
                    'last_accessed': None
                }
                
                self.context_entries.append(entry)
                self.stats['total_entries'] += 1
                self.stats['active_entries'] += 1
                
                logger.debug(f"Contexto agregado: {source} - {len(content)} chars")
                
        except Exception as e:
            logger.error(f"Error agregando contexto: {e}")
    
    def get_context(self, query: str, max_tokens: int = 2000, 
                   include_providers: bool = True) -> str:
        """
        Obtiene contexto relevante para una query.
        
        Args:
            query: Query del usuario
            max_tokens: Máximo número de tokens
            include_providers: Si incluir proveedores dinámicos
            
        Returns:
            Contexto relevante
        """
        try:
            with self._lock:
                # Limpiar entradas expiradas
                self._cleanup_expired()
                
                # Obtener contexto de proveedores
                provider_context = ""
                if include_providers:
                    provider_context = self._get_provider_context(query, max_tokens // 2)
                
                # Obtener contexto almacenado
                stored_context = self._get_stored_context(query, max_tokens - len(provider_context.split()))
                
                # Combinar contextos
                context_parts = []
                if provider_context:
                    context_parts.append(f"**Contexto Dinámico:**\n{provider_context}")
                if stored_context:
                    context_parts.append(f"**Contexto Histórico:**\n{stored_context}")
                
                final_context = "\n\n".join(context_parts)
                
                logger.debug(f"Contexto generado: {len(final_context.split())} tokens")
                return final_context
                
        except Exception as e:
            logger.error(f"Error obteniendo contexto: {e}")
            return ""
    
    def _get_provider_context(self, query: str, max_tokens: int) -> str:
        """Obtiene contexto de proveedores registrados."""
        try:
            # Ordenar proveedores por prioridad
            sorted_providers = sorted(
                self.context_providers.items(),
                key=lambda x: x[1]['priority'],
                reverse=True
            )
            
            context_parts = []
            current_tokens = 0
            
            for name, provider_info in sorted_providers:
                if not provider_info['enabled']:
                    continue
                
                if current_tokens >= max_tokens:
                    break
                
                try:
                    # Llamar proveedor
                    provider_context = provider_info['function'](query)
                    
                    if provider_context:
                        provider_tokens = len(provider_context.split())
                        
                        if current_tokens + provider_tokens <= max_tokens:
                            context_parts.append(f"[{name}] {provider_context}")
                            current_tokens += provider_tokens
                            
                            # Actualizar estadísticas
                            provider_info['last_called'] = datetime.now()
                            provider_info['call_count'] += 1
                            self.stats['provider_calls'] += 1
                
                except Exception as e:
                    logger.error(f"Error en proveedor '{name}': {e}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error obteniendo contexto de proveedores: {e}")
            return ""
    
    def _get_stored_context(self, query: str, max_tokens: int) -> str:
        """Obtiene contexto relevante del almacenamiento."""
        try:
            # Filtrar entradas activas
            active_entries = [
                entry for entry in self.context_entries
                if entry['expires_at'] > datetime.now()
            ]
            
            # Aplicar filtros
            filtered_entries = self._apply_filters(active_entries, query)
            
            # Ordenar por relevancia (acceso reciente, frecuencia)
            filtered_entries.sort(key=lambda x: (
                x['last_accessed'] or datetime.min,
                x['access_count']
            ), reverse=True)
            
            # Construir contexto
            context_parts = []
            current_tokens = 0
            
            for entry in filtered_entries:
                if current_tokens >= max_tokens:
                    break
                
                entry_tokens = len(entry['content'].split())
                
                if current_tokens + entry_tokens <= max_tokens:
                    context_parts.append(entry['content'])
                    current_tokens += entry_tokens
                    
                    # Actualizar estadísticas de acceso
                    entry['access_count'] += 1
                    entry['last_accessed'] = datetime.now()
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error obteniendo contexto almacenado: {e}")
            return ""
    
    def _apply_filters(self, entries: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Aplica filtros registrados a las entradas."""
        try:
            filtered_entries = entries
            
            for filter_name, filter_func in self.context_filters.items():
                try:
                    filtered_entries = filter_func(filtered_entries, query)
                except Exception as e:
                    logger.error(f"Error en filtro '{filter_name}': {e}")
            
            return filtered_entries
            
        except Exception as e:
            logger.error(f"Error aplicando filtros: {e}")
            return entries
    
    def _cleanup_expired(self):
        """Limpia entradas expiradas."""
        try:
            now = datetime.now()
            initial_count = len(self.context_entries)
            
            # Filtrar entradas no expiradas
            self.context_entries = deque([
                entry for entry in self.context_entries
                if entry['expires_at'] > now
            ], maxlen=self.max_context_size)
            
            expired_count = initial_count - len(self.context_entries)
            if expired_count > 0:
                self.stats['expired_entries'] += expired_count
                self.stats['active_entries'] = len(self.context_entries)
                logger.debug(f"Limpiadas {expired_count} entradas expiradas")
                
        except Exception as e:
            logger.error(f"Error limpiando entradas expiradas: {e}")
    
    def _generate_entry_id(self) -> str:
        """Genera ID único para entrada."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def clear_context(self, source: str = None):
        """
        Limpia contexto.
        
        Args:
            source: Fuente específica a limpiar (opcional)
        """
        try:
            with self._lock:
                if source:
                    # Limpiar solo de una fuente específica
                    self.context_entries = deque([
                        entry for entry in self.context_entries
                        if entry['source'] != source
                    ], maxlen=self.max_context_size)
                else:
                    # Limpiar todo
                    self.context_entries.clear()
                
                self.stats['active_entries'] = len(self.context_entries)
                logger.info(f"Contexto limpiado (fuente: {source or 'todas'})")
                
        except Exception as e:
            logger.error(f"Error limpiando contexto: {e}")
    
    def get_context_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retorna historial de contexto."""
        try:
            with self._lock:
                return list(self.context_entries)[-limit:]
        except Exception as e:
            logger.error(f"Error obteniendo historial: {e}")
            return []
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de proveedores."""
        try:
            with self._lock:
                stats = {}
                for name, info in self.context_providers.items():
                    stats[name] = {
                        'enabled': info['enabled'],
                        'priority': info['priority'],
                        'call_count': info['call_count'],
                        'last_called': info['last_called'].isoformat() if info['last_called'] else None
                    }
                return stats
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de proveedores: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas generales."""
        try:
            with self._lock:
                return {
                    'total_entries': self.stats['total_entries'],
                    'active_entries': len(self.context_entries),
                    'expired_entries': self.stats['expired_entries'],
                    'provider_calls': self.stats['provider_calls'],
                    'registered_providers': len(self.context_providers),
                    'registered_filters': len(self.context_filters),
                    'max_context_size': self.max_context_size
                }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def export_context(self, file_path: str):
        """Exporta contexto a archivo JSON."""
        try:
            with self._lock:
                export_data = {
                    'exported_at': datetime.now().isoformat(),
                    'stats': self.get_stats(),
                    'providers': self.get_provider_stats(),
                    'context_entries': list(self.context_entries)
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"Contexto exportado a {file_path}")
                
        except Exception as e:
            logger.error(f"Error exportando contexto: {e}")
    
    def import_context(self, file_path: str):
        """Importa contexto desde archivo JSON."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with self._lock:
                # Importar entradas de contexto
                if 'context_entries' in import_data:
                    for entry_data in import_data['context_entries']:
                        # Convertir strings de fecha a datetime
                        entry_data['created_at'] = datetime.fromisoformat(entry_data['created_at'])
                        entry_data['expires_at'] = datetime.fromisoformat(entry_data['expires_at'])
                        if entry_data.get('last_accessed'):
                            entry_data['last_accessed'] = datetime.fromisoformat(entry_data['last_accessed'])
                        
                        self.context_entries.append(entry_data)
                
                logger.info(f"Contexto importado desde {file_path}")
                
        except Exception as e:
            logger.error(f"Error importando contexto: {e}")


# Funciones de conveniencia
def create_dynamic_context(max_context_size: int = 1000) -> DynamicContext:
    """Crea una instancia de DynamicContext."""
    return DynamicContext(max_context_size)


# Filtros predefinidos
def relevance_filter(entries: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Filtro de relevancia basado en palabras clave."""
    try:
        query_words = set(query.lower().split())
        relevant_entries = []
        
        for entry in entries:
            content_words = set(entry['content'].lower().split())
            metadata_words = set()
            
            # Incluir palabras de metadata
            for key, value in entry.get('metadata', {}).items():
                if isinstance(value, str):
                    metadata_words.update(value.lower().split())
            
            # Calcular similitud
            all_entry_words = content_words.union(metadata_words)
            similarity = len(query_words.intersection(all_entry_words)) / len(query_words)
            
            if similarity > 0.1:  # Al menos 10% de similitud
                relevant_entries.append(entry)
        
        return relevant_entries
        
    except Exception as e:
        logger.error(f"Error en filtro de relevancia: {e}")
        return entries


def recency_filter(entries: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Filtro de recencia (últimas 24 horas)."""
    try:
        cutoff_time = datetime.now() - timedelta(hours=24)
        return [
            entry for entry in entries
            if entry['created_at'] > cutoff_time
        ]
    except Exception as e:
        logger.error(f"Error en filtro de recencia: {e}")
        return entries


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear DynamicContext
    context = create_dynamic_context()
    
    # Registrar filtros
    context.register_filter('relevance', relevance_filter)
    context.register_filter('recency', recency_filter)
    
    # Agregar contexto de prueba
    context.add_context(
        "Python es un lenguaje de programación interpretado",
        source="knowledge_base",
        metadata={"category": "programming", "language": "python"}
    )
    
    # Test obtención de contexto
    result = context.get_context("python programming", max_tokens=100)
    print("Contexto obtenido:")
    print(result)
    
    # Test estadísticas
    stats = context.get_stats()
    print(f"\nEstadísticas: {stats}")
