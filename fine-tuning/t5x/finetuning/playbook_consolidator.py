#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playbook Consolidator - Consolidación de playbooks ACE (top 5K/7K) y filtrado de agentes graduados.
"""

import logging
import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlaybookEntry:
    """Entrada de playbook consolidado."""
    id: str
    agent_id: str
    domain: str
    pattern: str
    solution: str
    quality_score: float
    success_rate: float
    usage_count: int
    last_used: datetime
    created_at: datetime
    metadata: Dict[str, Any]
    tags: List[str]


@dataclass
class ConsolidatedPlaybook:
    """Playbook consolidado."""
    id: str
    domain: str
    total_entries: int
    top_entries: List[PlaybookEntry]
    quality_threshold: float
    success_threshold: float
    consolidation_date: datetime
    metadata: Dict[str, Any]


@dataclass
class AgentFilter:
    """Filtro para agentes graduados."""
    min_graduation_score: float
    min_interactions: int
    min_success_rate: float
    min_domain_expertise: float
    max_age_days: int
    required_domains: List[str]


class PlaybookConsolidator:
    """Consolidador de playbooks ACE."""
    
    def __init__(self, 
                 playbook_dir: str = "backend/data/playbooks",
                 output_dir: str = "backend/data/consolidated_playbooks",
                 top_k: int = 5000):
        self.playbook_dir = playbook_dir
        self.output_dir = output_dir
        self.top_k = top_k
        
        # Configuración de filtros
        self.quality_threshold = 0.8
        self.success_threshold = 0.85
        self.min_usage_count = 5
        
        # Estadísticas
        self.consolidation_stats = {
            'total_playbooks_processed': 0,
            'total_entries_processed': 0,
            'entries_filtered': 0,
            'entries_consolidated': 0,
            'domains_processed': 0,
            'consolidation_time_seconds': 0
        }
        
        # Asegurar directorios
        os.makedirs(self.playbook_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"PlaybookConsolidator inicializado: top_k={top_k}, output_dir={output_dir}")
    
    def consolidate_playbooks(self, 
                            agent_filter: Optional[AgentFilter] = None,
                            domains: Optional[List[str]] = None) -> Dict[str, ConsolidatedPlaybook]:
        """Consolida playbooks de agentes graduados."""
        start_time = datetime.now()
        logger.info("Iniciando consolidación de playbooks")
        
        try:
            # Obtener playbooks de agentes graduados
            graduated_playbooks = self._get_graduated_playbooks(agent_filter, domains)
            
            if not graduated_playbooks:
                logger.warning("No se encontraron playbooks de agentes graduados")
                return {}
            
            # Procesar por dominio
            consolidated_playbooks = {}
            
            for domain, playbooks in graduated_playbooks.items():
                logger.info(f"Consolidando playbooks para dominio {domain}: {len(playbooks)} playbooks")
                
                # Consolidar dominio
                consolidated = self._consolidate_domain_playbooks(domain, playbooks)
                
                if consolidated:
                    consolidated_playbooks[domain] = consolidated
                    self.consolidation_stats['domains_processed'] += 1
            
            # Guardar playbooks consolidados
            self._save_consolidated_playbooks(consolidated_playbooks)
            
            # Actualizar estadísticas
            self.consolidation_stats['consolidation_time_seconds'] = (
                datetime.now() - start_time
            ).total_seconds()
            
            logger.info(f"Consolidación completada: {len(consolidated_playbooks)} dominios procesados")
            return consolidated_playbooks
            
        except Exception as e:
            logger.error(f"Error en consolidación de playbooks: {e}")
            return {}
    
    def _get_graduated_playbooks(self, 
                               agent_filter: Optional[AgentFilter] = None,
                               domains: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Obtiene playbooks de agentes graduados."""
        try:
            # Filtro por defecto
            if not agent_filter:
                agent_filter = AgentFilter(
                    min_graduation_score=0.85,
                    min_interactions=100,
                    min_success_rate=0.85,
                    min_domain_expertise=0.8,
                    max_age_days=30,
                    required_domains=[]
                )
            
            # Buscar archivos de playbooks
            playbook_files = []
            for filename in os.listdir(self.playbook_dir):
                if filename.startswith("agent_") and filename.endswith("_playbook.json"):
                    playbook_files.append(os.path.join(self.playbook_dir, filename))
            
            logger.info(f"Encontrados {len(playbook_files)} archivos de playbooks")
            
            # Procesar playbooks
            domain_playbooks = defaultdict(list)
            
            for playbook_file in playbook_files:
                try:
                    with open(playbook_file, 'r', encoding='utf-8') as f:
                        playbook_data = json.load(f)
                    
                    # Verificar si el agente cumple los criterios
                    if self._agent_meets_criteria(playbook_data, agent_filter):
                        domain = playbook_data.get('domain', 'general')
                        
                        # Filtrar por dominios si se especifica
                        if domains and domain not in domains:
                            continue
                        
                        domain_playbooks[domain].append(playbook_data)
                        self.consolidation_stats['total_playbooks_processed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error procesando playbook {playbook_file}: {e}")
                    continue
            
            return dict(domain_playbooks)
            
        except Exception as e:
            logger.error(f"Error obteniendo playbooks graduados: {e}")
            return {}
    
    def _agent_meets_criteria(self, playbook_data: Dict[str, Any], agent_filter: AgentFilter) -> bool:
        """Verifica si un agente cumple los criterios de filtrado."""
        try:
            # Verificar score de graduación
            graduation_score = playbook_data.get('graduation_score', 0.0)
            if graduation_score < agent_filter.min_graduation_score:
                return False
            
            # Verificar interacciones
            total_interactions = playbook_data.get('total_interactions', 0)
            if total_interactions < agent_filter.min_interactions:
                return False
            
            # Verificar tasa de éxito
            success_rate = playbook_data.get('success_rate', 0.0)
            if success_rate < agent_filter.min_success_rate:
                return False
            
            # Verificar expertise en dominio
            domain_expertise = playbook_data.get('domain_expertise', 0.0)
            if domain_expertise < agent_filter.min_domain_expertise:
                return False
            
            # Verificar edad del playbook
            created_at_str = playbook_data.get('created_at', '')
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    age_days = (datetime.now() - created_at).days
                    if age_days > agent_filter.max_age_days:
                        return False
                except:
                    pass
            
            # Verificar dominios requeridos
            domain = playbook_data.get('domain', '')
            if agent_filter.required_domains and domain not in agent_filter.required_domains:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando criterios del agente: {e}")
            return False
    
    def _consolidate_domain_playbooks(self, 
                                    domain: str, 
                                    playbooks: List[Dict[str, Any]]) -> Optional[ConsolidatedPlaybook]:
        """Consolida playbooks de un dominio específico."""
        try:
            # Extraer todas las entradas de patrones
            all_entries = []
            
            for playbook in playbooks:
                patterns = playbook.get('patterns', [])
                agent_id = playbook.get('agent_id', 'unknown')
                graduation_score = playbook.get('graduation_score', 0.0)
                success_rate = playbook.get('success_rate', 0.0)
                
                for pattern in patterns:
                    entry = PlaybookEntry(
                        id=self._generate_entry_id(pattern, agent_id),
                        agent_id=agent_id,
                        domain=domain,
                        pattern=pattern.get('query_pattern', ''),
                        solution=pattern.get('response_template', ''),
                        quality_score=pattern.get('quality_score', 0.0),
                        success_rate=success_rate,
                        usage_count=1,  # Simulado
                        last_used=datetime.now(),
                        created_at=datetime.now(),
                        metadata={
                            'graduation_score': graduation_score,
                            'execution_time': pattern.get('execution_time', 0),
                            'corrections_applied': pattern.get('corrections_applied', 0)
                        },
                        tags=self._extract_tags(pattern)
                    )
                    
                    all_entries.append(entry)
                    self.consolidation_stats['total_entries_processed'] += 1
            
            if not all_entries:
                logger.warning(f"No se encontraron entradas para el dominio {domain}")
                return None
            
            # Filtrar entradas por calidad
            filtered_entries = [
                entry for entry in all_entries
                if (entry.quality_score >= self.quality_threshold and
                    entry.success_rate >= self.success_threshold and
                    entry.usage_count >= self.min_usage_count)
            ]
            
            self.consolidation_stats['entries_filtered'] += len(filtered_entries)
            
            # Ordenar por calidad y éxito
            filtered_entries.sort(
                key=lambda x: (x.quality_score * 0.6 + x.success_rate * 0.4),
                reverse=True
            )
            
            # Tomar top K entradas
            top_entries = filtered_entries[:self.top_k]
            self.consolidation_stats['entries_consolidated'] += len(top_entries)
            
            # Crear playbook consolidado
            consolidated = ConsolidatedPlaybook(
                id=f"consolidated_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                domain=domain,
                total_entries=len(all_entries),
                top_entries=top_entries,
                quality_threshold=self.quality_threshold,
                success_threshold=self.success_threshold,
                consolidation_date=datetime.now(),
                metadata={
                    'source_playbooks': len(playbooks),
                    'filtered_entries': len(filtered_entries),
                    'consolidated_entries': len(top_entries),
                    'avg_quality_score': np.mean([e.quality_score for e in top_entries]) if top_entries else 0.0,
                    'avg_success_rate': np.mean([e.success_rate for e in top_entries]) if top_entries else 0.0
                }
            )
            
            logger.info(f"Dominio {domain} consolidado: {len(top_entries)} entradas de {len(all_entries)} totales")
            return consolidated
            
        except Exception as e:
            logger.error(f"Error consolidando playbooks del dominio {domain}: {e}")
            return None
    
    def _generate_entry_id(self, pattern: Dict[str, Any], agent_id: str) -> str:
        """Genera ID único para entrada de playbook."""
        content = f"{pattern.get('query_pattern', '')}_{pattern.get('response_template', '')}_{agent_id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_tags(self, pattern: Dict[str, Any]) -> List[str]:
        """Extrae tags de un patrón."""
        tags = []
        
        # Tags basados en calidad
        quality_score = pattern.get('quality_score', 0.0)
        if quality_score >= 0.9:
            tags.append("high_quality")
        elif quality_score >= 0.8:
            tags.append("good_quality")
        
        # Tags basados en tiempo de ejecución
        execution_time = pattern.get('execution_time', 0)
        if execution_time < 1000:
            tags.append("fast_execution")
        elif execution_time > 5000:
            tags.append("slow_execution")
        
        # Tags basados en correcciones
        corrections = pattern.get('corrections_applied', 0)
        if corrections == 0:
            tags.append("no_corrections")
        elif corrections > 2:
            tags.append("multiple_corrections")
        
        return tags
    
    def _save_consolidated_playbooks(self, consolidated_playbooks: Dict[str, ConsolidatedPlaybook]):
        """Guarda playbooks consolidados."""
        try:
            for domain, consolidated in consolidated_playbooks.items():
                # Convertir a diccionario para JSON
                consolidated_dict = {
                    'id': consolidated.id,
                    'domain': consolidated.domain,
                    'total_entries': consolidated.total_entries,
                    'top_entries': [asdict(entry) for entry in consolidated.top_entries],
                    'quality_threshold': consolidated.quality_threshold,
                    'success_threshold': consolidated.success_threshold,
                    'consolidation_date': consolidated.consolidation_date.isoformat(),
                    'metadata': consolidated.metadata
                }
                
                # Guardar archivo
                filename = f"consolidated_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(consolidated_dict, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Playbook consolidado guardado: {filepath}")
                
        except Exception as e:
            logger.error(f"Error guardando playbooks consolidados: {e}")
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de consolidación."""
        return self.consolidation_stats.copy()
    
    def analyze_playbook_quality(self, domain: str) -> Dict[str, Any]:
        """Analiza la calidad de playbooks de un dominio."""
        try:
            # Buscar playbooks del dominio
            domain_playbooks = []
            for filename in os.listdir(self.playbook_dir):
                if filename.startswith("agent_") and filename.endswith("_playbook.json"):
                    filepath = os.path.join(self.playbook_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            playbook_data = json.load(f)
                        
                        if playbook_data.get('domain') == domain:
                            domain_playbooks.append(playbook_data)
                    except:
                        continue
            
            if not domain_playbooks:
                return {'error': f'No se encontraron playbooks para el dominio {domain}'}
            
            # Análisis de calidad
            quality_scores = []
            success_rates = []
            graduation_scores = []
            interaction_counts = []
            
            for playbook in domain_playbooks:
                patterns = playbook.get('patterns', [])
                for pattern in patterns:
                    quality_scores.append(pattern.get('quality_score', 0.0))
                
                success_rates.append(playbook.get('success_rate', 0.0))
                graduation_scores.append(playbook.get('graduation_score', 0.0))
                interaction_counts.append(playbook.get('total_interactions', 0))
            
            analysis = {
                'domain': domain,
                'total_playbooks': len(domain_playbooks),
                'total_patterns': len(quality_scores),
                'quality_analysis': {
                    'avg_quality_score': np.mean(quality_scores) if quality_scores else 0.0,
                    'min_quality_score': np.min(quality_scores) if quality_scores else 0.0,
                    'max_quality_score': np.max(quality_scores) if quality_scores else 0.0,
                    'std_quality_score': np.std(quality_scores) if quality_scores else 0.0
                },
                'success_analysis': {
                    'avg_success_rate': np.mean(success_rates) if success_rates else 0.0,
                    'min_success_rate': np.min(success_rates) if success_rates else 0.0,
                    'max_success_rate': np.max(success_rates) if success_rates else 0.0
                },
                'graduation_analysis': {
                    'avg_graduation_score': np.mean(graduation_scores) if graduation_scores else 0.0,
                    'min_graduation_score': np.min(graduation_scores) if graduation_scores else 0.0,
                    'max_graduation_score': np.max(graduation_scores) if graduation_scores else 0.0
                },
                'interaction_analysis': {
                    'avg_interactions': np.mean(interaction_counts) if interaction_counts else 0.0,
                    'total_interactions': np.sum(interaction_counts) if interaction_counts else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando calidad de playbooks para dominio {domain}: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Test del PlaybookConsolidator
    logging.basicConfig(level=logging.INFO)
    
    consolidator = PlaybookConsolidator(top_k=1000)
    
    # Crear playbooks de prueba
    test_playbooks = [
        {
            'agent_id': 'test_agent_001',
            'domain': 'python',
            'graduation_score': 0.9,
            'success_rate': 0.88,
            'total_interactions': 150,
            'domain_expertise': 0.85,
            'created_at': datetime.now().isoformat(),
            'patterns': [
                {
                    'query_pattern': 'How to create a Python function?',
                    'response_template': 'Use def keyword: def function_name():',
                    'quality_score': 0.9,
                    'execution_time': 500,
                    'corrections_applied': 0
                },
                {
                    'query_pattern': 'Python list comprehension',
                    'response_template': '[expression for item in iterable if condition]',
                    'quality_score': 0.85,
                    'execution_time': 300,
                    'corrections_applied': 1
                }
            ]
        },
        {
            'agent_id': 'test_agent_002',
            'domain': 'python',
            'graduation_score': 0.87,
            'success_rate': 0.86,
            'total_interactions': 120,
            'domain_expertise': 0.82,
            'created_at': datetime.now().isoformat(),
            'patterns': [
                {
                    'query_pattern': 'Python decorators',
                    'response_template': '@decorator_name above function definition',
                    'quality_score': 0.88,
                    'execution_time': 800,
                    'corrections_applied': 0
                }
            ]
        }
    ]
    
    # Guardar playbooks de prueba
    for i, playbook in enumerate(test_playbooks):
        filename = f"agent_{playbook['agent_id']}_playbook.json"
        filepath = os.path.join(consolidator.playbook_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(playbook, f, indent=2, ensure_ascii=False)
    
    # Consolidar playbooks
    consolidated = consolidator.consolidate_playbooks()
    print(f"Playbooks consolidados: {len(consolidated)}")
    
    # Mostrar estadísticas
    stats = consolidator.get_consolidation_stats()
    print(f"Estadísticas de consolidación: {stats}")
    
    # Analizar calidad
    analysis = consolidator.analyze_playbook_quality('python')
    print(f"Análisis de calidad: {analysis}")
