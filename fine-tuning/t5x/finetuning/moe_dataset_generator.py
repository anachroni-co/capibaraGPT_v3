#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoE Dataset Generator - Generación de datasets para Mixture of Experts.
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

from .playbook_consolidator import ConsolidatedPlaybook, PlaybookEntry
from .e2b_log_processor import TrainingExample, CodePattern

logger = logging.getLogger(__name__)


@dataclass
class ExpertDomain:
    """Dominio de experto."""
    domain_id: str
    domain_name: str
    description: str
    keywords: List[str]
    complexity_levels: List[str]  # beginner, intermediate, advanced
    specializations: List[str]


@dataclass
class MoEDataset:
    """Dataset para Mixture of Experts."""
    dataset_id: str
    expert_domain: str
    total_examples: int
    training_examples: List[Dict[str, Any]]
    validation_examples: List[Dict[str, Any]]
    test_examples: List[Dict[str, Any]]
    domain_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]
    generation_date: datetime
    metadata: Dict[str, Any]


@dataclass
class ExpertRoutingExample:
    """Ejemplo de routing para expertos."""
    example_id: str
    input_text: str
    expected_expert: str
    confidence_score: float
    complexity_score: float
    domain_features: List[str]
    routing_features: Dict[str, Any]


class MoEDatasetGenerator:
    """Generador de datasets para Mixture of Experts."""
    
    def __init__(self, 
                 consolidated_playbooks_dir: str = "backend/data/consolidated_playbooks",
                 training_datasets_dir: str = "backend/data/training_datasets",
                 output_dir: str = "backend/data/moe_datasets"):
        self.consolidated_playbooks_dir = consolidated_playbooks_dir
        self.training_datasets_dir = training_datasets_dir
        self.output_dir = output_dir
        
        # Configuración de expertos
        self.expert_domains = self._initialize_expert_domains()
        
        # Configuración de datasets
        self.train_ratio = 0.7
        self.validation_ratio = 0.15
        self.test_ratio = 0.15
        self.min_examples_per_domain = 100
        self.max_examples_per_domain = 10000
        
        # Estadísticas
        self.generation_stats = {
            'total_datasets_generated': 0,
            'total_examples_processed': 0,
            'domains_processed': 0,
            'routing_examples_generated': 0,
            'generation_time_seconds': 0
        }
        
        # Asegurar directorios
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"MoEDatasetGenerator inicializado: output_dir={output_dir}")
    
    def _initialize_expert_domains(self) -> Dict[str, ExpertDomain]:
        """Inicializa dominios de expertos."""
        return {
            'python': ExpertDomain(
                domain_id='python',
                domain_name='Python Programming',
                description='Python programming, data science, web development, automation',
                keywords=['python', 'data science', 'web', 'automation', 'scripting', 'pandas', 'numpy', 'django', 'flask'],
                complexity_levels=['beginner', 'intermediate', 'advanced'],
                specializations=['data_analysis', 'web_development', 'automation', 'machine_learning', 'scripting']
            ),
            'sql': ExpertDomain(
                domain_id='sql',
                domain_name='SQL Database',
                description='SQL queries, database design, optimization, analytics',
                keywords=['sql', 'database', 'query', 'optimization', 'analytics', 'postgresql', 'mysql', 'sqlite'],
                complexity_levels=['beginner', 'intermediate', 'advanced'],
                specializations=['query_optimization', 'database_design', 'analytics', 'performance_tuning']
            ),
            'javascript': ExpertDomain(
                domain_id='javascript',
                domain_name='JavaScript Development',
                description='JavaScript, Node.js, frontend frameworks, web development',
                keywords=['javascript', 'nodejs', 'react', 'vue', 'angular', 'frontend', 'backend', 'web'],
                complexity_levels=['beginner', 'intermediate', 'advanced'],
                specializations=['frontend', 'backend', 'fullstack', 'frameworks', 'async_programming']
            ),
            'debug': ExpertDomain(
                domain_id='debug',
                domain_name='Debugging & Troubleshooting',
                description='Error analysis, debugging techniques, performance optimization',
                keywords=['debug', 'error', 'troubleshoot', 'optimization', 'performance', 'fix', 'bug'],
                complexity_levels=['beginner', 'intermediate', 'advanced'],
                specializations=['error_analysis', 'performance_debugging', 'system_troubleshooting', 'code_review']
            ),
            'ml': ExpertDomain(
                domain_id='ml',
                domain_name='Machine Learning',
                description='Machine learning, AI, data modeling, algorithms',
                keywords=['machine learning', 'ai', 'model', 'algorithm', 'tensorflow', 'pytorch', 'scikit-learn'],
                complexity_levels=['intermediate', 'advanced'],
                specializations=['deep_learning', 'nlp', 'computer_vision', 'model_optimization', 'data_preprocessing']
            ),
            'api': ExpertDomain(
                domain_id='api',
                domain_name='API Development',
                description='REST APIs, microservices, integration, documentation',
                keywords=['api', 'rest', 'microservices', 'integration', 'documentation', 'endpoint', 'http'],
                complexity_levels=['beginner', 'intermediate', 'advanced'],
                specializations=['rest_api', 'graphql', 'microservices', 'api_design', 'integration']
            )
        }
    
    def generate_moe_datasets(self, 
                            domains: Optional[List[str]] = None,
                            include_routing: bool = True) -> Dict[str, MoEDataset]:
        """Genera datasets para Mixture of Experts."""
        start_time = datetime.now()
        logger.info("Iniciando generación de datasets MoE")
        
        try:
            # Cargar datos consolidados
            consolidated_data = self._load_consolidated_data(domains)
            
            if not consolidated_data:
                logger.warning("No se encontraron datos consolidados para generar datasets MoE")
                return {}
            
            # Generar datasets por dominio
            moe_datasets = {}
            
            for domain, data in consolidated_data.items():
                if domain not in self.expert_domains:
                    logger.warning(f"Dominio {domain} no está configurado como experto")
                    continue
                
                logger.info(f"Generando dataset MoE para dominio {domain}")
                
                # Generar dataset del dominio
                moe_dataset = self._generate_domain_dataset(domain, data)
                
                if moe_dataset:
                    moe_datasets[domain] = moe_dataset
                    self.generation_stats['domains_processed'] += 1
            
            # Generar dataset de routing si se solicita
            if include_routing and moe_datasets:
                routing_dataset = self._generate_routing_dataset(moe_datasets)
                if routing_dataset:
                    moe_datasets['routing'] = routing_dataset
            
            # Guardar datasets
            self._save_moe_datasets(moe_datasets)
            
            # Actualizar estadísticas
            self.generation_stats['generation_time_seconds'] = (
                datetime.now() - start_time
            ).total_seconds()
            
            logger.info(f"Generación completada: {len(moe_datasets)} datasets generados")
            return moe_datasets
            
        except Exception as e:
            logger.error(f"Error generando datasets MoE: {e}")
            return {}
    
    def _load_consolidated_data(self, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """Carga datos consolidados de playbooks y training datasets."""
        try:
            consolidated_data = {}
            
            # Cargar playbooks consolidados
            if os.path.exists(self.consolidated_playbooks_dir):
                for filename in os.listdir(self.consolidated_playbooks_dir):
                    if filename.startswith("consolidated_") and filename.endswith(".json"):
                        filepath = os.path.join(self.consolidated_playbooks_dir, filename)
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                playbook_data = json.load(f)
                            
                            domain = playbook_data.get('domain', 'unknown')
                            
                            # Filtrar por dominios si se especifica
                            if domains and domain not in domains:
                                continue
                            
                            if domain not in consolidated_data:
                                consolidated_data[domain] = {
                                    'playbooks': [],
                                    'training_examples': []
                                }
                            
                            consolidated_data[domain]['playbooks'].append(playbook_data)
                            
                        except Exception as e:
                            logger.error(f"Error cargando playbook {filename}: {e}")
                            continue
            
            # Cargar training datasets
            if os.path.exists(self.training_datasets_dir):
                for filename in os.listdir(self.training_datasets_dir):
                    if filename.startswith("training_dataset_") and filename.endswith(".json"):
                        filepath = os.path.join(self.training_datasets_dir, filename)
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                training_data = json.load(f)
                            
                            language = training_data.get('language', 'unknown')
                            
                            # Mapear lenguaje a dominio
                            domain = self._map_language_to_domain(language)
                            
                            # Filtrar por dominios si se especifica
                            if domains and domain not in domains:
                                continue
                            
                            if domain not in consolidated_data:
                                consolidated_data[domain] = {
                                    'playbooks': [],
                                    'training_examples': []
                                }
                            
                            # Agregar ejemplos de entrenamiento
                            training_examples = training_data.get('training_examples', [])
                            consolidated_data[domain]['training_examples'].extend(training_examples)
                            
                        except Exception as e:
                            logger.error(f"Error cargando training dataset {filename}: {e}")
                            continue
            
            logger.info(f"Datos consolidados cargados: {len(consolidated_data)} dominios")
            return consolidated_data
            
        except Exception as e:
            logger.error(f"Error cargando datos consolidados: {e}")
            return {}
    
    def _map_language_to_domain(self, language: str) -> str:
        """Mapea lenguaje a dominio de experto."""
        language_mapping = {
            'python': 'python',
            'sql': 'sql',
            'javascript': 'javascript',
            'js': 'javascript',
            'debug': 'debug',
            'ml': 'ml',
            'api': 'api'
        }
        
        return language_mapping.get(language.lower(), 'python')  # Default a python
    
    def _generate_domain_dataset(self, domain: str, data: Dict[str, Any]) -> Optional[MoEDataset]:
        """Genera dataset para un dominio específico."""
        try:
            # Combinar ejemplos de playbooks y training datasets
            all_examples = []
            
            # Agregar ejemplos de playbooks
            for playbook in data.get('playbooks', []):
                playbook_examples = self._convert_playbook_to_examples(playbook)
                all_examples.extend(playbook_examples)
            
            # Agregar ejemplos de training datasets
            training_examples = data.get('training_examples', [])
            all_examples.extend(training_examples)
            
            if not all_examples:
                logger.warning(f"No se encontraron ejemplos para el dominio {domain}")
                return None
            
            # Filtrar y limpiar ejemplos
            filtered_examples = self._filter_examples(all_examples, domain)
            
            if len(filtered_examples) < self.min_examples_per_domain:
                logger.warning(f"Dominio {domain} tiene muy pocos ejemplos: {len(filtered_examples)}")
                return None
            
            # Limitar número de ejemplos
            if len(filtered_examples) > self.max_examples_per_domain:
                filtered_examples = filtered_examples[:self.max_examples_per_domain]
            
            # Dividir en train/validation/test
            train_examples, val_examples, test_examples = self._split_examples(filtered_examples)
            
            # Calcular métricas de calidad
            quality_metrics = self._calculate_quality_metrics(filtered_examples)
            
            # Calcular distribuciones
            domain_distribution = self._calculate_domain_distribution(filtered_examples)
            difficulty_distribution = self._calculate_difficulty_distribution(filtered_examples)
            
            # Crear dataset MoE
            moe_dataset = MoEDataset(
                dataset_id=f"moe_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                expert_domain=domain,
                total_examples=len(filtered_examples),
                training_examples=train_examples,
                validation_examples=val_examples,
                test_examples=test_examples,
                domain_distribution=domain_distribution,
                difficulty_distribution=difficulty_distribution,
                quality_metrics=quality_metrics,
                generation_date=datetime.now(),
                metadata={
                    'expert_domain_info': asdict(self.expert_domains[domain]),
                    'source_playbooks': len(data.get('playbooks', [])),
                    'source_training_examples': len(data.get('training_examples', [])),
                    'filtered_examples': len(filtered_examples)
                }
            )
            
            self.generation_stats['total_examples_processed'] += len(filtered_examples)
            
            logger.info(f"Dataset MoE generado para {domain}: {len(filtered_examples)} ejemplos")
            return moe_dataset
            
        except Exception as e:
            logger.error(f"Error generando dataset para dominio {domain}: {e}")
            return None
    
    def _convert_playbook_to_examples(self, playbook: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convierte playbook a ejemplos de entrenamiento."""
        examples = []
        
        top_entries = playbook.get('top_entries', [])
        
        for entry in top_entries:
            example = {
                'example_id': f"playbook_{entry.get('id', 'unknown')}",
                'language': playbook.get('domain', 'unknown'),
                'input_text': f"Context: {entry.get('pattern', '')}\nWrite code to solve this problem.",
                'output_code': entry.get('solution', ''),
                'explanation': f"This code demonstrates {playbook.get('domain', 'programming')} best practices.",
                'difficulty_level': self._determine_difficulty_from_quality(entry.get('quality_score', 0.5)),
                'domain': playbook.get('domain', 'unknown'),
                'success_rate': entry.get('success_rate', 0.8),
                'quality_score': entry.get('quality_score', 0.8),
                'metadata': {
                    'source': 'playbook',
                    'agent_id': entry.get('agent_id', 'unknown'),
                    'usage_count': entry.get('usage_count', 1),
                    'tags': entry.get('tags', [])
                }
            }
            
            examples.append(example)
        
        return examples
    
    def _filter_examples(self, examples: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Filtra ejemplos por calidad y relevancia."""
        filtered = []
        
        for example in examples:
            # Verificar calidad mínima
            quality_score = example.get('quality_score', 0.0)
            success_rate = example.get('success_rate', 0.0)
            
            if quality_score < 0.6 or success_rate < 0.7:
                continue
            
            # Verificar que el ejemplo sea relevante para el dominio
            if not self._is_relevant_to_domain(example, domain):
                continue
            
            # Verificar que tenga contenido válido
            if not example.get('input_text') or not example.get('output_code'):
                continue
            
            filtered.append(example)
        
        return filtered
    
    def _is_relevant_to_domain(self, example: Dict[str, Any], domain: str) -> bool:
        """Verifica si un ejemplo es relevante para el dominio."""
        if domain not in self.expert_domains:
            return False
        
        expert_domain = self.expert_domains[domain]
        
        # Verificar por dominio del ejemplo
        example_domain = example.get('domain', '').lower()
        if example_domain == domain:
            return True
        
        # Verificar por keywords en el input
        input_text = example.get('input_text', '').lower()
        for keyword in expert_domain.keywords:
            if keyword.lower() in input_text:
                return True
        
        # Verificar por keywords en el código
        output_code = example.get('output_code', '').lower()
        for keyword in expert_domain.keywords:
            if keyword.lower() in output_code:
                return True
        
        return False
    
    def _determine_difficulty_from_quality(self, quality_score: float) -> str:
        """Determina dificultad basada en score de calidad."""
        if quality_score >= 0.9:
            return "advanced"
        elif quality_score >= 0.7:
            return "intermediate"
        else:
            return "beginner"
    
    def _split_examples(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Divide ejemplos en train/validation/test."""
        # Mezclar ejemplos
        np.random.shuffle(examples)
        
        total = len(examples)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.validation_ratio)
        
        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:]
        
        return train_examples, val_examples, test_examples
    
    def _calculate_quality_metrics(self, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula métricas de calidad del dataset."""
        if not examples:
            return {}
        
        quality_scores = [ex.get('quality_score', 0.0) for ex in examples]
        success_rates = [ex.get('success_rate', 0.0) for ex in examples]
        
        return {
            'avg_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'std_quality_score': np.std(quality_scores),
            'avg_success_rate': np.mean(success_rates),
            'min_success_rate': np.min(success_rates),
            'max_success_rate': np.max(success_rates)
        }
    
    def _calculate_domain_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calcula distribución por dominio."""
        domain_counter = Counter()
        
        for example in examples:
            domain = example.get('domain', 'unknown')
            domain_counter[domain] += 1
        
        return dict(domain_counter)
    
    def _calculate_difficulty_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calcula distribución por dificultad."""
        difficulty_counter = Counter()
        
        for example in examples:
            difficulty = example.get('difficulty_level', 'unknown')
            difficulty_counter[difficulty] += 1
        
        return dict(difficulty_counter)
    
    def _generate_routing_dataset(self, moe_datasets: Dict[str, MoEDataset]) -> Optional[MoEDataset]:
        """Genera dataset de routing para expertos."""
        try:
            routing_examples = []
            
            for domain, dataset in moe_datasets.items():
                if domain == 'routing':
                    continue
                
                # Crear ejemplos de routing para este dominio
                domain_routing_examples = self._create_routing_examples_for_domain(domain, dataset)
                routing_examples.extend(domain_routing_examples)
            
            if not routing_examples:
                logger.warning("No se pudieron generar ejemplos de routing")
                return None
            
            # Dividir en train/validation/test
            train_examples, val_examples, test_examples = self._split_examples(routing_examples)
            
            # Calcular métricas
            quality_metrics = self._calculate_quality_metrics(routing_examples)
            domain_distribution = self._calculate_domain_distribution(routing_examples)
            difficulty_distribution = self._calculate_difficulty_distribution(routing_examples)
            
            routing_dataset = MoEDataset(
                dataset_id=f"moe_routing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                expert_domain='routing',
                total_examples=len(routing_examples),
                training_examples=train_examples,
                validation_examples=val_examples,
                test_examples=test_examples,
                domain_distribution=domain_distribution,
                difficulty_distribution=difficulty_distribution,
                quality_metrics=quality_metrics,
                generation_date=datetime.now(),
                metadata={
                    'routing_examples': len(routing_examples),
                    'source_domains': list(moe_datasets.keys())
                }
            )
            
            self.generation_stats['routing_examples_generated'] += len(routing_examples)
            
            logger.info(f"Dataset de routing generado: {len(routing_examples)} ejemplos")
            return routing_dataset
            
        except Exception as e:
            logger.error(f"Error generando dataset de routing: {e}")
            return None
    
    def _create_routing_examples_for_domain(self, domain: str, dataset: MoEDataset) -> List[Dict[str, Any]]:
        """Crea ejemplos de routing para un dominio específico."""
        routing_examples = []
        
        # Tomar una muestra de ejemplos del dataset
        sample_size = min(100, len(dataset.training_examples))
        sample_examples = np.random.choice(dataset.training_examples, sample_size, replace=False)
        
        for example in sample_examples:
            # Crear ejemplo de routing
            routing_example = {
                'example_id': f"routing_{domain}_{example.get('example_id', 'unknown')}",
                'language': 'routing',
                'input_text': f"Route this query to the appropriate expert: {example.get('input_text', '')}",
                'output_code': domain,  # El experto correcto
                'explanation': f"This query should be routed to the {domain} expert based on the content and context.",
                'difficulty_level': example.get('difficulty_level', 'intermediate'),
                'domain': 'routing',
                'success_rate': 1.0,  # Routing correcto
                'quality_score': 0.9,  # Alta calidad de routing
                'metadata': {
                    'source': 'routing',
                    'target_expert': domain,
                    'original_example_id': example.get('example_id', 'unknown'),
                    'routing_features': self._extract_routing_features(example, domain)
                }
            }
            
            routing_examples.append(routing_example)
        
        return routing_examples
    
    def _extract_routing_features(self, example: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Extrae características para routing."""
        input_text = example.get('input_text', '')
        output_code = example.get('output_code', '')
        
        # Características básicas
        features = {
            'input_length': len(input_text),
            'code_length': len(output_code),
            'difficulty_level': example.get('difficulty_level', 'intermediate'),
            'quality_score': example.get('quality_score', 0.0),
            'success_rate': example.get('success_rate', 0.0)
        }
        
        # Características de dominio
        expert_domain = self.expert_domains.get(domain)
        if expert_domain:
            features['domain_keywords_found'] = sum(
                1 for keyword in expert_domain.keywords 
                if keyword.lower() in input_text.lower()
            )
            features['domain_specializations'] = expert_domain.specializations
        
        return features
    
    def _save_moe_datasets(self, moe_datasets: Dict[str, MoEDataset]):
        """Guarda datasets MoE."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for domain, dataset in moe_datasets.items():
                # Convertir a diccionario para JSON
                dataset_dict = {
                    'dataset_id': dataset.dataset_id,
                    'expert_domain': dataset.expert_domain,
                    'total_examples': dataset.total_examples,
                    'training_examples': dataset.training_examples,
                    'validation_examples': dataset.validation_examples,
                    'test_examples': dataset.test_examples,
                    'domain_distribution': dataset.domain_distribution,
                    'difficulty_distribution': dataset.difficulty_distribution,
                    'quality_metrics': dataset.quality_metrics,
                    'generation_date': dataset.generation_date.isoformat(),
                    'metadata': dataset.metadata
                }
                
                # Guardar archivo
                filename = f"moe_dataset_{domain}_{timestamp}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Dataset MoE guardado: {filepath}")
                self.generation_stats['total_datasets_generated'] += 1
                
        except Exception as e:
            logger.error(f"Error guardando datasets MoE: {e}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de generación."""
        return self.generation_stats.copy()
    
    def analyze_dataset_quality(self, domain: str) -> Dict[str, Any]:
        """Analiza la calidad de un dataset MoE."""
        try:
            # Buscar archivo de dataset
            dataset_files = [f for f in os.listdir(self.output_dir) 
                           if f.startswith(f"moe_dataset_{domain}_") and f.endswith(".json")]
            
            if not dataset_files:
                return {'error': f'No se encontró dataset para el dominio {domain}'}
            
            # Cargar el dataset más reciente
            latest_file = sorted(dataset_files)[-1]
            filepath = os.path.join(self.output_dir, latest_file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            
            # Análisis de calidad
            analysis = {
                'domain': domain,
                'dataset_id': dataset_data.get('dataset_id', ''),
                'total_examples': dataset_data.get('total_examples', 0),
                'quality_metrics': dataset_data.get('quality_metrics', {}),
                'domain_distribution': dataset_data.get('domain_distribution', {}),
                'difficulty_distribution': dataset_data.get('difficulty_distribution', {}),
                'generation_date': dataset_data.get('generation_date', ''),
                'metadata': dataset_data.get('metadata', {})
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando calidad del dataset {domain}: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Test del MoEDatasetGenerator
    logging.basicConfig(level=logging.INFO)
    
    generator = MoEDatasetGenerator()
    
    # Generar datasets MoE
    moe_datasets = generator.generate_moe_datasets(domains=['python', 'sql', 'javascript'])
    print(f"Datasets MoE generados: {len(moe_datasets)}")
    
    # Mostrar estadísticas
    stats = generator.get_generation_stats()
    print(f"Estadísticas de generación: {stats}")
    
    # Analizar calidad
    for domain in moe_datasets.keys():
        analysis = generator.analyze_dataset_quality(domain)
        print(f"Análisis de calidad {domain}: {analysis}")
