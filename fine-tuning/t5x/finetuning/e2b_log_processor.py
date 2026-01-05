#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2B Log Processor - Procesamiento de logs E2B para generación de datasets de fine-tuning.
"""

import logging
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class E2BExecutionLog:
    """Log de ejecución E2B."""
    execution_id: str
    agent_id: str
    language: str
    code: str
    output: str
    error: Optional[str]
    success: bool
    execution_time_ms: int
    memory_used_mb: float
    cpu_used_percent: float
    corrections_applied: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class CodePattern:
    """Patrón de código extraído."""
    pattern_id: str
    language: str
    code_type: str  # function, class, script, expression
    code_snippet: str
    context: str
    success_rate: float
    usage_count: int
    avg_execution_time: float
    common_errors: List[str]
    tags: List[str]


@dataclass
class TrainingExample:
    """Ejemplo de entrenamiento generado."""
    example_id: str
    language: str
    input_text: str
    output_code: str
    explanation: str
    difficulty_level: str  # beginner, intermediate, advanced
    domain: str
    success_rate: float
    quality_score: float
    metadata: Dict[str, Any]


class E2BLogProcessor:
    """Procesador de logs E2B para fine-tuning."""
    
    def __init__(self, 
                 log_dir: str = "backend/data/e2b_logs",
                 output_dir: str = "backend/data/training_datasets",
                 min_success_rate: float = 0.7):
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.min_success_rate = min_success_rate
        
        # Configuración de procesamiento
        self.min_code_length = 10
        self.max_code_length = 2000
        self.min_usage_count = 3
        
        # Patrones de código por lenguaje
        self.code_patterns = {
            'python': {
                'function': r'def\s+\w+\s*\([^)]*\)\s*:',
                'class': r'class\s+\w+\s*[\(:]',
                'import': r'(import|from)\s+\w+',
                'loop': r'(for|while)\s+\w+',
                'condition': r'if\s+.*:',
                'list_comp': r'\[.*for.*in.*\]',
                'lambda': r'lambda\s+.*:'
            },
            'javascript': {
                'function': r'function\s+\w+\s*\([^)]*\)\s*\{',
                'arrow_function': r'\w+\s*=\s*\([^)]*\)\s*=>',
                'class': r'class\s+\w+\s*\{',
                'import': r'(import|require)\s+.*',
                'loop': r'(for|while)\s*\(',
                'condition': r'if\s*\(',
                'async': r'async\s+function'
            },
            'sql': {
                'select': r'SELECT\s+.*FROM',
                'insert': r'INSERT\s+INTO',
                'update': r'UPDATE\s+\w+\s+SET',
                'delete': r'DELETE\s+FROM',
                'join': r'JOIN\s+\w+',
                'where': r'WHERE\s+.*',
                'group_by': r'GROUP\s+BY'
            }
        }
        
        # Estadísticas
        self.processing_stats = {
            'total_logs_processed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'patterns_extracted': 0,
            'training_examples_generated': 0,
            'languages_processed': defaultdict(int),
            'processing_time_seconds': 0
        }
        
        # Asegurar directorios
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"E2BLogProcessor inicializado: log_dir={log_dir}, output_dir={output_dir}")
    
    def process_e2b_logs(self, 
                        time_range_days: int = 30,
                        languages: Optional[List[str]] = None,
                        min_quality_score: float = 0.6) -> Dict[str, Any]:
        """Procesa logs E2B y genera datasets de entrenamiento."""
        start_time = datetime.now()
        logger.info(f"Iniciando procesamiento de logs E2B (últimos {time_range_days} días)")
        
        try:
            # Obtener logs del rango de tiempo
            logs = self._get_logs_in_time_range(time_range_days, languages)
            
            if not logs:
                logger.warning("No se encontraron logs E2B en el rango de tiempo especificado")
                return {}
            
            # Procesar logs por lenguaje
            processed_data = {}
            
            for language, language_logs in logs.items():
                logger.info(f"Procesando {len(language_logs)} logs de {language}")
                
                # Extraer patrones de código
                patterns = self._extract_code_patterns(language, language_logs)
                
                # Generar ejemplos de entrenamiento
                training_examples = self._generate_training_examples(language, patterns, language_logs)
                
                # Filtrar por calidad
                filtered_examples = [
                    ex for ex in training_examples
                    if ex.quality_score >= min_quality_score
                ]
                
                processed_data[language] = {
                    'patterns': patterns,
                    'training_examples': filtered_examples,
                    'total_logs': len(language_logs),
                    'successful_logs': len([log for log in language_logs if log.success]),
                    'patterns_count': len(patterns),
                    'examples_count': len(filtered_examples)
                }
                
                self.processing_stats['languages_processed'][language] = len(language_logs)
            
            # Guardar datasets
            self._save_training_datasets(processed_data)
            
            # Actualizar estadísticas
            self.processing_stats['processing_time_seconds'] = (
                datetime.now() - start_time
            ).total_seconds()
            
            logger.info(f"Procesamiento completado: {len(processed_data)} lenguajes procesados")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error procesando logs E2B: {e}")
            return {}
    
    def _get_logs_in_time_range(self, 
                               time_range_days: int, 
                               languages: Optional[List[str]] = None) -> Dict[str, List[E2BExecutionLog]]:
        """Obtiene logs E2B en un rango de tiempo."""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            logs_by_language = defaultdict(list)
            
            # Buscar archivos de logs
            if not os.path.exists(self.log_dir):
                logger.warning(f"Directorio de logs no existe: {self.log_dir}")
                return {}
            
            for filename in os.listdir(self.log_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(self.log_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    # Verificar si es un log de ejecución E2B
                    if self._is_e2b_execution_log(log_data):
                        log = self._parse_execution_log(log_data)
                        
                        # Filtrar por fecha
                        if log.timestamp >= cutoff_date:
                            # Filtrar por lenguaje si se especifica
                            if languages and log.language not in languages:
                                continue
                            
                            logs_by_language[log.language].append(log)
                            self.processing_stats['total_logs_processed'] += 1
                            
                            if log.success:
                                self.processing_stats['successful_executions'] += 1
                            else:
                                self.processing_stats['failed_executions'] += 1
                
                except Exception as e:
                    logger.error(f"Error procesando archivo de log {filename}: {e}")
                    continue
            
            return dict(logs_by_language)
            
        except Exception as e:
            logger.error(f"Error obteniendo logs en rango de tiempo: {e}")
            return {}
    
    def _is_e2b_execution_log(self, log_data: Dict[str, Any]) -> bool:
        """Verifica si un log es de ejecución E2B."""
        required_fields = ['execution_id', 'agent_id', 'language', 'code', 'success']
        return all(field in log_data for field in required_fields)
    
    def _parse_execution_log(self, log_data: Dict[str, Any]) -> E2BExecutionLog:
        """Parsea un log de ejecución E2B."""
        return E2BExecutionLog(
            execution_id=log_data.get('execution_id', ''),
            agent_id=log_data.get('agent_id', ''),
            language=log_data.get('language', ''),
            code=log_data.get('code', ''),
            output=log_data.get('output', ''),
            error=log_data.get('error'),
            success=log_data.get('success', False),
            execution_time_ms=log_data.get('execution_time_ms', 0),
            memory_used_mb=log_data.get('memory_used_mb', 0.0),
            cpu_used_percent=log_data.get('cpu_used_percent', 0.0),
            corrections_applied=log_data.get('corrections_applied', 0),
            timestamp=datetime.fromisoformat(log_data.get('timestamp', datetime.now().isoformat())),
            metadata=log_data.get('metadata', {})
        )
    
    def _extract_code_patterns(self, 
                             language: str, 
                             logs: List[E2BExecutionLog]) -> List[CodePattern]:
        """Extrae patrones de código de los logs."""
        try:
            patterns = []
            pattern_counter = defaultdict(int)
            pattern_success = defaultdict(list)
            pattern_errors = defaultdict(list)
            pattern_times = defaultdict(list)
            
            # Agrupar código por patrones
            for log in logs:
                if not log.success or len(log.code) < self.min_code_length:
                    continue
                
                # Detectar tipo de código
                code_type = self._detect_code_type(language, log.code)
                
                if code_type:
                    pattern_key = f"{code_type}_{self._normalize_code(log.code)}"
                    pattern_counter[pattern_key] += 1
                    pattern_success[pattern_key].append(log.success)
                    pattern_times[pattern_key].append(log.execution_time_ms)
                    
                    if log.error:
                        pattern_errors[pattern_key].append(log.error)
            
            # Crear patrones
            for pattern_key, count in pattern_counter.items():
                if count >= self.min_usage_count:
                    success_rate = np.mean(pattern_success[pattern_key])
                    
                    if success_rate >= self.min_success_rate:
                        # Extraer snippet representativo
                        code_snippet = self._extract_code_snippet(pattern_key, logs)
                        
                        pattern = CodePattern(
                            pattern_id=f"pattern_{hash(pattern_key) % 100000:05d}",
                            language=language,
                            code_type=pattern_key.split('_')[0],
                            code_snippet=code_snippet,
                            context=self._generate_context(code_snippet, language),
                            success_rate=success_rate,
                            usage_count=count,
                            avg_execution_time=np.mean(pattern_times[pattern_key]),
                            common_errors=list(set(pattern_errors[pattern_key]))[:5],
                            tags=self._generate_tags(code_snippet, language, success_rate)
                        )
                        
                        patterns.append(pattern)
                        self.processing_stats['patterns_extracted'] += 1
            
            # Ordenar por uso y éxito
            patterns.sort(key=lambda p: (p.usage_count * p.success_rate), reverse=True)
            
            logger.info(f"Extraídos {len(patterns)} patrones de código para {language}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error extrayendo patrones de código para {language}: {e}")
            return []
    
    def _detect_code_type(self, language: str, code: str) -> Optional[str]:
        """Detecta el tipo de código."""
        if language not in self.code_patterns:
            return None
        
        patterns = self.code_patterns[language]
        
        for code_type, pattern in patterns.items():
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                return code_type
        
        return 'script'  # Tipo por defecto
    
    def _normalize_code(self, code: str) -> str:
        """Normaliza código para agrupación."""
        # Remover comentarios
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalizar espacios
        code = re.sub(r'\s+', ' ', code)
        code = code.strip()
        
        # Truncar si es muy largo
        if len(code) > 200:
            code = code[:200] + "..."
        
        return code
    
    def _extract_code_snippet(self, pattern_key: str, logs: List[E2BExecutionLog]) -> str:
        """Extrae snippet de código representativo."""
        # Buscar el código más exitoso para este patrón
        matching_logs = []
        
        for log in logs:
            if log.success and self._normalize_code(log.code) in pattern_key:
                matching_logs.append(log)
        
        if matching_logs:
            # Seleccionar el más rápido y exitoso
            best_log = min(matching_logs, key=lambda l: l.execution_time_ms)
            return best_log.code
        
        return ""
    
    def _generate_context(self, code: str, language: str) -> str:
        """Genera contexto para el código."""
        context_parts = []
        
        # Detectar propósito del código
        if 'def ' in code or 'function ' in code:
            context_parts.append("Function definition")
        elif 'class ' in code:
            context_parts.append("Class definition")
        elif 'import ' in code or 'require ' in code:
            context_parts.append("Import statement")
        elif 'for ' in code or 'while ' in code:
            context_parts.append("Loop structure")
        elif 'if ' in code:
            context_parts.append("Conditional statement")
        else:
            context_parts.append("Code script")
        
        # Agregar información del lenguaje
        context_parts.append(f"{language} programming")
        
        return " | ".join(context_parts)
    
    def _generate_tags(self, code: str, language: str, success_rate: float) -> List[str]:
        """Genera tags para el código."""
        tags = [language]
        
        # Tags de calidad
        if success_rate >= 0.9:
            tags.append("high_success")
        elif success_rate >= 0.8:
            tags.append("good_success")
        
        # Tags de complejidad
        if len(code.split('\n')) > 10:
            tags.append("complex")
        elif len(code.split('\n')) > 5:
            tags.append("medium")
        else:
            tags.append("simple")
        
        # Tags de funcionalidad
        if 'def ' in code or 'function ' in code:
            tags.append("function")
        if 'class ' in code:
            tags.append("class")
        if 'import ' in code or 'require ' in code:
            tags.append("import")
        if 'for ' in code or 'while ' in code:
            tags.append("loop")
        if 'if ' in code:
            tags.append("conditional")
        
        return tags
    
    def _generate_training_examples(self, 
                                  language: str, 
                                  patterns: List[CodePattern],
                                  logs: List[E2BExecutionLog]) -> List[TrainingExample]:
        """Genera ejemplos de entrenamiento."""
        try:
            examples = []
            
            for pattern in patterns:
                # Generar ejemplos para cada patrón
                pattern_examples = self._create_examples_from_pattern(pattern, logs)
                examples.extend(pattern_examples)
            
            # Generar ejemplos adicionales de logs exitosos
            successful_logs = [log for log in logs if log.success and log.quality_score >= 0.7]
            
            for log in successful_logs[:100]:  # Limitar para evitar demasiados ejemplos
                example = self._create_example_from_log(log)
                if example:
                    examples.append(example)
            
            self.processing_stats['training_examples_generated'] += len(examples)
            
            logger.info(f"Generados {len(examples)} ejemplos de entrenamiento para {language}")
            return examples
            
        except Exception as e:
            logger.error(f"Error generando ejemplos de entrenamiento para {language}: {e}")
            return []
    
    def _create_examples_from_pattern(self, 
                                    pattern: CodePattern, 
                                    logs: List[E2BExecutionLog]) -> List[TrainingExample]:
        """Crea ejemplos de entrenamiento a partir de un patrón."""
        examples = []
        
        # Buscar logs que coincidan con el patrón
        matching_logs = []
        for log in logs:
            if (log.language == pattern.language and 
                log.success and 
                self._normalize_code(log.code) in pattern.code_snippet):
                matching_logs.append(log)
        
        if not matching_logs:
            return examples
        
        # Crear ejemplo principal del patrón
        main_example = TrainingExample(
            example_id=f"example_{pattern.pattern_id}_main",
            language=pattern.language,
            input_text=self._generate_input_text(pattern),
            output_code=pattern.code_snippet,
            explanation=self._generate_explanation(pattern),
            difficulty_level=self._determine_difficulty(pattern),
            domain=pattern.language,
            success_rate=pattern.success_rate,
            quality_score=pattern.success_rate,
            metadata={
                'pattern_id': pattern.pattern_id,
                'code_type': pattern.code_type,
                'usage_count': pattern.usage_count,
                'avg_execution_time': pattern.avg_execution_time
            }
        )
        
        examples.append(main_example)
        
        # Crear variaciones del patrón
        for i, log in enumerate(matching_logs[:3]):  # Máximo 3 variaciones
            if log.code != pattern.code_snippet:
                variation_example = TrainingExample(
                    example_id=f"example_{pattern.pattern_id}_var_{i}",
                    language=pattern.language,
                    input_text=self._generate_input_text(pattern, log),
                    output_code=log.code,
                    explanation=self._generate_explanation(pattern, log),
                    difficulty_level=self._determine_difficulty(pattern),
                    domain=pattern.language,
                    success_rate=1.0,  # Log exitoso
                    quality_score=0.8,
                    metadata={
                        'pattern_id': pattern.pattern_id,
                        'variation': i,
                        'execution_time': log.execution_time_ms
                    }
                )
                
                examples.append(variation_example)
        
        return examples
    
    def _create_example_from_log(self, log: E2BExecutionLog) -> Optional[TrainingExample]:
        """Crea ejemplo de entrenamiento a partir de un log."""
        try:
            # Generar input text basado en el contexto
            input_text = self._generate_input_from_log(log)
            
            if not input_text:
                return None
            
            # Generar explicación
            explanation = self._generate_explanation_from_log(log)
            
            # Determinar dificultad
            difficulty = self._determine_difficulty_from_log(log)
            
            example = TrainingExample(
                example_id=f"example_log_{log.execution_id}",
                language=log.language,
                input_text=input_text,
                output_code=log.code,
                explanation=explanation,
                difficulty_level=difficulty,
                domain=log.language,
                success_rate=1.0,
                quality_score=0.8,
                metadata={
                    'execution_id': log.execution_id,
                    'agent_id': log.agent_id,
                    'execution_time': log.execution_time_ms,
                    'corrections_applied': log.corrections_applied
                }
            )
            
            return example
            
        except Exception as e:
            logger.error(f"Error creando ejemplo desde log {log.execution_id}: {e}")
            return None
    
    def _generate_input_text(self, pattern: CodePattern, log: Optional[E2BExecutionLog] = None) -> str:
        """Genera texto de entrada para el ejemplo."""
        base_prompts = {
            'function': f"Write a {pattern.language} function that",
            'class': f"Create a {pattern.language} class that",
            'script': f"Write {pattern.language} code that",
            'import': f"Import the necessary modules in {pattern.language}",
            'loop': f"Write a {pattern.language} loop that",
            'conditional': f"Write a {pattern.language} conditional statement that"
        }
        
        base_prompt = base_prompts.get(pattern.code_type, f"Write {pattern.language} code that")
        
        # Agregar contexto específico si hay log
        if log and hasattr(log, 'context') and log.context:
            return f"{base_prompt} {log.context}"
        
        # Generar contexto genérico
        context = f"demonstrates {pattern.code_type} usage"
        return f"{base_prompt} {context}"
    
    def _generate_explanation(self, pattern: CodePattern, log: Optional[E2BExecutionLog] = None) -> str:
        """Genera explicación para el ejemplo."""
        explanations = {
            'function': f"This {pattern.language} function demonstrates proper function definition and usage.",
            'class': f"This {pattern.language} class shows object-oriented programming concepts.",
            'script': f"This {pattern.language} script performs the requested functionality.",
            'import': f"This {pattern.language} import statement loads the necessary modules.",
            'loop': f"This {pattern.language} loop iterates through data efficiently.",
            'conditional': f"This {pattern.language} conditional statement handles different cases."
        }
        
        base_explanation = explanations.get(pattern.code_type, f"This {pattern.language} code demonstrates the requested functionality.")
        
        # Agregar información de rendimiento si está disponible
        if pattern.avg_execution_time > 0:
            base_explanation += f" It executes in approximately {pattern.avg_execution_time:.0f}ms."
        
        return base_explanation
    
    def _determine_difficulty(self, pattern: CodePattern) -> str:
        """Determina el nivel de dificultad del patrón."""
        code_length = len(pattern.code_snippet.split('\n'))
        
        if code_length <= 3:
            return "beginner"
        elif code_length <= 10:
            return "intermediate"
        else:
            return "advanced"
    
    def _generate_input_from_log(self, log: E2BExecutionLog) -> Optional[str]:
        """Genera input text desde un log."""
        # Intentar extraer contexto del metadata
        context = log.metadata.get('context', '')
        user_intent = log.metadata.get('user_intent', '')
        
        if context and user_intent:
            return f"Context: {context}\nUser intent: {user_intent}\nWrite {log.language} code to solve this."
        elif context:
            return f"Context: {context}\nWrite {log.language} code to solve this."
        elif user_intent:
            return f"User intent: {user_intent}\nWrite {log.language} code to solve this."
        else:
            # Generar input genérico basado en el código
            code_type = self._detect_code_type(log.language, log.code)
            if code_type:
                return f"Write {log.language} {code_type} code that demonstrates best practices."
        
        return None
    
    def _generate_explanation_from_log(self, log: E2BExecutionLog) -> str:
        """Genera explicación desde un log."""
        explanation = f"This {log.language} code was successfully executed"
        
        if log.execution_time_ms > 0:
            explanation += f" in {log.execution_time_ms}ms"
        
        if log.corrections_applied > 0:
            explanation += f" with {log.corrections_applied} corrections applied"
        
        explanation += "."
        
        return explanation
    
    def _determine_difficulty_from_log(self, log: E2BExecutionLog) -> str:
        """Determina dificultad desde un log."""
        code_length = len(log.code.split('\n'))
        
        if code_length <= 3:
            return "beginner"
        elif code_length <= 10:
            return "intermediate"
        else:
            return "advanced"
    
    def _save_training_datasets(self, processed_data: Dict[str, Any]):
        """Guarda datasets de entrenamiento."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for language, data in processed_data.items():
                # Preparar datos para JSON
                dataset = {
                    'language': language,
                    'generation_date': datetime.now().isoformat(),
                    'total_logs': data['total_logs'],
                    'successful_logs': data['successful_logs'],
                    'patterns_count': data['patterns_count'],
                    'examples_count': data['examples_count'],
                    'patterns': [asdict(pattern) for pattern in data['patterns']],
                    'training_examples': [asdict(example) for example in data['training_examples']]
                }
                
                # Guardar archivo
                filename = f"training_dataset_{language}_{timestamp}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Dataset de entrenamiento guardado: {filepath}")
                
        except Exception as e:
            logger.error(f"Error guardando datasets de entrenamiento: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de procesamiento."""
        return dict(self.processing_stats)


if __name__ == "__main__":
    # Test del E2BLogProcessor
    logging.basicConfig(level=logging.INFO)
    
    processor = E2BLogProcessor()
    
    # Crear logs de prueba
    test_logs = [
        {
            'execution_id': 'exec_001',
            'agent_id': 'test_agent_001',
            'language': 'python',
            'code': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
            'output': 'Function defined successfully',
            'error': None,
            'success': True,
            'execution_time_ms': 50,
            'memory_used_mb': 10.0,
            'cpu_used_percent': 5.0,
            'corrections_applied': 0,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'context': 'Calculate Fibonacci numbers',
                'user_intent': 'Learn recursive functions'
            }
        },
        {
            'execution_id': 'exec_002',
            'agent_id': 'test_agent_001',
            'language': 'python',
            'code': 'numbers = [1, 2, 3, 4, 5]\nsquared = [x**2 for x in numbers]\nprint(squared)',
            'output': '[1, 4, 9, 16, 25]',
            'error': None,
            'success': True,
            'execution_time_ms': 30,
            'memory_used_mb': 5.0,
            'cpu_used_percent': 3.0,
            'corrections_applied': 0,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'context': 'List comprehension example',
                'user_intent': 'Learn list comprehensions'
            }
        }
    ]
    
    # Guardar logs de prueba
    for i, log in enumerate(test_logs):
        filename = f"e2b_log_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(processor.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    
    # Procesar logs
    processed_data = processor.process_e2b_logs(time_range_days=1, languages=['python'])
    print(f"Datos procesados: {len(processed_data)} lenguajes")
    
    # Mostrar estadísticas
    stats = processor.get_processing_stats()
    print(f"Estadísticas de procesamiento: {stats}")