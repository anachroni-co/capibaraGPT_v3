#!/usr/bin/env python3
"""
Format Manager Ultra Optimized - Gestiona conversiones automáticas entre JSON y TOON con latencia mínima
"""

import json
from typing import Any, Dict, List, Tuple, Union, Optional

from .parser_ultra_optimized import ToonParserUltraOptimized
from .encoder_ultra_optimized import ToonEncoderUltraOptimized


class FormatManagerUltraOptimized:
    """
    Gestor de formatos ultra optimizado que decide automáticamente cuándo usar TOON vs JSON
    con latencia mínima gracias a mecanismos avanzados de caché
    """

    # Constantes
    MIN_ARRAY_SIZE_FOR_TOON = 5  # Mínimo elementos para considerar TOON
    MIN_SAVINGS_PERCENT = 20      # Mínimo % de ahorro para recomendar TOON
    
    # Caché para decisiones y conversiones
    _conversion_cache = {}
    _decision_cache = {}
    _analysis_cache = {}

    @classmethod
    def encode(
        cls,
        data: Union[Dict, List],
        preferred_format: str = 'auto',
        key: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Codifica datos en el formato óptimo (versión ultra optimizada)
        """
        # Crear clave de caché para esta combinación específica
        try:
            data_hash = hash(json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data))
            cache_key = (data_hash, preferred_format, key)
            if cache_key in cls._conversion_cache:
                return cls._conversion_cache[cache_key]
        except TypeError:
            # Si no se puede hashear, continuar sin caché
            pass

        # Formato explícito
        if preferred_format == 'json':
            result = json.dumps(data, separators=(',', ':')), 'json'
        elif preferred_format == 'toon':
            result = ToonEncoderUltraOptimized.encode(data, key), 'toon'
        elif preferred_format == 'compact_toon':
            result = ToonEncoderUltraOptimized.encode_compact_ultra_optimized(data, key), 'toon'
        elif preferred_format == 'auto':
            # Decidir basado en eficiencia (ultra rápido)
            should_use_toon = cls.should_use_toon_ultra_optimized(data)
            if should_use_toon:
                result = ToonEncoderUltraOptimized.encode(data, key), 'toon'
            else:
                result = json.dumps(data, separators=(',', ':')), 'json'
        else:
            # Default: JSON
            result = json.dumps(data, separators=(',', ':')), 'json'

        # Guardar en caché si es posible
        try:
            if cache_key not in cls._conversion_cache:
                cls._conversion_cache[cache_key] = result
                # Limitar tamaño de caché
                if len(cls._conversion_cache) > 1000:
                    oldest_key = next(iter(cls._conversion_cache))
                    del cls._conversion_cache[oldest_key]
        except:
            pass

        return result

    @classmethod
    def decode(
        cls,
        content: str,
        format_type: str = 'auto'
    ) -> Union[Dict, List]:
        """
        Decodifica contenido del formato especificado (versión ultra optimizada)
        """
        # Cachear resultados de decodificación
        cache_key = (content, format_type)
        if cache_key in cls._conversion_cache:
            return cls._conversion_cache[cache_key]

        if not content or not content.strip():
            result = {}
        else:
            # Detección automática ultra rápida
            if format_type == 'auto':
                format_type = cls.detect_format_ultra_optimized(content)

            if format_type == 'toon':
                result = ToonParserUltraOptimized.parse(content)
            else:
                # JSON
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Si falla JSON, intentar TOON como fallback
                    try:
                        result = ToonParserUltraOptimized.parse(content)
                    except Exception:
                        raise ValueError(f"No se pudo decodificar el contenido en ningún formato válido")

        # Cachear resultado
        try:
            cls._conversion_cache[cache_key] = result
            # Limitar tamaño de caché
            if len(cls._conversion_cache) > 1000:
                oldest_key = next(iter(cls._conversion_cache))
                del cls._conversion_cache[oldest_key]
        except:
            pass

        return result

    @classmethod
    def detect_format_ultra_optimized(cls, content: str) -> str:
        """
        Detecta automáticamente el formato del contenido (versión ultra optimizada)
        """
        content = content.strip()

        # JSON típicamente empieza con { o [
        if content and content[0] in ('{', '['):
            return 'json'

        # TOON tiene patrones característicos: key[n]{...}: o key{...}:
        # Usar detección ultra rápida sin regex
        if ':' in content:
            colon_idx = content.find(':')
            if colon_idx > 0:  # Asegurarse de que hay algo antes del :
                before_colon = content[:colon_idx]
                # Buscar patrones específicos de TOON más rápidamente
                if ('[' in before_colon and ']' in before_colon and 
                    '{' in before_colon and '}' in before_colon):
                    return 'toon'

        # Default: JSON
        return 'json'

    @classmethod
    def should_use_toon_ultra_optimized(cls, data: Any) -> bool:
        """
        Determina si TOON es más eficiente que JSON para los datos dados (versión ultra optimizada)
        """
        # Usar caché para decisiones repetidas
        try:
            data_hash = hash(json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data))
            cache_key = ('decision', data_hash)
            if cache_key in cls._decision_cache:
                return cls._decision_cache[cache_key]
        except TypeError:
            pass

        # Verificar si hay arrays grandes de objetos uniformes
        if isinstance(data, dict):
            for key, value in data.items():
                if cls._is_large_uniform_array_ultra_optimized(value):
                    # Verificar ahorro real ultra rápido
                    savings_data = {key: value}
                    savings_hash = hash(json.dumps(savings_data, sort_keys=True))
                    savings_cache_key = ('savings', savings_hash)
                    
                    if savings_cache_key in cls._conversion_cache:
                        recommended = cls._conversion_cache[savings_cache_key]['recommended']
                    else:
                        recommended = ToonEncoderUltraOptimized.estimate_token_savings_ultra_optimized(savings_data)['recommended']
                    
                    # Cachear decisión
                    try:
                        cls._decision_cache[cache_key] = recommended
                        if len(cls._decision_cache) > 500:
                            oldest_key = next(iter(cls._decision_cache))
                            del cls._decision_cache[oldest_key]
                    except:
                        pass
                    
                    return recommended
        elif cls._is_large_uniform_array_ultra_optimized(data):
            data_for_savings = {"data": data}
            savings_hash = hash(json.dumps(data_for_savings, sort_keys=True))
            savings_cache_key = ('savings', savings_hash)
            
            if savings_cache_key in cls._conversion_cache:
                recommended = cls._conversion_cache[savings_cache_key]['recommended']
            else:
                recommended = ToonEncoderUltraOptimized.estimate_token_savings_ultra_optimized(data_for_savings)['recommended']
            
            # Cachear decisión
            try:
                cls._decision_cache[cache_key] = recommended
                if len(cls._decision_cache) > 500:
                    oldest_key = next(iter(cls._decision_cache))
                    del cls._decision_cache[oldest_key]
            except:
                pass
            
            return recommended

        # Cachear decisión de no usar TOON
        try:
            cls._decision_cache[cache_key] = False
            if len(cls._decision_cache) > 500:
                oldest_key = next(iter(cls._decision_cache))
                del cls._decision_cache[oldest_key]
        except:
            pass

        return False

    @classmethod
    def _is_large_uniform_array_ultra_optimized(cls, data: Any) -> bool:
        """
        Verifica si los datos son un array grande de objetos con la misma estructura (versión ultra optimizada)
        """
        if not isinstance(data, list):
            return False

        if len(data) < cls.MIN_ARRAY_SIZE_FOR_TOON:
            return False

        # Verificar que todos sean dicts
        if not data or not all(isinstance(item, dict) for item in data):
            return False

        # Verificar que tengan las mismas claves - ultra optimizado
        first_element = data[0]
        first_keys = frozenset(first_element.keys())  # frozenset es más rápido para comparaciones
        expected_size = len(first_keys)

        for item in data[1:]:
            if len(item) != expected_size or frozenset(item.keys()) != first_keys:
                return False

        return True

    @classmethod
    def convert(
        cls,
        content: str,
        source_format: str,
        target_format: str,
        key: Optional[str] = None
    ) -> str:
        """
        Convierte entre formatos (versión ultra optimizada)
        """
        # Cachear la conversión
        cache_key = (content, source_format, target_format, key)
        if cache_key in cls._conversion_cache:
            return cls._conversion_cache[cache_key]

        # Decodificar
        data = cls.decode(content, source_format)

        # Codificar en nuevo formato
        result, _ = cls.encode(data, target_format, key)

        # Cachear resultado
        try:
            cls._conversion_cache[cache_key] = result
            if len(cls._conversion_cache) > 1000:
                oldest_key = next(iter(cls._conversion_cache))
                del cls._conversion_cache[oldest_key]
        except:
            pass

        return result

    @classmethod
    def get_content_type(cls, format_type: str) -> str:
        """
        Retorna el Content-Type HTTP apropiado para el formato
        """
        if format_type == 'toon':
            return 'application/toon'
        else:
            return 'application/json'

    @classmethod
    def analyze_data_ultra_optimized(cls, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Analiza datos y retorna estadísticas de eficiencia (versión ultra optimizada)
        """
        # Cachear análisis
        try:
            data_hash = hash(json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data))
            cache_key = ('analysis', data_hash)
            if cache_key in cls._analysis_cache:
                return cls._analysis_cache[cache_key]
        except TypeError:
            pass

        stats = ToonEncoderUltraOptimized.estimate_token_savings_ultra_optimized(data)

        # Contar arrays uniformes ultra rápido
        uniform_arrays = 0

        def count_uniform_arrays_ultra_optimized(obj):
            nonlocal uniform_arrays
            if isinstance(obj, dict):
                for value in obj.values():
                    if cls._is_large_uniform_array_ultra_optimized(value):
                        uniform_arrays += 1
                    elif isinstance(value, (dict, list)):
                        count_uniform_arrays_ultra_optimized(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        count_uniform_arrays_ultra_optimized(item)

        count_uniform_arrays_ultra_optimized(data)

        result = {
            'format_recommended': 'toon' if stats['recommended'] else 'json',
            'json_size': stats['json_size'],
            'toon_size': stats['toon_size'],
            'savings_percent': stats['savings_percent'],
            'savings_bytes': stats['savings'],
            'has_uniform_arrays': uniform_arrays > 0,
            'uniform_array_count': uniform_arrays,
            'toon_recommended': stats['recommended']
        }

        # Cachear resultado
        try:
            cls._analysis_cache[cache_key] = result
            if len(cls._analysis_cache) > 500:
                oldest_key = next(iter(cls._analysis_cache))
                del cls._analysis_cache[oldest_key]
        except:
            pass

        return result