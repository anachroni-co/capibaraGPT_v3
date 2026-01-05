#!/usr/bin/env python3
"""
Format Manager Optimized - Gestiona conversiones automáticas entre JSON y TOON con menor latencia
"""

import json
from typing import Any, Dict, List, Tuple, Union, Optional

from .parser_optimized import ToonParserOptimized
from .encoder_optimized import ToonEncoderOptimized


class FormatManagerOptimized:
    """
    Gestor de formatos optimizado que decide automáticamente cuándo usar TOON vs JSON
    con menor latencia que la implementación anterior
    """

    # Constantes
    MIN_ARRAY_SIZE_FOR_TOON = 5  # Mínimo elementos para considerar TOON
    MIN_SAVINGS_PERCENT = 20      # Mínimo % de ahorro para recomendar TOON

    @classmethod
    def encode(
        cls,
        data: Union[Dict, List],
        preferred_format: str = 'auto',
        key: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Codifica datos en el formato óptimo (versión optimizada)
        """
        # Formato explícito
        if preferred_format == 'json':
            return json.dumps(data, separators=(',', ':')), 'json'

        if preferred_format == 'toon':
            return ToonEncoderOptimized.encode(data, key), 'toon'

        if preferred_format == 'compact_toon':
            return ToonEncoderOptimized.encode_compact_optimized(data, key), 'toon'

        # Auto: Decidir basado en eficiencia (más rápido)
        if preferred_format == 'auto':
            should_use_toon = cls.should_use_toon_optimized(data)

            if should_use_toon:
                return ToonEncoderOptimized.encode(data, key), 'toon'
            else:
                return json.dumps(data, separators=(',', ':')), 'json'

        # Default: JSON
        return json.dumps(data, separators=(',', ':')), 'json'

    @classmethod
    def decode(
        cls,
        content: str,
        format_type: str = 'auto'
    ) -> Union[Dict, List]:
        """
        Decodifica contenido del formato especificado (versión optimizada)
        """
        if not content or not content.strip():
            return {}

        # Detección automática
        if format_type == 'auto':
            format_type = cls.detect_format_optimized(content)

        if format_type == 'toon':
            return ToonParserOptimized.parse(content)
        else:
            # JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Si falla JSON, intentar TOON como fallback
                try:
                    return ToonParserOptimized.parse(content)
                except Exception:
                    raise ValueError(f"No se pudo decodificar el contenido en ningún formato válido")

    @classmethod
    def detect_format_optimized(cls, content: str) -> str:
        """
        Detecta automáticamente el formato del contenido (versión optimizada)
        """
        content = content.strip()

        # JSON típicamente empieza con { o [
        if content.startswith('{') or content.startswith('['):
            return 'json'

        # TOON tiene patrones característicos: key[n]{...}: o key{...}:
        # Usar detección sin regex para mayor velocidad
        if ':' in content:
            # Buscar patrones TOON sin regex
            colon_idx = content.find(':')
            if colon_idx != -1:
                before_colon = content[:colon_idx]
                if ('[' in before_colon and ']' in before_colon and 
                    '{' in before_colon and '}' in before_colon):
                    return 'toon'

        # Default: JSON
        return 'json'

    @classmethod
    def should_use_toon_optimized(cls, data: Any) -> bool:
        """
        Determina si TOON es más eficiente que JSON para los datos dados (versión optimizada)
        """
        # Verificar si hay arrays grandes de objetos uniformes
        if isinstance(data, dict):
            for key, value in data.items():
                if cls._is_large_uniform_array_optimized(value):
                    # Verificar ahorro real - usar una versión más rápida de estimación
                    return ToonEncoderOptimized.estimate_token_savings_optimized({key: value})['recommended']

        elif cls._is_large_uniform_array_optimized(data):
            return ToonEncoderOptimized.estimate_token_savings_optimized({"data": data})['recommended']

        return False

    @classmethod
    def _is_large_uniform_array_optimized(cls, data: Any) -> bool:
        """
        Verifica si los datos son un array grande de objetos con la misma estructura (versión optimizada)
        """
        if not isinstance(data, list):
            return False

        if len(data) < cls.MIN_ARRAY_SIZE_FOR_TOON:
            return False

        # Verificar que todos sean dicts
        if not data or not all(isinstance(item, dict) for item in data):
            return False

        # Verificar que tengan las mismas claves - optimizado
        first_keys = set(data[0].keys())
        expected_size = len(first_keys)

        for item in data[1:]:
            if len(item) != expected_size or set(item.keys()) != first_keys:
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
        Convierte entre formatos (versión optimizada)
        """
        # Decodificar
        data = cls.decode(content, source_format)

        # Codificar en nuevo formato
        result, _ = cls.encode(data, target_format, key)
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
    def analyze_data_optimized(cls, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Analiza datos y retorna estadísticas de eficiencia (versión optimizada)
        """
        stats = ToonEncoderOptimized.estimate_token_savings_optimized(data)

        # Contar arrays uniformes de forma optimizada
        uniform_arrays = 0

        def count_uniform_arrays_optimized(obj):
            nonlocal uniform_arrays
            if isinstance(obj, dict):
                for value in obj.values():
                    if cls._is_large_uniform_array_optimized(value):
                        uniform_arrays += 1
                    elif isinstance(value, (dict, list)):
                        count_uniform_arrays_optimized(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        count_uniform_arrays_optimized(item)

        count_uniform_arrays_optimized(data)

        return {
            'format_recommended': 'toon' if stats['recommended'] else 'json',
            'json_size': stats['json_size'],
            'toon_size': stats['toon_size'],
            'savings_percent': stats['savings_percent'],
            'savings_bytes': stats['savings'],
            'has_uniform_arrays': uniform_arrays > 0,
            'uniform_array_count': uniform_arrays,
            'toon_recommended': stats['recommended']
        }