#!/usr/bin/env python3
"""
Format Manager - Gestiona conversiones automáticas entre JSON y TOON
"""

import json
from typing import Any, Dict, List, Tuple, Union, Optional

from .parser import ToonParser
from .encoder import ToonEncoder


class FormatManager:
    """
    Gestor de formatos que decide automáticamente cuándo usar TOON vs JSON
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
        Codifica datos en el formato óptimo

        Args:
            data: Datos a codificar
            preferred_format: 'auto', 'json', 'toon', o 'compact_toon'
            key: Clave opcional para array root

        Returns:
            Tuple de (contenido_codificado, tipo_formato)

        Ejemplos:
            >>> data = {"users": [{"id": 1, "name": "Alice"}]}
            >>> content, fmt = FormatManager.encode(data, 'auto')
            >>> print(fmt)  # 'json' o 'toon' dependiendo de eficiencia
        """
        # Formato explícito
        if preferred_format == 'json':
            return json.dumps(data, separators=(',', ':')), 'json'

        if preferred_format == 'toon':
            return ToonEncoder.encode(data, key), 'toon'

        if preferred_format == 'compact_toon':
            return ToonEncoder.encode_compact(data, key), 'toon'

        # Auto: Decidir basado en eficiencia
        if preferred_format == 'auto':
            should_use_toon = cls.should_use_toon(data)

            if should_use_toon:
                return ToonEncoder.encode(data, key), 'toon'
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
        Decodifica contenido del formato especificado

        Args:
            content: Contenido a decodificar
            format_type: 'auto', 'json', o 'toon'

        Returns:
            Dict o List con datos decodificados

        Ejemplos:
            >>> toon_str = "users[1]{id,name}:\\n  1,Alice"
            >>> data = FormatManager.decode(toon_str, 'toon')
            >>> print(data)
            {'users': [{'id': '1', 'name': 'Alice'}]}
        """
        if not content or not content.strip():
            return {}

        # Detección automática
        if format_type == 'auto':
            format_type = cls.detect_format(content)

        if format_type == 'toon':
            return ToonParser.parse(content)
        else:
            # JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Si falla JSON, intentar TOON como fallback
                try:
                    return ToonParser.parse(content)
                except Exception:
                    raise ValueError(f"No se pudo decodificar el contenido en ningún formato válido")

    @classmethod
    def detect_format(cls, content: str) -> str:
        """
        Detecta automáticamente el formato del contenido

        Args:
            content: String con el contenido

        Returns:
            'json' o 'toon'
        """
        content = content.strip()

        # JSON típicamente empieza con { o [
        if content.startswith('{') or content.startswith('['):
            return 'json'

        # TOON tiene patrones característicos: key[n]{...}: o key{...}:
        import re
        if re.search(r'\w+\[\d+\]\{[^}]+\}:', content) or \
           re.search(r'\w+\{[^}]+\}:', content):
            return 'toon'

        # Default: JSON
        return 'json'

    @classmethod
    def should_use_toon(cls, data: Any) -> bool:
        """
        Determina si TOON es más eficiente que JSON para los datos dados

        Heurística:
        - Arrays grandes (>= MIN_ARRAY_SIZE) de objetos con la misma estructura
        - Ahorro estimado >= MIN_SAVINGS_PERCENT

        Args:
            data: Datos a evaluar

        Returns:
            True si se recomienda usar TOON
        """
        # Verificar si hay arrays grandes de objetos uniformes
        if isinstance(data, dict):
            for key, value in data.items():
                if cls._is_large_uniform_array(value):
                    # Verificar ahorro real
                    stats = ToonEncoder.estimate_token_savings({key: value})
                    if stats['savings_percent'] >= cls.MIN_SAVINGS_PERCENT:
                        return True

        elif cls._is_large_uniform_array(data):
            stats = ToonEncoder.estimate_token_savings({"data": data})
            if stats['savings_percent'] >= cls.MIN_SAVINGS_PERCENT:
                return True

        return False

    @classmethod
    def _is_large_uniform_array(cls, data: Any) -> bool:
        """
        Verifica si los datos son un array grande de objetos con la misma estructura

        Args:
            data: Datos a verificar

        Returns:
            True si es un array uniforme >= MIN_ARRAY_SIZE
        """
        if not isinstance(data, list):
            return False

        if len(data) < cls.MIN_ARRAY_SIZE_FOR_TOON:
            return False

        # Verificar que todos sean dicts
        if not all(isinstance(item, dict) for item in data):
            return False

        # Verificar que tengan las mismas claves
        if not data:
            return False

        first_keys = set(data[0].keys())

        for item in data[1:]:
            if set(item.keys()) != first_keys:
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
        Convierte entre formatos

        Args:
            content: Contenido a convertir
            source_format: 'json' o 'toon'
            target_format: 'json' o 'toon'
            key: Clave opcional para array root

        Returns:
            Contenido convertido
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

        Args:
            format_type: 'json' o 'toon'

        Returns:
            MIME type string
        """
        if format_type == 'toon':
            return 'application/toon'
        else:
            return 'application/json'

    @classmethod
    def analyze_data(cls, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Analiza datos y retorna estadísticas de eficiencia

        Args:
            data: Datos a analizar

        Returns:
            Dict con estadísticas:
                - format_recommended: 'json' o 'toon'
                - json_size: Tamaño estimado en JSON
                - toon_size: Tamaño estimado en TOON
                - savings_percent: Porcentaje de ahorro
                - has_uniform_arrays: bool
                - array_count: Número de arrays uniformes encontrados
        """
        stats = ToonEncoder.estimate_token_savings(data)

        # Contar arrays uniformes
        uniform_arrays = 0

        def count_uniform_arrays(obj):
            nonlocal uniform_arrays
            if isinstance(obj, dict):
                for value in obj.values():
                    if cls._is_large_uniform_array(value):
                        uniform_arrays += 1
                    elif isinstance(value, (dict, list)):
                        count_uniform_arrays(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        count_uniform_arrays(item)

        count_uniform_arrays(data)

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
