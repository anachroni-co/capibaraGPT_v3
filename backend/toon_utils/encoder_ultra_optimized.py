#!/usr/bin/env python3
"""
TOON Encoder Ultra Optimized - Convierte estructuras Python a formato TOON con latencia mínima
"""

from typing import Any, Dict, List, Union
import json
from functools import lru_cache


class ToonEncoderUltraOptimized:
    """Encoder ultra optimizado para formato TOON con latencia mínima"""

    # Caché para estructuras de datos comunes
    _structure_cache = {}
    _encoding_cache = {}

    @classmethod
    def encode(cls, data: Union[Dict, List], key: str = None) -> str:
        """
        Codifica estructuras Python a formato TOON (versión ultra optimizada)
        """
        # Usar caché si los datos son hashables
        try:
            data_hash = hash(json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data))
            cache_key = (data_hash, key)
            if cache_key in cls._encoding_cache:
                return cls._encoding_cache[cache_key]
        except TypeError:
            # Si no se puede hashear, continuar sin caché
            pass

        if isinstance(data, dict):
            result = cls._encode_dict_ultra_optimized(data)
        elif isinstance(data, list):
            if key:
                result = cls._encode_array_ultra_optimized(data, key)
            else:
                result = cls._encode_dict_ultra_optimized({"data": data})
        else:
            result = str(data)

        # Guardar en caché si es posible
        try:
            if cache_key not in cls._encoding_cache:
                cls._encoding_cache[cache_key] = result
                # Limitar el tamaño de caché
                if len(cls._encoding_cache) > 1000:
                    # Quitar elementos antiguos (FIFO)
                    oldest_key = next(iter(cls._encoding_cache))
                    del cls._encoding_cache[oldest_key]
        except:
            pass

        return result

    @staticmethod
    def _encode_dict_ultra_optimized(data: Dict) -> str:
        """Codifica un diccionario a TOON (versión ultra optimizada)"""
        # Usar lista de strings para construir eficientemente
        parts = []
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                parts.append(ToonEncoderUltraOptimized._encode_array_ultra_optimized(value, key))
            elif isinstance(value, dict):
                parts.append(ToonEncoderUltraOptimized._encode_object_ultra_optimized(value, key))
            elif isinstance(value, list):
                # Array de valores simples
                encoded_values = ','.join(str(v) for v in value)
                parts.append(f"{key}: [{encoded_values}]")
            else:
                # Valor simple
                parts.append(f"{key}: {ToonEncoderUltraOptimized._encode_value_ultra_optimized(value)}")

        return '\n'.join(parts)

    @staticmethod
    def _encode_array_ultra_optimized(data: List[Dict], key: str) -> str:
        """
        Codifica un array de objetos a TOON (versión ultra optimizada)
        """
        if not data:
            return f"{key}[0]{{}}: "

        if not isinstance(data[0], dict):
            # Array de valores simples
            values = ','.join(str(v) for v in data)
            return f"{key}: [{values}]"

        fields = tuple(data[0].keys())  # Usar tuple para ser hashable
        count = len(data)

        # Pre-calculate field string
        field_str = ','.join(fields)
        lines = [f"{key}[{count}]{{{field_str}}}:"]

        # Cache the field values processing
        for item in data:
            values = [ToonEncoderUltraOptimized._encode_value_ultra_optimized(item.get(field)) for field in fields]
            lines.append(f"  {','.join(values)}")

        return '\n'.join(lines)

    @staticmethod
    def _encode_object_ultra_optimized(data: Dict, key: str) -> str:
        """
        Codifica un objeto a TOON (versión ultra optimizada)
        """
        if not data:
            return f"{key}{{}}: "

        fields = tuple(data.keys())  # Usar tuple para ser hashable
        field_str = ','.join(fields)

        values = [ToonEncoderUltraOptimized._encode_value_ultra_optimized(data.get(field)) for field in fields]

        return f"{key}{{{field_str}}}:\n  {','.join(values)}"

    @staticmethod
    def _encode_value_ultra_optimized(value: Any) -> str:
        """
        Codifica un valor individual (versión ultra optimizada)
        """
        if value is None:
            return "null"

        if value is True:
            return "true"
        if value is False:
            return "false"

        if isinstance(value, (int, float)):
            return str(value)

        # String handling ultra optimizado
        value_str = str(value)

        # Check for special characters efficiently
        if ',' in value_str or '\n' in value_str or value_str != value_str.strip():
            value_str = value_str.replace('"', '\\"')
            return f'"{value_str}"'

        return value_str

    @classmethod
    def encode_compact_ultra_optimized(cls, data: Union[Dict, List], key: str = None) -> str:
        """
        Versión compacta sin indentación (ultra optimizada)
        """
        result = cls.encode(data, key)
        # Optimización: unir líneas de forma más eficiente
        lines = []
        for line in result.split('\n'):
            lines.append(line.lstrip())
        return '\n'.join(lines)

    @classmethod
    def estimate_token_savings_ultra_optimized(cls, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Estima el ahorro de tokens usando TOON vs JSON (versión ultra optimizada)
        """
        # Usar caché para cálculos repetidos
        try:
            data_hash = hash(json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data))
            cache_key = ('savings', data_hash)
            if cache_key in cls._encoding_cache:
                return cls._encoding_cache[cache_key]
        except TypeError:
            pass

        # Tamaño JSON
        json_str = json.dumps(data, separators=(',', ':'))
        json_size = len(json_str)

        # Tamaño TOON
        toon_str = cls.encode(data)
        toon_size = len(toon_str)

        # Calcular ahorro
        savings = json_size - toon_size
        savings_percent = (savings / json_size * 100) if json_size > 0 else 0

        # Recomendar TOON si ahorra > 20%
        recommended = savings_percent > 20

        result = {
            'json_size': json_size,
            'toon_size': toon_size,
            'savings': savings,
            'savings_percent': round(savings_percent, 2),
            'recommended': recommended
        }

        # Cachear resultado
        try:
            cls._encoding_cache[cache_key] = result
            if len(cls._encoding_cache) > 100:
                oldest_key = next(iter(cls._encoding_cache))
                del cls._encoding_cache[oldest_key]
        except:
            pass

        return result