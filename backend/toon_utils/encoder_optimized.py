#!/usr/bin/env python3
"""
TOON Encoder Optimized - Convierte estructuras Python a formato TOON con latencia reducida
"""

from typing import Any, Dict, List, Union
import json


class ToonEncoderOptimized:
    """Encoder optimizado para formato TOON con menor latencia"""

    @staticmethod
    def encode(data: Union[Dict, List], key: str = None) -> str:
        """
        Codifica estructuras Python a formato TOON (versión optimizada)
        """
        if isinstance(data, dict):
            return ToonEncoderOptimized._encode_dict_optimized(data)
        elif isinstance(data, list):
            if key:
                return ToonEncoderOptimized._encode_array_optimized(data, key)
            else:
                return ToonEncoderOptimized._encode_dict_optimized({"data": data})
        else:
            return str(data)

    @staticmethod
    def _encode_dict_optimized(data: Dict) -> str:
        """Codifica un diccionario a TOON (versión optimizada)"""
        lines = []
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(ToonEncoderOptimized._encode_array_optimized(value, key))
            elif isinstance(value, dict):
                lines.append(ToonEncoderOptimized._encode_object_optimized(value, key))
            elif isinstance(value, list):
                # Array de valores simples
                encoded_values = ','.join(str(v) for v in value)
                lines.append(f"{key}: [{encoded_values}]")
            else:
                # Valor simple
                lines.append(f"{key}: {ToonEncoderOptimized._encode_value_optimized(value)}")

        return '\n'.join(lines)

    @staticmethod
    def _encode_array_optimized(data: List[Dict], key: str) -> str:
        """
        Codifica un array de objetos a TOON (versión optimizada)
        """
        if not data:
            return f"{key}[0]{{}}: "

        if not isinstance(data[0], dict):
            # Array de valores simples
            values = ','.join(str(v) for v in data)
            return f"{key}: [{values}]"

        fields = list(data[0].keys())
        count = len(data)

        # Pre-calculate field string
        field_str = ','.join(fields)
        lines = [f"{key}[{count}]{{{field_str}}}:"]

        # Cache the field values processing
        for item in data:
            values = [ToonEncoderOptimized._encode_value_optimized(item.get(field)) for field in fields]
            lines.append(f"  {','.join(values)}")

        return '\n'.join(lines)

    @staticmethod
    def _encode_object_optimized(data: Dict, key: str) -> str:
        """
        Codifica un objeto a TOON (versión optimizada)
        """
        if not data:
            return f"{key}{{}}: "

        fields = list(data.keys())
        field_str = ','.join(fields)

        values = [ToonEncoderOptimized._encode_value_optimized(data.get(field)) for field in fields]

        return f"{key}{{{field_str}}}:\n  {','.join(values)}"

    @staticmethod
    def _encode_value_optimized(value: Any) -> str:
        """
        Codifica un valor individual (versión optimizada)
        """
        if value is None:
            return "null"

        if value is True:
            return "true"
        if value is False:
            return "false"

        if isinstance(value, (int, float)):
            return str(value)

        # String handling optimized
        value_str = str(value)

        # Check for special characters efficiently
        if ',' in value_str or '\n' in value_str or value_str != value_str.strip():
            value_str = value_str.replace('"', '\\"')
            return f'"{value_str}"'

        return value_str

    @classmethod
    def encode_compact_optimized(cls, data: Union[Dict, List], key: str = None) -> str:
        """
        Versión compacta sin indentación (optimizada)
        """
        result = cls.encode(data, key)
        lines = [line.lstrip() for line in result.split('\n')]
        return '\n'.join(lines)

    @classmethod
    def estimate_token_savings_optimized(cls, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Estima el ahorro de tokens usando TOON vs JSON (versión optimizada)
        """
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

        return {
            'json_size': json_size,
            'toon_size': toon_size,
            'savings': savings,
            'savings_percent': round(savings_percent, 2),
            'recommended': recommended
        }