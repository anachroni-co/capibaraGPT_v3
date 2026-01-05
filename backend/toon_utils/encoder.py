#!/usr/bin/env python3
"""
TOON Encoder - Convierte estructuras Python a formato TOON
"""

from typing import Any, Dict, List, Union


class ToonEncoder:
    """Encoder para formato TOON"""

    @staticmethod
    def encode(data: Union[Dict, List], key: str = None) -> str:
        """
        Codifica estructuras Python a formato TOON

        Args:
            data: Dict o List a codificar
            key: Nombre opcional para la clave raíz

        Returns:
            String en formato TOON

        Ejemplos:
            Input: {"users": [{"id": 1, "name": "Alice", "role": "admin"}]}
            Output:
                users[1]{id,name,role}:
                  1,Alice,admin
        """
        if isinstance(data, dict):
            return ToonEncoder._encode_dict(data)
        elif isinstance(data, list):
            if key:
                return ToonEncoder._encode_array(data, key)
            else:
                # Si no hay clave, envolver en un dict
                return ToonEncoder._encode_dict({"data": data})
        else:
            # Valor simple
            return str(data)

    @staticmethod
    def _encode_dict(data: Dict) -> str:
        """Codifica un diccionario a TOON"""
        lines = []

        for key, value in data.items():
            if isinstance(value, list):
                # Array de objetos
                if value and isinstance(value[0], dict):
                    lines.append(ToonEncoder._encode_array(value, key))
                else:
                    # Array de valores simples - usar sintaxis simple
                    encoded_values = ','.join(str(v) for v in value)
                    lines.append(f"{key}: [{encoded_values}]")

            elif isinstance(value, dict):
                # Objeto único
                lines.append(ToonEncoder._encode_object(value, key))

            else:
                # Valor simple
                lines.append(f"{key}: {ToonEncoder._encode_value(value)}")

        return '\n'.join(lines)

    @staticmethod
    def _encode_array(data: List[Dict], key: str) -> str:
        """
        Codifica un array de objetos a TOON

        Formato: key[count]{field1,field2,...}:
                   value1,value2,...
        """
        if not data:
            return f"{key}[0]{{}}: "

        # Obtener campos del primer objeto
        if not isinstance(data[0], dict):
            # Array de valores simples
            values = ','.join(str(v) for v in data)
            return f"{key}: [{values}]"

        fields = list(data[0].keys())
        count = len(data)

        # Declaración
        field_str = ','.join(fields)
        lines = [f"{key}[{count}]{{{field_str}}}:"]

        # Datos
        for item in data:
            values = []
            for field in fields:
                value = item.get(field)
                values.append(ToonEncoder._encode_value(value))

            lines.append(f"  {','.join(values)}")

        return '\n'.join(lines)

    @staticmethod
    def _encode_object(data: Dict, key: str) -> str:
        """
        Codifica un objeto a TOON

        Formato: key{field1,field2,...}:
                   value1,value2,...
        """
        if not data:
            return f"{key}{{}}: "

        fields = list(data.keys())
        field_str = ','.join(fields)

        values = []
        for field in fields:
            value = data.get(field)
            values.append(ToonEncoder._encode_value(value))

        return f"{key}{{{field_str}}}:\n  {','.join(values)}"

    @staticmethod
    def _encode_value(value: Any) -> str:
        """
        Codifica un valor individual

        - None -> "null"
        - bool -> "true"/"false"
        - números -> string del número
        - strings con comas -> entre comillas
        """
        if value is None:
            return "null"

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float)):
            return str(value)

        # String
        value_str = str(value)

        # Si contiene comas, espacios o caracteres especiales, usar comillas
        if ',' in value_str or '\n' in value_str or value_str != value_str.strip():
            # Escapar comillas dobles
            value_str = value_str.replace('"', '\\"')
            return f'"{value_str}"'

        return value_str

    @classmethod
    def encode_compact(cls, data: Union[Dict, List], key: str = None) -> str:
        """
        Versión compacta sin indentación

        Útil para minimizar aún más el uso de tokens
        """
        result = cls.encode(data, key)
        # Remover indentación
        lines = [line.lstrip() for line in result.split('\n')]
        return '\n'.join(lines)

    @classmethod
    def estimate_token_savings(cls, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Estima el ahorro de tokens usando TOON vs JSON

        Returns:
            Dict con:
                - json_size: Tamaño aproximado en JSON
                - toon_size: Tamaño en TOON
                - savings_percent: Porcentaje de ahorro
                - recommended: bool, si se recomienda usar TOON
        """
        import json

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
