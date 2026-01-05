#!/usr/bin/env python3
"""
TOON Parser - Convierte formato TOON a estructuras Python
"""

import re
from typing import Any, Dict, List, Union


class ToonParser:
    """Parser para formato TOON"""

    @staticmethod
    def parse(toon_string: str) -> Union[Dict, List]:
        """
        Parsea una cadena TOON a estructuras Python

        Ejemplos:
            users[2]{id,name,role}:
              1,Alice,admin
              2,Bob,user

            Retorna:
            {
                "users": [
                    {"id": "1", "name": "Alice", "role": "admin"},
                    {"id": "2", "name": "Bob", "role": "user"}
                ]
            }

        Args:
            toon_string: Cadena en formato TOON

        Returns:
            Dict o List con los datos parseados
        """
        if not toon_string or not toon_string.strip():
            return {}

        lines = toon_string.strip().split('\n')
        result = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Ignorar líneas vacías o comentarios
            if not line or line.startswith('#'):
                i += 1
                continue

            # Detectar si es una declaración de array
            if re.match(r'^\w+\[\d+\]\{.+\}:', line):
                key, data = ToonParser._parse_array_declaration(line, lines, i)
                result[key] = data
                # Saltar las líneas de datos
                i += len(data) + 1
            # Detectar objeto simple
            elif re.match(r'^\w+\{.+\}:', line):
                key, data = ToonParser._parse_object_declaration(line, lines, i)
                result[key] = data
                i += 2  # Declaración + datos
            # Valor simple
            elif ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = ToonParser._parse_value(value.strip())
                i += 1
            else:
                i += 1

        return result

    @staticmethod
    def _parse_array_declaration(declaration: str, lines: List[str], start_idx: int) -> tuple:
        """
        Parsea una declaración de array TOON

        Formato: key[count]{field1,field2,...}:
        """
        # Extraer nombre, tamaño y campos
        match = re.match(r'^(\w+)\[(\d+)\]\{([^}]+)\}:', declaration)
        if not match:
            raise ValueError(f"Invalid TOON array declaration: {declaration}")

        key = match.group(1)
        count = int(match.group(2))
        fields = [f.strip() for f in match.group(3).split(',')]

        # Parsear las líneas de datos
        data = []
        for i in range(1, count + 1):
            if start_idx + i >= len(lines):
                break

            line = lines[start_idx + i].strip()
            if not line or line.startswith('#'):
                continue

            values = [v.strip() for v in line.split(',')]

            # Crear objeto con campos y valores
            obj = {}
            for j, field in enumerate(fields):
                if j < len(values):
                    obj[field] = ToonParser._parse_value(values[j])
                else:
                    obj[field] = None

            data.append(obj)

        return key, data

    @staticmethod
    def _parse_object_declaration(declaration: str, lines: List[str], start_idx: int) -> tuple:
        """
        Parsea una declaración de objeto TOON

        Formato: key{field1,field2,...}:
        """
        match = re.match(r'^(\w+)\{([^}]+)\}:', declaration)
        if not match:
            raise ValueError(f"Invalid TOON object declaration: {declaration}")

        key = match.group(1)
        fields = [f.strip() for f in match.group(2).split(',')]

        # Parsear línea de datos
        if start_idx + 1 < len(lines):
            line = lines[start_idx + 1].strip()
            values = [v.strip() for v in line.split(',')]

            obj = {}
            for i, field in enumerate(fields):
                if i < len(values):
                    obj[field] = ToonParser._parse_value(values[i])
                else:
                    obj[field] = None

            return key, obj

        return key, {}

    @staticmethod
    def _parse_value(value: str) -> Any:
        """
        Convierte un valor string a su tipo apropiado

        Soporta: int, float, bool, None, str, list
        """
        value = value.strip()

        # None/null
        if value.lower() in ('null', 'none', ''):
            return None

        # Boolean
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False

        # Verificar si es un array (comienza con [ y termina con ])
        if len(value) >= 2 and value.startswith('[') and value.endswith(']'):
            # Extraer contenido del array
            inner = value[1:-1].strip()
            if not inner:
                return []  # Array vacío
            # Separar elementos por comas
            elements = []
            current = ''
            bracket_count = 0  # Para manejar arrays anidados
            in_quotes = False
            quote_char = None

            for char in inner:
                if char in ['"', "'"] and not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                elif char == '[' and not in_quotes:
                    bracket_count += 1
                elif char == ']' and not in_quotes:
                    bracket_count -= 1
                elif char == ',' and not in_quotes and bracket_count == 0:
                    elements.append(current.strip())
                    current = ''
                    continue
                current += char

            if current.strip():
                elements.append(current.strip())

            # Procesar cada elemento del array
            result = []
            for elem in elements:
                result.append(ToonParser._parse_value(elem))
            return result

        # Números
        try:
            # Intentar int primero
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass

        # String (remover comillas si las tiene)
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            # Remover escapado de comillas internas
            unescaped_value = value[1:-1].replace('\\"', '"').replace("\\'", "'")
            return unescaped_value

        return value

    @classmethod
    def parse_to_dict(cls, toon_string: str) -> Dict:
        """Alias para parse() que garantiza retornar un dict"""
        result = cls.parse(toon_string)
        if isinstance(result, dict):
            return result
        return {"data": result}

    @classmethod
    def parse_to_list(cls, toon_string: str) -> List:
        """Parse TOON y retorna lista si es posible"""
        result = cls.parse(toon_string)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and len(result) == 1:
            # Si hay solo una clave y es una lista, retornarla
            key = list(result.keys())[0]
            if isinstance(result[key], list):
                return result[key]
        return [result]
