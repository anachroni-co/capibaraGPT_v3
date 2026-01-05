#!/usr/bin/env python3
"""
TOON Parser Optimized - Convierte formato TOON a estructuras Python con latencia reducida
"""

from typing import Any, Dict, List, Union


class ToonParserOptimized:
    """Parser optimizado para formato TOON con menor latencia"""

    @staticmethod
    def parse(toon_string: str) -> Union[Dict, List]:
        """
        Parsea una cadena TOON a estructuras Python (versión optimizada)
        """
        if not toon_string or not toon_string.strip():
            return {}

        lines = [line.strip() for line in toon_string.strip().split('\n')]
        result = {}

        i = 0
        while i < len(lines):
            line = lines[i]

            # Ignorar líneas vacías o comentarios
            if not line or line.startswith('#'):
                i += 1
                continue

            # Detectar si es una declaración de array (sin regex)
            if '[' in line and ']' in line and '{' in line and '}' in line and ':' in line:
                if ToonParserOptimized._is_array_declaration(line):
                    key, data = ToonParserOptimized._parse_array_declaration_optimized(line, lines, i)
                    result[key] = data
                    # Saltar las líneas de datos
                    i += len(data) + 1
                    continue
            # Detectar objeto simple
            elif '{' in line and '}' in line and ':' in line and not '[' in line:
                if ToonParserOptimized._is_object_declaration(line):
                    key, data = ToonParserOptimized._parse_object_declaration_optimized(line, lines, i)
                    result[key] = data
                    i += 2  # Declaración + datos
                    continue
            # Valor simple
            elif ':' in line and not ('{' in line and '}' in line) and not ('[' in line and ']' in line):
                key, value = line.split(':', 1)
                result[key.strip()] = ToonParserOptimized._parse_value_optimized(value.strip())
                i += 1
                continue
            else:
                i += 1

        return result

    @staticmethod
    def _is_array_declaration(line: str) -> bool:
        """Verifica si una línea es una declaración de array (sin regex)"""
        # Buscar patrón key[n]{...}:
        colon_idx = line.find(':')
        if colon_idx == -1:
            return False

        before_colon = line[:colon_idx].strip()
        # Buscar [n]{...} pattern
        bracket_open = before_colon.find('[')
        bracket_close = before_colon.find(']')
        brace_open = before_colon.find('{')
        brace_close = before_colon.find('}')

        return (bracket_open != -1 and bracket_close != -1 and brace_open != -1 and brace_close != -1
                and bracket_open < bracket_close < brace_open < brace_close)

    @staticmethod
    def _is_object_declaration(line: str) -> bool:
        """Verifica si una línea es una declaración de objeto (sin regex)"""
        colon_idx = line.find(':')
        if colon_idx == -1:
            return False

        before_colon = line[:colon_idx].strip()
        # Buscar {...} pattern sin []
        brace_open = before_colon.find('{')
        brace_close = before_colon.find('}')

        return (brace_open != -1 and brace_close != -1 and brace_open < brace_close
                and '[' not in before_colon)

    @staticmethod
    def _parse_array_declaration_optimized(declaration: str, lines: List[str], start_idx: int) -> tuple:
        """
        Parsea una declaración de array TOON (versión optimizada)
        """
        # Extraer nombre, tamaño y campos sin regex
        colon_idx = declaration.find(':')
        before_colon = declaration[:colon_idx]
        
        # Extraer nombre
        bracket_open_idx = before_colon.find('[')
        key = before_colon[:bracket_open_idx].strip()
        
        # Extraer número entre []
        bracket_close_idx = before_colon.find(']')
        count_str = before_colon[bracket_open_idx + 1:bracket_close_idx]
        count = int(count_str)
        
        # Extraer campos entre {}
        brace_open_idx = before_colon.find('{')
        brace_close_idx = before_colon.find('}')
        fields_str = before_colon[brace_open_idx + 1:brace_close_idx]
        fields = [f.strip() for f in fields_str.split(',')]

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
                    obj[field] = ToonParserOptimized._parse_value_optimized(values[j])
                else:
                    obj[field] = None

            data.append(obj)

        return key, data

    @staticmethod
    def _parse_object_declaration_optimized(declaration: str, lines: List[str], start_idx: int) -> tuple:
        """
        Parsea una declaración de objeto TOON (versión optimizada)
        """
        colon_idx = declaration.find(':')
        before_colon = declaration[:colon_idx]
        
        # Extraer nombre
        brace_open_idx = before_colon.find('{')
        key = before_colon[:brace_open_idx].strip()
        
        # Extraer campos entre {}
        brace_close_idx = before_colon.find('}')
        fields_str = before_colon[brace_open_idx + 1:brace_close_idx]
        fields = [f.strip() for f in fields_str.split(',')]

        # Parsear línea de datos
        if start_idx + 1 < len(lines):
            line = lines[start_idx + 1].strip()
            values = [v.strip() for v in line.split(',')]

            obj = {}
            for i, field in enumerate(fields):
                if i < len(values):
                    obj[field] = ToonParserOptimized._parse_value_optimized(values[i])
                else:
                    obj[field] = None

            return key, obj

        return key, {}

    @staticmethod
    def _parse_value_optimized(value: str) -> Any:
        """
        Convierte un valor string a su tipo apropiado (versión optimizada)
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
                result.append(ToonParserOptimized._parse_value_optimized(elem))
            return result

        # Números (optimizado)
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # String (remover comillas si las tiene)
        if len(value) >= 2:
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                # Remover escapado de comillas internas
                unescaped_value = value[1:-1].replace('\\"', '"').replace("\\'", "'")
                return unescaped_value

        return value

    @classmethod
    def parse_to_dict_optimized(cls, toon_string: str) -> Dict:
        """Alias para parse() que garantiza retornar un dict"""
        result = cls.parse(toon_string)
        if isinstance(result, dict):
            return result
        return {"data": result}

    @classmethod
    def parse_to_list_optimized(cls, toon_string: str) -> List:
        """Parse TOON y retorna lista si es posible"""
        result = cls.parse(toon_string)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and len(result) == 1:
            key = list(result.keys())[0]
            if isinstance(result[key], list):
                return result[key]
        return [result]