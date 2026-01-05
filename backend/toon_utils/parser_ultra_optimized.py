#!/usr/bin/env python3
"""
TOON Parser Ultra Optimized - Convierte formato TOON a estructuras Python con latencia mínima
"""

from typing import Any, Dict, List, Union


class ToonParserUltraOptimized:
    """Parser ultra optimizado para formato TOON con latencia mínima"""

    # Caché para resultados de parsing
    _parsing_cache = {}

    @classmethod
    def parse(cls, toon_string: str) -> Union[Dict, List]:
        """
        Parsea una cadena TOON a estructuras Python (versión ultra optimizada)
        """
        # Usar caché para strings repetidos
        if toon_string in cls._parsing_cache:
            return cls._parsing_cache[toon_string]

        if not toon_string or not toon_string.strip():
            result = {}
        else:
            lines = [line.strip() for line in toon_string.strip().split('\n')]
            result = {}

            i = 0
            while i < len(lines):
                line = lines[i]

                # Ignorar líneas vacías o comentarios
                if not line or line.startswith('#'):
                    i += 1
                    continue

                # Detectar si es una declaración de array
                if ':' in line:
                    colon_idx = line.find(':')
                    before_colon = line[:colon_idx].strip()
                    
                    # Detectar patrones TOON más específicos sin regex
                    if ('[' in before_colon and ']' in before_colon and 
                        '{' in before_colon and '}' in before_colon):
                        # Es una declaración de array
                        key, data = cls._parse_array_declaration_ultra_optimized(line, lines, i)
                        result[key] = data
                        # Saltar las líneas de datos
                        i += len(data) + 1
                        continue
                    elif ('{' in before_colon and '}' in before_colon and 
                          '[' not in before_colon):
                        # Es una declaración de objeto
                        key, data = cls._parse_object_declaration_ultra_optimized(line, lines, i)
                        result[key] = data
                        i += 2  # Declaración + datos
                        continue
                    elif ('[' not in before_colon and '{' not in before_colon and 
                          ']' not in before_colon and '}' not in before_colon):
                        # Valor simple
                        key, value = line.split(':', 1)
                        result[key.strip()] = cls._parse_value_ultra_optimized(value.strip())
                        i += 1
                        continue
                    else:
                        i += 1
                else:
                    i += 1

        # Guardar en caché si no es demasiado grande
        if len(toon_string) < 10000:  # Solo cachear strings razonablemente pequeños
            cls._parsing_cache[toon_string] = result
            # Limitar tamaño de caché
            if len(cls._parsing_cache) > 500:
                oldest_key = next(iter(cls._parsing_cache))
                del cls._parsing_cache[oldest_key]

        return result

    @classmethod
    def _parse_array_declaration_ultra_optimized(cls, declaration: str, lines: List[str], start_idx: int) -> tuple:
        """
        Parsea una declaración de array TOON (versión ultra optimizada)
        """
        # Extraer nombre, tamaño y campos de forma ultra eficiente
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
        fields = tuple(f.strip() for f in fields_str.split(','))  # Tuple para ser más eficiente

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
                    obj[field] = cls._parse_value_ultra_optimized(values[j])
                else:
                    obj[field] = None

            data.append(obj)

        return key, data

    @classmethod
    def _parse_object_declaration_ultra_optimized(cls, declaration: str, lines: List[str], start_idx: int) -> tuple:
        """
        Parsea una declaración de objeto TOON (versión ultra optimizada)
        """
        colon_idx = declaration.find(':')
        before_colon = declaration[:colon_idx]
        
        # Extraer nombre
        brace_open_idx = before_colon.find('{')
        key = before_colon[:brace_open_idx].strip()
        
        # Extraer campos entre {}
        brace_close_idx = before_colon.find('}')
        fields_str = before_colon[brace_open_idx + 1:brace_close_idx]
        fields = tuple(f.strip() for f in fields_str.split(','))  # Tuple para ser más eficiente

        # Parsear línea de datos
        if start_idx + 1 < len(lines):
            line = lines[start_idx + 1].strip()
            values = [v.strip() for v in line.split(',')]

            obj = {}
            for i, field in enumerate(fields):
                if i < len(values):
                    obj[field] = cls._parse_value_ultra_optimized(values[i])
                else:
                    obj[field] = None

            return key, obj

        return key, {}

    @staticmethod
    def _parse_value_ultra_optimized(value: str) -> Any:
        """
        Convierte un valor string a su tipo apropiado (versión ultra optimizada)
        """
        value = value.strip()

        # Comparaciones directas y eficientes
        if value.lower() in ('null', 'none', ''):
            return None

        # Booleanos primero
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
            # Separar elementos por comas, manejando correctamente los que están entre comillas
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
                result.append(ToonParserUltraOptimized._parse_value_ultra_optimized(elem))
            return result

        # Números (intentar int primero, luego float)
        try:
            # Verificar si es un número entero
            dot_index = value.find('.')
            if dot_index == -1:
                # No tiene punto decimal, probar como entero
                return int(value)
            else:
                # Tiene punto decimal, probar como float
                return float(value)
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
    def parse_to_dict_ultra_optimized(cls, toon_string: str) -> Dict:
        """Alias para parse() que garantiza retornar un dict"""
        result = cls.parse(toon_string)
        if isinstance(result, dict):
            return result
        return {"data": result}

    @classmethod
    def parse_to_list_ultra_optimized(cls, toon_string: str) -> List:
        """Parse TOON y retorna lista si es posible"""
        result = cls.parse(toon_string)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and len(result) == 1:
            key = list(result.keys())[0]
            if isinstance(result[key], list):
                return result[key]
        return [result]