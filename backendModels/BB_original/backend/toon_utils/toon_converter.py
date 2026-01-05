"""
Utilidades para TOON en el proyecto Capibara6
Conversión entre JSON y TOON para optimizar tokens en comunicaciones con LLMs
"""

import json
import re
from typing import Any, Dict, List, Union

def toon_encode(obj: Any) -> str:
    """
    Codifica un objeto Python a TOON (Token-Oriented Object Notation)
    """
    if isinstance(obj, dict):
        return _encode_dict(obj)
    elif isinstance(obj, list):
        return _encode_list(obj)
    else:
        # Para primitivos, devolver como string simple
        return str(obj)

def toon_decode(toon_str: str) -> Any:
    """
    Decodifica una string TOON a objeto Python (funcionalidad básica)
    """
    # Para esta implementación básica, simplemente parsear casos simples
    lines = toon_str.strip().split('\n')
    
    # Detectar formato de array tabular: nombre[longitud]{campos}:
    tabular_pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\[([0-9]+)\]\{([^}]+)\}:$'
    
    for i, line in enumerate(lines):
        match = re.match(tabular_pattern, line.strip())
        if match:
            array_name, count_str, fields_str = match.groups()
            fields = [f.strip() for f in fields_str.split(',')]
            
            # Procesar filas de datos
            data_rows = []
            for j in range(i + 1, len(lines)):
                data_line = lines[j].strip()
                if not data_line or data_line.startswith('#'):
                    continue
                if data_line and not data_line.endswith(':') and not re.match(tabular_pattern, data_line):
                    values = [val.strip() for val in data_line.split(',')]
                    row_dict = {}
                    for k, field in enumerate(fields):
                        if k < len(values):
                            # Intentar convertir tipos básicos
                            val = values[k].strip()
                            if val.lower() in ['true', 'false']:
                                row_dict[field] = val.lower() == 'true'
                            elif val.isdigit():
                                row_dict[field] = int(val)
                            else:
                                # Intentar parsear como número flotante
                                try:
                                    row_dict[field] = float(val)
                                except ValueError:
                                    row_dict[field] = val
                    data_rows.append(row_dict)
            
            return {array_name: data_rows}
    
    # Si no es tabular, devolver el string sin procesar
    # En una implementación completa, se haría el parsing completo
    return json.loads(toon_str)  # Fallback para este ejemplo

def _encode_dict(d: Dict) -> str:
    """
    Codifica un diccionario a TOON
    """
    lines = []
    for key, value in d.items():
        if isinstance(value, list) and _is_uniform_list_of_dicts(value):
            # Codificar como array tabular
            lines.append(_encode_tabular_array(key, value))
        elif isinstance(value, dict):
            # Codificar como objeto indentado
            lines.append(f"{key}:")
            nested_lines = _encode_dict(value).split('\n')
            for nested_line in nested_lines:
                if nested_line.strip():
                    lines.append(f"  {nested_line}")
        elif isinstance(value, list):
            # Codificar como array simple
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            # Valor primitivo
            lines.append(f"{key}: {value}")
    return '\n'.join(lines)

def _encode_tabular_array(name: str, arr: List[Dict]) -> str:
    """
    Codifica un array de objetos con la misma estructura como array tabular TOON
    """
    if not arr:
        return f"{name}[0]{{}}:"
    
    # Obtener todos los campos posibles
    all_fields = set()
    for item in arr:
        if isinstance(item, dict):
            all_fields.update(item.keys())
    
    fields = sorted(list(all_fields))
    lines = [f"{name}[{len(arr)}]{{{','.join(fields)}}}:"]
    
    for item in arr:
        row_values = []
        for field in fields:
            value = item.get(field, '')
            row_values.append(str(value))
        lines.append(f"  {','.join(row_values)}")
    
    return '\n'.join(lines)

def _is_uniform_list_of_dicts(lst: List) -> bool:
    """
    Verifica si una lista es de objetos con la misma estructura
    """
    if not lst:
        return False
    
    if not all(isinstance(item, dict) for item in lst):
        return False
    
    # Verificar que todos tengan las mismas claves
    first_keys = set(lst[0].keys())
    return all(set(item.keys()) == first_keys for item in lst)