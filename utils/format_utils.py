"""
Utilidades de formateo for CapibaraGPT.
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta


def format_bytes(bytes_value: Union[int, float]) -> str:
    """
    Formatea un valor en bytes a una representación legible.
    
    Args:
        bytes_value: Valor en bytes
        
    Returns:
        Cadena formateada (ej: "1.5 GB")
    """
    if not isinstance(bytes_value, (int, float)) or bytes_value < 0:
        raise ValueError("bytes_value debe ser un número no negativo")
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(bytes_value)
    
    for unit in units:
        if size < 1024.0:
            if unit == 'B':
                return f"{int(size)} {unit}"
            else:
                return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} {units[-1]}"


def format_duration(seconds: Union[int, float]) -> str:
    """
    Formatea una duración en segundos a formato legible.
    
    Args:
        seconds: Duración en segundos
        
    Returns:
        Cadena formateada (ej: "2h 30m 15s")
    """
    if not isinstance(seconds, (int, float)) or seconds < 0:
        raise ValueError("seconds debe ser un número no negativo")
    
    if seconds == 0:
        return "0s"
    
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # show segundos if not hay otras unidades
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """
    Formatea un número with separadores de miles.
    
    Args:
        number: Número a formatear
        precision: precision decimal for números flotantes
        
    Returns:
        Número formateado
    """
    if not isinstance(number, (int, float)):
        raise ValueError("number debe ser un número")
    
    if isinstance(number, int):
        return f"{number:,}"
    else:
        return f"{number:,.{precision}f}"


def format_percentage(value: Union[int, float], total: Union[int, float], precision: int = 1) -> str:
    """
    Formatea un value how porcentaje.
    
    Args:
        value: value current
        total: value total
        precision: precision decimal
        
    Returns:
        Porcentaje formateado
    """
    if not isinstance(value, (int, float)) or not isinstance(total, (int, float)):
        raise ValueError("value y total deben ser números")
    
    if total == 0:
        return "0.0%"
    
    percentage = (value / total) * 100
    return f"{percentage:.{precision}f}%"


def truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    Trunca texto a una length máxima.
    
    Args:
        text: Texto a truncar
        max_length: length máxima
        ellipsis: Texto a add al end
        
    Returns:
        Texto truncado
    """
    if not isinstance(text, str):
        raise ValueError("text debe ser una cadena")
    
    if max_length < 0:
        raise ValueError("max_length debe ser no negativo")
    
    if len(text) <= max_length:
        return text
    
    if max_length < len(ellipsis):
        return text[:max_length]
    
    return text[:max_length - len(ellipsis)] + ellipsis


def format_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
    """
    Formatea data en una table ASCII.
    
    Args:
        data: list de diccionarios with los data
        headers: list de encabezados (optional)
        
    Returns:
        table formateada how string
    """
    if not data:
        return ""
    
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("data debe ser una lista de diccionarios")
    
    # obtain todas las claves if not se proporcionan headers
    if headers is None:
        headers = list(data[0].keys())
    
    # calculate anchos de column
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))
        for row in data:
            value = str(row.get(header, ""))
            col_widths[header] = max(col_widths[header], len(value))
    
    # build table
    lines = []
    
    # Encabezados
    header_line = " | ".join(header.ljust(col_widths[header]) for header in headers)
    lines.append(header_line)
    
    # Separador
    separator = "-+-".join("-" * col_widths[header] for header in headers)
    lines.append(separator)
    
    # data
    for row in data:
        data_line = " | ".join(str(row.get(header, "")).ljust(col_widths[header]) for header in headers)
        lines.append(data_line)
    
    return "\n".join(lines)


def format_json_pretty(data: Any, indent: int = 2) -> str:
    """
    Formatea data how JSON with indentación.
    
    Args:
        data: data a formatear
        indent: Espacios de indentación
        
    Returns:
        JSON formateado
    """
    import json
    
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True)
    except TypeError as e:
        raise ValueError(f"Los datos no son serializables a JSON: {e}")


def format_list_items(items: List[Any], bullet: str = "•", indent: int = 2) -> str:
    """
    Formatea una list how elementos with viñetas.
    
    Args:
        items: list de elementos
        bullet: Carácter de viñeta
        indent: Espacios de indentación
        
    Returns:
        list formateada
    """
    if not isinstance(items, list):
        raise ValueError("items debe ser una lista")
    
    if not items:
        return ""
    
    indent_str = " " * indent
    lines = []
    
    for item in items:
        lines.append(f"{indent_str}{bullet} {str(item)}")
    
    return "\n".join(lines)


def format_key_value_pairs(data: Dict[str, Any], separator: str = ": ", indent: int = 0) -> str:
    """
    Formatea un diccionario how pares key-value.
    
    Args:
        data: Diccionario a formatear
        separator: Separador between key and value
        indent: Espacios de indentación
        
    Returns:
        Pares key-value formateados
    """
    if not isinstance(data, dict):
        raise ValueError("data debe ser un diccionario")
    
    if not data:
        return ""
    
    indent_str = " " * indent
    lines = []
    
    for key, value in data.items():
        lines.append(f"{indent_str}{key}{separator}{value}")
    
    return "\n".join(lines)


def format_progress_bar(current: int, total: int, width: int = 50, fill: str = "█", empty: str = "░") -> str:
    """
    Formatea una barra de progreso.
    
    Args:
        current: value current
        total: value total
        width: width de la barra
        fill: Carácter de relleno
        empty: Carácter empty
        
    Returns:
        Barra de progreso formateada
    """
    if not isinstance(current, int) or not isinstance(total, int):
        raise ValueError("current y total deben ser enteros")
    
    if total <= 0:
        raise ValueError("total debe ser mayor que 0")
    
    if current < 0:
        current = 0
    elif current > total:
        current = total
    
    percentage = current / total
    filled_width = int(width * percentage)
    empty_width = width - filled_width
    
    bar = fill * filled_width + empty * empty_width
    percent_str = f"{percentage * 100:.1f}%"
    
    return f"[{bar}] {percent_str} ({current}/{total})"


def format_timestamp(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Formatea un timestamp.
    
    Args:
        timestamp: Timestamp a formatear (None for use el current)
        format_str: format de output
        
    Returns:
        Timestamp formateado
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    if not isinstance(timestamp, datetime):
        raise ValueError("timestamp debe ser un objeto datetime")
    
    return timestamp.strftime(format_str)


def format_relative_time(timestamp: datetime) -> str:
    """
    Formatea un timestamp como tiempo relativo.
    
    Args:
        timestamp: Timestamp a formatear
        
    Returns:
        Tiempo relativo (ej: "hace 2 horas")
    """
    if not isinstance(timestamp, datetime):
        raise ValueError("timestamp debe ser un objeto datetime")
    
    now = datetime.now()
    diff = now - timestamp
    
    if diff.total_seconds() < 0:
        return "en el futuro"
    
    seconds = int(diff.total_seconds())
    
    if seconds < 60:
        return f"hace {seconds} segundo{'s' if seconds != 1 else ''}"
    
    minutes = seconds // 60
    if minutes < 60:
        return f"hace {minutes} minuto{'s' if minutes != 1 else ''}"
    
    hours = minutes // 60
    if hours < 24:
        return f"hace {hours} hora{'s' if hours != 1 else ''}"
    
    days = hours // 24
    if days < 30:
        return f"hace {days} día{'s' if days != 1 else ''}"
    
    months = days // 30
    if months < 12:
        return f"hace {months} mes{'es' if months != 1 else ''}"
    
    years = months // 12
    return f"hace {years} año{'s' if years != 1 else ''}"


def format_file_size_comparison(size1: int, size2: int) -> str:
    """
    Compara dos tamaños de file and formatea la diferencia.
    
    Args:
        size1: first size
        size2: Segundo size
        
    Returns:
        Comparación formateada
    """
    if not isinstance(size1, int) or not isinstance(size2, int):
        raise ValueError("Los tamaños deben ser enteros")
    
    if size1 < 0 or size2 < 0:
        raise ValueError("Los tamaños deben ser no negativos")
    
    diff = size2 - size1
    
    if diff == 0:
        return f"{format_bytes(size1)} (sin cambio)"
    elif diff > 0:
        return f"{format_bytes(size1)} → {format_bytes(size2)} (+{format_bytes(diff)})"
    else:
        return f"{format_bytes(size1)} → {format_bytes(size2)} (-{format_bytes(abs(diff))})"


def format_code_block(code: str, language: str = "", indent: int = 4) -> str:
    """
    Formatea código en un bloque with indentación.
    
    Args:
        code: Código a formatear
        language: Lenguaje del código
        indent: Espacios de indentación
        
    Returns:
        Bloque de código formateado
    """
    if not isinstance(code, str):
        raise ValueError("code debe ser una cadena")
    
    indent_str = " " * indent
    lines = code.split('\n')
    indented_lines = [indent_str + line for line in lines]
    
    if language:
        header = f"```{language}"
        footer = "```"
        return f"{header}\n" + "\n".join(indented_lines) + f"\n{footer}"
    else:
        return "\n".join(indented_lines)