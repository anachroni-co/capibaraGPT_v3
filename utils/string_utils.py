"""
Utilidades for manipulación de strings en CapibaraGPT.
"""

import re
from typing import List, Optional, Dict, Union


def clean_text(text: str, remove_extra_spaces: bool = True) -> str:
    """
    Limpia un texto eliminando caracteres not deseados.
    
    Args:
        text: Texto a clean
        remove_extra_spaces: if eliminate espacios extra
        
    Returns:
        Texto limpio
    """
    if not isinstance(text, str):
        raise TypeError("El texto debe ser una cadena")
    
    # eliminate caracteres de control
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # eliminate espacios extra if se solicita
    if remove_extra_spaces:
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
    
    return cleaned


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Trunca un texto a una length máxima.
    
    Args:
        text: Texto a truncar
        max_length: length máxima
        suffix: Sufijo a add if se trunca
        
    Returns:
        Texto truncado
        
    Raises:
        ValueError: if max_length es negativo
    """
    if max_length < 0:
        raise ValueError("max_length debe ser no negativo")
    
    if len(text) <= max_length:
        return text
    
    if max_length < len(suffix):
        return text[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


def split_into_sentences(text: str) -> List[str]:
    """
    Divide un texto en oraciones.
    
    Args:
        text: Texto a divide
        
    Returns:
        list de oraciones
    """
    # pattern simple for divide oraciones
    sentences = re.split(r'[.!?]+', text)
    
    # clean and filtrar oraciones vacías
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def count_words(text: str) -> int:
    """
    Cuenta las palabras en un texto.
    
    Args:
        text: Texto a analyze
        
    Returns:
        number of palabras
    """
    if not text.strip():
        return 0
    
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extrae palabras key de un texto.
    
    Args:
        text: Texto a analyze
        min_length: length mínima de las palabras
        
    Returns:
        list de palabras key únicas
    """
    # Convertir a minúsculas and extraer palabras
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filtrar by length and eliminate duplicados
    keywords = list(set(word for word in words if len(word) >= min_length))
    
    return sorted(keywords)


def normalize_whitespace(text: str) -> str:
    """
    Normaliza los espacios en blanco en un texto.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto with espacios normalizados
    """
    # Reemplazar todos los tipos de espacios en blanco with espacios normales
    normalized = re.sub(r'\s+', ' ', text)
    return normalized.strip()


def remove_html_tags(text: str) -> str:
    """
    Elimina etiquetas HTML de un texto.
    
    Args:
        text: Texto with posibles etiquetas HTML
        
    Returns:
        Texto without etiquetas HTML
    """
    # pattern for eliminate etiquetas HTML
    clean = re.sub(r'<[^>]+>', '', text)
    return clean


def capitalize_sentences(text: str) -> str:
    """
    Capitaliza la primera letra de each oración.
    
    Args:
        text: Texto a capitalizar
        
    Returns:
        Texto with oraciones capitalizadas
    """
    # divide en oraciones and capitalizar each una
    sentences = split_into_sentences(text)
    capitalized = [sentence.capitalize() for sentence in sentences]
    
    return '. '.join(capitalized) + ('.' if capitalized else '')


def find_urls(text: str) -> List[str]:
    """
    Encuentra URLs en un texto.
    
    Args:
        text: Texto a analyze
        
    Returns:
        list de URLs encontradas
    """
    # pattern simple for URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    return urls


def replace_multiple_spaces(text: str, replacement: str = " ") -> str:
    """
    Reemplaza múltiples espacios consecutivos with un reemplazo.
    
    Args:
        text: Texto a process
        replacement: Cadena de reemplazo
        
    Returns:
        Texto with espacios reemplazados
    """
    return re.sub(r' {2,}', replacement, text)


def extract_numbers(text: str) -> List[float]:
    """
    Extrae números de un texto.
    
    Args:
        text: Texto a analyze
        
    Returns:
        list de números encontrados
    """
    # pattern for números (enteros and decimales, incluyendo .5)
    number_pattern = r'-?(?:\d+\.?\d*|\.\d+)'
    numbers = re.findall(number_pattern, text)
    
    # Convertir a float and filtrar strings vacíos
    result = []
    for num_str in numbers:
        if num_str and num_str != '.':
            try:
                result.append(float(num_str))
            except ValueError:
                continue
    
    return result


def is_email(text: str) -> bool:
    """
    Verifica if un texto es una dirección de email válida.
    
    Args:
        text: Texto a verify
        
    Returns:
        True if es un email valid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, text.strip()))


def mask_sensitive_info(text: str, mask_char: str = "*") -> str:
    """
    Enmascara information sensible how emails and números de teléfono.
    
    Args:
        text: Texto a enmascarar
        mask_char: Carácter for enmascarar
        
    Returns:
        Texto with information sensible enmascarada
    """
    # Enmascarar emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    masked = re.sub(email_pattern, f'{mask_char * 8}@{mask_char * 5}.com', text)
    
    # Enmascarar números que parecen teléfonos (secuencias de 7+ dígitos)
    phone_pattern = r'\b\d{7,}\b'
    masked = re.sub(phone_pattern, mask_char * 10, masked)
    
    return masked


def get_text_stats(text: str) -> Dict[str, Union[int, float]]:
    """
    Obtiene estadísticas de un texto.
    
    Args:
        text: Texto a analyze
        
    Returns:
        Diccionario with estadísticas del texto
    """
    sentences = split_into_sentences(text)
    words = count_words(text)
    
    return {
        'characters': len(text),
        'characters_no_spaces': len(text.replace(' ', '')),
        'words': words,
        'sentences': len(sentences),
        'avg_words_per_sentence': words / len(sentences) if sentences else 0,
        'avg_chars_per_word': len(text.replace(' ', '')) / words if words else 0
    }