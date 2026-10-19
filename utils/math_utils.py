"""
Utilidades matemáticas for CapibaraGPT.
"""

import math
from typing import List, Union, Optional


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    División segura que evita división by cero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: value by defect if el denominador es cero
        
    Returns:
        result de la división or value by defect
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Limita un value between un minimum and maximum.
    
    Args:
        value: value a limitar
        min_val: value minimum
        max_val: value maximum
        
    Returns:
        value limitado between min_val and max_val
    """
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """
    Interpolación lineal between dos valores.
    
    Args:
        a: value inicial
        b: value end
        t: Factor de interpolación (0.0 a 1.0)
        
    Returns:
        value interpolado
    """
    return a + t * (b - a)


def normalize_list(values: List[float]) -> List[float]:
    """
    Normaliza una list de valores for que sumen 1.0.
    
    Args:
        values: list de valores a normalizar
        
    Returns:
        list normalizada
        
    Raises:
        ValueError: if la list está vacía or la suma es cero
    """
    if not values:
        raise ValueError("La lista no puede estar vacía")
    
    total = sum(values)
    if total == 0:
        raise ValueError("La suma de los valores no puede ser cero")
    
    return [v / total for v in values]


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """
    Aplica la function softmax a una list de valores.
    
    Args:
        values: list de valores
        temperature: parameter de temperatura for suavizar la distribución
        
    Returns:
        list with probabilidades softmax
        
    Raises:
        ValueError: if la list está vacía or la temperatura es cero
    """
    if not values:
        raise ValueError("La lista no puede estar vacía")
    
    if temperature == 0:
        raise ValueError("La temperatura no puede ser cero")
    
    # apply temperatura and estabilidad numérica
    scaled_values = [v / temperature for v in values]
    max_val = max(scaled_values)
    exp_values = [math.exp(v - max_val) for v in scaled_values]
    
    return normalize_list(exp_values)


def moving_average(values: List[float], window_size: int) -> List[float]:
    """
    Calcula la media móvil de una list de valores.
    
    Args:
        values: list de valores
        window_size: size de la ventana
        
    Returns:
        list with las medias móviles
        
    Raises:
        ValueError: if window_size es invalid
    """
    if window_size <= 0:
        raise ValueError("El tamaño de ventana debe ser positivo")
    
    if window_size > len(values):
        raise ValueError("El tamaño de ventana no puede ser mayor que la lista")
    
    result = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        result.append(sum(window) / window_size)
    
    return result


def calculate_perplexity(loss: float) -> float:
    """
    Calcula la perplejidad a leave de la pérdida.
    
    Args:
        loss: value de pérdida
        
    Returns:
        value de perplejidad
    """
    return math.exp(loss)


def log_sum_exp(values: List[float]) -> float:
    """
    Calcula log(sum(exp(x))) de forma numéricamente estable.
    
    Args:
        values: list de valores
        
    Returns:
        result de log-sum-exp
        
    Raises:
        ValueError: if la list está vacía
    """
    if not values:
        raise ValueError("La lista no puede estar vacía")
    
    max_val = max(values)
    exp_sum = sum(math.exp(v - max_val) for v in values)
    return max_val + math.log(exp_sum)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calcula la similitud coseno between dos vectores.
    
    Args:
        a: first vector
        b: Segundo vector
        
    Returns:
        Similitud coseno (-1 a 1)
        
    Raises:
        ValueError: if los vectores tienen diferentes longitudes or son vacíos
    """
    if not a or not b:
        raise ValueError("Los vectores no pueden estar vacíos")
    
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma longitud")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)