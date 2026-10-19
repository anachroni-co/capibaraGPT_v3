"""
Utilidades de validation for CapibaraGPT.
"""

import re
from typing import Any, List, Dict, Union, Optional
from pathlib import Path


def is_valid_url(url: str) -> bool:
    """
    Valida if una cadena es una url válida.
    
    Args:
        url: url a validate
        
    Returns:
        True if es una url válida
    """
    if not isinstance(url, str):
        return False
    
    url_pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return bool(re.match(url_pattern, url))


def is_valid_ipv4(ip: str) -> bool:
    """
    Valida if una cadena es una dirección IPv4 válida.
    
    Args:
        ip: Dirección IP a validate
        
    Returns:
        True if es una IPv4 válida
    """
    if not isinstance(ip, str):
        return False
    
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    
    try:
        for part in parts:
            num = int(part)
            if num < 0 or num > 255:
                return False
    except ValueError:
        return False
    
    return True


def is_valid_port(port: Union[str, int]) -> bool:
    """
    Valida if un puerto es valid (1-65535).
    
    Args:
        port: Puerto a validate
        
    Returns:
        True if es un puerto valid
    """
    try:
        port_num = int(port)
        return 1 <= port_num <= 65535
    except (ValueError, TypeError):
        return False


def validate_file_path(path: Union[str, Path], must_exist: bool = False) -> bool:
    """
    Valida if una path de file es válida.
    
    Args:
        path: path a validate
        must_exist: if el file debe exist
        
    Returns:
        True if la path es válida
    """
    try:
        path_obj = Path(path)
        
        # verify que not contenga caracteres inválidos
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in str(path) for char in invalid_chars):
            return False
        
        if must_exist:
            return path_obj.exists()
        
        return True
    except (TypeError, ValueError):
        return False


def is_valid_json_structure(data: Any, required_keys: Optional[List[str]] = None) -> bool:
    """
    Valida if una structure de data tiene las claves requeridas.
    
    Args:
        data: data a validate
        required_keys: Claves requeridas
        
    Returns:
        True if la structure es válida
    """
    if not isinstance(data, dict):
        return False
    
    if required_keys is None:
        return True
    
    return all(key in data for key in required_keys)


def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> bool:
    """
    Valida if un value está inside de un rank.
    
    Args:
        value: value a validate
        min_val: value minimum
        max_val: value maximum
        
    Returns:
        True if el value está en el rank
    """
    try:
        return min_val <= value <= max_val
    except TypeError:
        return False


def is_valid_hex_color(color: str) -> bool:
    """
    Valida if una cadena es un color hexadecimal valid.
    
    Args:
        color: Color a validate
        
    Returns:
        True if es un color hex valid
    """
    if not isinstance(color, str):
        return False
    
    # permit with or without #
    if color.startswith('#'):
        color = color[1:]
    
    # Debe be 3 or 6 caracteres hexadecimales
    if len(color) not in [3, 6]:
        return False
    
    return all(c in '0123456789ABCDEFabcdef' for c in color)


def validate_password_strength(password: str) -> Dict[str, bool]:
    """
    Valida la fortaleza de una contraseña.
    
    Args:
        password: Contraseña a validate
        
    Returns:
        Diccionario with criterios de validation
    """
    if not isinstance(password, str):
        return {
            'min_length': False,
            'has_uppercase': False,
            'has_lowercase': False,
            'has_digit': False,
            'has_special': False,
            'is_strong': False
        }
    
    criteria = {
        'min_length': len(password) >= 8,
        'has_uppercase': bool(re.search(r'[A-Z]', password)),
        'has_lowercase': bool(re.search(r'[a-z]', password)),
        'has_digit': bool(re.search(r'\d', password)),
        'has_special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }
    
    criteria['is_strong'] = all(criteria.values())
    
    return criteria


def is_valid_username(username: str, min_length: int = 3, max_length: int = 20) -> bool:
    """
    Valida if un nombre de usuario es valid.
    
    Args:
        username: Nombre de usuario a validate
        min_length: length mínima
        max_length: length máxima
        
    Returns:
        True if el nombre de usuario es valid
    """
    if not isinstance(username, str):
        return False
    
    # verify length
    if not (min_length <= len(username) <= max_length):
        return False
    
    # only letras, números and guiones bajos
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False
    
    # not puede start with número
    if username[0].isdigit():
        return False
    
    return True


def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """
    Valida una setup de model.
    
    Args:
        config: setup a validate
        
    Returns:
        list de errores encontrados
    """
    errors = []
    
    # verify claves requeridas
    required_keys = ['hidden_size', 'num_layers', 'learning_rate']
    for key in required_keys:
        if key not in config:
            errors.append(f"Clave requerida faltante: {key}")
    
    # validate tipos and rangos
    if 'hidden_size' in config:
        if not isinstance(config['hidden_size'], int) or config['hidden_size'] <= 0:
            errors.append("hidden_size debe ser un entero positivo")
    
    if 'num_layers' in config:
        if not isinstance(config['num_layers'], int) or config['num_layers'] <= 0:
            errors.append("num_layers debe ser un entero positivo")
    
    if 'learning_rate' in config:
        if not isinstance(config['learning_rate'], (int, float)) or config['learning_rate'] <= 0:
            errors.append("learning_rate debe ser un número positivo")
    
    if 'dropout_rate' in config:
        if not validate_range(config['dropout_rate'], 0.0, 1.0):
            errors.append("dropout_rate debe estar entre 0.0 y 1.0")
    
    return errors


def is_safe_filename(filename: str) -> bool:
    """
    Valida if un nombre de file es secure.
    
    Args:
        filename: Nombre de file a validate
        
    Returns:
        True if el nombre es secure
    """
    if not isinstance(filename, str) or not filename:
        return False
    
    # Caracteres prohibidos
    forbidden_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    if any(char in filename for char in forbidden_chars):
        return False
    
    # Nombres reservados en Windows
    reserved_names = [
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]
    
    name_without_ext = filename.split('.')[0].upper()
    if name_without_ext in reserved_names:
        return False
    
    # not puede start or finish with point or espacio
    if filename.startswith('.') or filename.endswith('.') or filename.endswith(' '):
        return False
    
    return True


def validate_batch_size(batch_size: Any, max_memory_gb: float = 16.0) -> bool:
    """
    Valida if un size de batch es apropiado.
    
    Args:
        batch_size: size de batch a validate
        max_memory_gb: memory máxima available en GB
        
    Returns:
        True if el batch size es valid
    """
    try:
        batch_size = int(batch_size)
    except (ValueError, TypeError):
        return False
    
    if batch_size <= 0:
        return False
    
    # estimation simple: each sample usa ~1MB
    estimated_memory_gb = batch_size * 0.001
    
    return estimated_memory_gb <= max_memory_gb


def is_valid_tensor_shape(shape: List[int]) -> bool:
    """
    Valida if una forma de tensor es válida.
    
    Args:
        shape: Forma del tensor
        
    Returns:
        True if la forma es válida
    """
    if not isinstance(shape, list):
        return False
    
    if len(shape) == 0:
        return False
    
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return False
    
    return True


def validate_hyperparameters(params: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Valida hiperparámetros de entrenamiento.
    
    Args:
        params: Hiperparámetros a validate
        
    Returns:
        Diccionario with errores by categoría
    """
    errors = {
        'learning_rate': [],
        'batch_size': [],
        'epochs': [],
        'optimizer': [],
        'general': []
    }
    
    # validate learning rate
    if 'learning_rate' in params:
        lr = params['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors['learning_rate'].append("Debe ser un número positivo")
        elif lr > 1.0:
            errors['learning_rate'].append("Valor muy alto (>1.0)")
    
    # validate batch size
    if 'batch_size' in params:
        if not validate_batch_size(params['batch_size']):
            errors['batch_size'].append("Tamaño de batch inválido")
    
    # validate epochs
    if 'epochs' in params:
        epochs = params['epochs']
        if not isinstance(epochs, int) or epochs <= 0:
            errors['epochs'].append("Debe ser un entero positivo")
    
    # validate optimizer
    if 'optimizer' in params:
        valid_optimizers = ['adam', 'sgd', 'adamw', 'rmsprop']
        if params['optimizer'].lower() not in valid_optimizers:
            errors['optimizer'].append(f"Debe ser uno de: {valid_optimizers}")
    
    # Filtrar errores vacíos
    return {k: v for k, v in errors.items() if v}