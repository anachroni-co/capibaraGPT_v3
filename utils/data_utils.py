"""
Utilidades for manipulación de data en CapibaraGPT.
"""

import json
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path


def flatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """
    Aplana un diccionario anidado.
    
    Args:
        data: Diccionario a aplanar
        separator: Separador for las claves anidadas
        
    Returns:
        Diccionario aplanado
    """
    def _flatten(obj: Any, parent_key: str = "") -> Dict[str, Any]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            return {parent_key: obj}
        
        return dict(items)
    
    return _flatten(data)


def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """
    Desaplana un diccionario aplanado.
    
    Args:
        data: Diccionario aplanado
        separator: Separador usado en las claves
        
    Returns:
        Diccionario anidado
    """
    result = {}
    
    for key, value in data.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona múltiples diccionarios de forma recursiva.
    
    Args:
        *dicts: Diccionarios a merge
        
    Returns:
        Diccionario fusionado
    """
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
    
    return result


def filter_dict(data: Dict[str, Any], keys: List[str], include: bool = True) -> Dict[str, Any]:
    """
    Filtra un diccionario by claves específicas.
    
    Args:
        data: Diccionario a filtrar
        keys: list de claves
        include: if True, incluye only las claves especificadas. if False, las excluye.
        
    Returns:
        Diccionario filtrado
    """
    if include:
        return {k: v for k, v in data.items() if k in keys}
    else:
        return {k: v for k, v in data.items() if k not in keys}


def deep_get(data: Dict[str, Any], path: str, default: Any = None, separator: str = ".") -> Any:
    """
    Obtiene un valor de un diccionario anidado usando una ruta.
    
    Args:
        data: Diccionario anidado
        path: Ruta a la clave (ej: "user.profile.name")
        default: Valor por defecto si no se encuentra
        separator: Separador de la ruta
        
    Returns:
        Valor encontrado o valor por defecto
    """
    keys = path.split(separator)
    current = data
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def deep_set(data: Dict[str, Any], path: str, value: Any, separator: str = ".") -> None:
    """
    Establece un value en un diccionario anidado usando una path.
    
    Args:
        data: Diccionario anidado (se modifica in-place)
        path: path a la key
        value: value a establish
        separator: Separador de la path
    """
    keys = path.split(separator)
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Divide una list en chunks de size específico.
    
    Args:
        data: list a divide
        chunk_size: size de each chunk
        
    Returns:
        list de chunks
        
    Raises:
        ValueError: if chunk_size es menor or equal a 0
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0")
    
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def deduplicate_list(data: List[Any], preserve_order: bool = True) -> List[Any]:
    """
    Elimina duplicados de una list.
    
    Args:
        data: list with posibles duplicados
        preserve_order: if preservar el order original
        
    Returns:
        list without duplicados
    """
    if preserve_order:
        seen = set()
        result = []
        for item in data:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        return list(set(data))


def transpose_list_of_dicts(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Transpone una list de diccionarios a un diccionario de listas.
    
    Args:
        data: list de diccionarios
        
    Returns:
        Diccionario de listas
    """
    if not data:
        return {}
    
    result = {}
    for d in data:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    
    return result


def group_by(data: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Agrupa una list de diccionarios by una key específica.
    
    Args:
        data: list de diccionarios
        key: key by la cual group
        
    Returns:
        Diccionario with grupos
    """
    result = {}
    
    for item in data:
        group_key = item.get(key)
        if group_key not in result:
            result[group_key] = []
        result[group_key].append(item)
    
    return result


def safe_json_load(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    load un file JSON de forma segura.
    
    Args:
        file_path: path al file JSON
        
    Returns:
        Contenido del JSON or None if hay error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def safe_json_save(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Guarda data en un file JSON de forma segura.
    
    Args:
        data: data a save
        file_path: path del file
        indent: Indentación del JSON
        
    Returns:
        True if se guardó exitosamente, False otherwise
    """
    try:
        # create directory padre if not existe
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (PermissionError, OSError, TypeError):
        return False


def validate_data_types(data: Dict[str, Any], schema: Dict[str, type]) -> List[str]:
    """
    Valida los tipos de data en un diccionario.
    
    Args:
        data: data a validate
        schema: schema with tipos esperados
        
    Returns:
        list de errores encontrados
    """
    errors = []
    
    for key, expected_type in schema.items():
        if key not in data:
            errors.append(f"Clave faltante: {key}")
        elif not isinstance(data[key], expected_type):
            errors.append(f"Tipo incorrecto para {key}: esperado {expected_type.__name__}, obtenido {type(data[key]).__name__}")
    
    return errors


def normalize_keys(data: Dict[str, Any], case: str = "lower") -> Dict[str, Any]:
    """
    Normaliza las claves de un diccionario.
    
    Args:
        data: Diccionario a normalizar
        case: Tipo de normalización ("lower", "upper", "title")
        
    Returns:
        Diccionario con claves normalizadas
    """
    if case == "lower":
        return {k.lower(): v for k, v in data.items()}
    elif case == "upper":
        return {k.upper(): v for k, v in data.items()}
    elif case == "title":
        return {k.title(): v for k, v in data.items()}
    else:
        raise ValueError("case debe ser 'lower', 'upper' o 'title'")


def find_differences(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encuentra las diferencias between dos diccionarios.
    
    Args:
        dict1: first diccionario
        dict2: Segundo diccionario
        
    Returns:
        Diccionario with las diferencias
    """
    differences = {
        "added": {},
        "removed": {},
        "modified": {}
    }
    
    # Claves añadidas
    for key in dict2.keys() - dict1.keys():
        differences["added"][key] = dict2[key]
    
    # Claves removidas
    for key in dict1.keys() - dict2.keys():
        differences["removed"][key] = dict1[key]
    
    # Claves modificadas
    for key in dict1.keys() & dict2.keys():
        if dict1[key] != dict2[key]:
            differences["modified"][key] = {
                "old": dict1[key],
                "new": dict2[key]
            }
    
    return differences


def sample_data(data: List[Any], n: int, random_seed: Optional[int] = None) -> List[Any]:
    """
    Toma una sample aleatoria de los data.
    
    Args:
        data: list de data
        n: number of elementos a muestrear
        random_seed: Semilla for reproducibilidad
        
    Returns:
        sample de los data
    """
    import random
    
    if random_seed is not None:
        random.seed(random_seed)
    
    if n >= len(data):
        return data.copy()
    
    return random.sample(data, n)


def batch_process(data: List[Any], batch_size: int, process_func) -> List[Any]:
    """
    Procesa data en lotes usando una function.
    
    Args:
        data: data a process
        batch_size: size del lote
        process_func: function de procesamiento
        
    Returns:
        Resultados procesados
    """
    results = []
    
    for batch in chunk_list(data, batch_size):
        batch_result = process_func(batch)
        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)
    
    return results


def get_nested_size(data: Any) -> int:
    """
    Calcula el size de una structure de data anidada.
    
    Args:
        data: structure de data
        
    Returns:
        Número total of elementos
    """
    if isinstance(data, dict):
        return sum(get_nested_size(v) for v in data.values())
    elif isinstance(data, (list, tuple)):
        return sum(get_nested_size(item) for item in data)
    else:
        return 1