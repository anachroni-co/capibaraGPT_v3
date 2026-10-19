"""
Utilidades de setup for CapibaraGPT.
"""

import os
import json
import yaml  # type: ignore
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    load un file de setup (JSON or YAML).
    
    Args:
        file_path: path al file de setup
        
    Returns:
        Diccionario with la setup
        
    Raises:
        FileNotFoundError: if el file not existe
        ValueError: if el format not es soportado
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {file_path}")
    
    suffix = path.suffix.lower()
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if suffix in ['.json']:
                return json.load(f)
            elif suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            else:
                raise ValueError(f"Formato de archivo no soportado: {suffix}")
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Error al parsear el archivo {file_path}: {e}")


def save_config_file(config: Dict[str, Any], file_path: Union[str, Path], format_type: str = "auto") -> None:
    """
    Guarda un diccionario de configuración en un archivo.
    
    Args:
        config: Configuración a guardar
        file_path: Ruta donde guardar el archivo
        format_type: Formato del archivo ("json", "yaml", "auto")
        
    Raises:
        ValueError: Si el formato no es soportado
    """
    path = Path(file_path)
    
    # create directory padre if not existe
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "auto":
        suffix = path.suffix.lower()
        if suffix in ['.json']:
            format_type = "json"
        elif suffix in ['.yaml', '.yml']:
            format_type = "yaml"
        else:
            format_type = "json"  # Default
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            if format_type == "json":
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif format_type == "yaml":
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Formato no soportado: {format_type}")
    except (TypeError, yaml.YAMLError) as e:
        raise ValueError(f"Error al guardar la configuración: {e}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona múltiples configuraciones de forma recursiva.
    
    Args:
        *configs: Configuraciones a merge
        
    Returns:
        setup fusionada
    """
    result = {}
    
    for config in configs:
        if not isinstance(config, dict):
            continue
            
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None, separator: str = ".") -> Any:
    """
    Obtiene un valor de configuración usando una ruta de claves.
    
    Args:
        config: Diccionario de configuración
        key_path: Ruta a la clave (ej: "model.hidden_size")
        default: Valor por defecto
        separator: Separador de la ruta
        
    Returns:
        Valor encontrado o valor por defecto
    """
    keys = key_path.split(separator)
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any, separator: str = ".") -> None:
    """
    Establece un value de setup usando una path de claves.
    
    Args:
        config: Diccionario de setup (se modifica in-place)
        key_path: path a la key
        value: value a establish
        separator: Separador de la path
    """
    keys = key_path.split(separator)
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Valida una setup against un schema simple.
    
    Args:
        config: setup a validate
        schema: schema de validation
        
    Returns:
        list de errores encontrados
    """
    errors = []
    
    def _validate_recursive(data: Any, schema_part: Any, path: str = "") -> None:
        if isinstance(schema_part, dict):
            if not isinstance(data, dict):
                errors.append(f"{path}: esperado dict, obtenido {type(data).__name__}")
                return
            
            for key, expected_type in schema_part.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in data:
                    errors.append(f"{current_path}: clave requerida faltante")
                else:
                    _validate_recursive(data[key], expected_type, current_path)
        
        elif isinstance(schema_part, type):
            if not isinstance(data, schema_part):
                errors.append(f"{path}: esperado {schema_part.__name__}, obtenido {type(data).__name__}")
        
        elif isinstance(schema_part, list) and len(schema_part) == 1:
            # list with type de elementos
            if not isinstance(data, list):
                errors.append(f"{path}: esperado list, obtenido {type(data).__name__}")
            else:
                for i, item in enumerate(data):
                    _validate_recursive(item, schema_part[0], f"{path}[{i}]")
    
    _validate_recursive(config, schema)
    return errors


def load_env_config(prefix: str = "CAPIBARA_") -> Dict[str, str]:
    """
    load setup since variables de entorno.
    
    Args:
        prefix: Prefijo de las variables de entorno
        
    Returns:
        Diccionario with las variables de entorno
    """
    config = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # remove prefijo and convertir a minúsculas
            config_key = key[len(prefix):].lower()
            config[config_key] = value
    
    return config


def expand_config_variables(config: Dict[str, Any], variables: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Expande variables en la setup.
    
    Args:
        config: setup with posibles variables
        variables: Variables adicionales for expansión
        
    Returns:
        setup with variables expandidas
    """
    if variables is None:
        variables = {}
    
    # combine with variables de entorno
    all_variables = {**os.environ, **variables}
    
    def _expand_value(value: Any) -> Any:
        if isinstance(value, str):
            # expand variables del type ${VAR} or $VAR
            import re
            
            def replace_var(match):
                var_name = match.group(1) or match.group(2)
                return all_variables.get(var_name, match.group(0))
            
            # pattern for ${VAR} and $VAR
            pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
            return re.sub(pattern, replace_var, value)
        
        elif isinstance(value, dict):
            return {k: _expand_value(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [_expand_value(item) for item in value]
        
        else:
            return value
    
    return _expand_value(config)


def create_config_template(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea una template de setup basada en un schema.
    
    Args:
        schema: schema de setup
        
    Returns:
        template de setup
    """
    def _create_template(schema_part: Any) -> Any:
        if isinstance(schema_part, dict):
            return {key: _create_template(value) for key, value in schema_part.items()}
        
        elif isinstance(schema_part, type):
            if schema_part == str:
                return ""
            elif schema_part == int:
                return 0
            elif schema_part == float:
                return 0.0
            elif schema_part == bool:
                return False
            elif schema_part == list:
                return []
            elif schema_part == dict:
                return {}
            else:
                return None
        
        elif isinstance(schema_part, list) and len(schema_part) == 1:
            return []
        
        else:
            return None
    
    return _create_template(schema)


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara dos configuraciones and encuentra las diferencias.
    
    Args:
        config1: Primera setup
        config2: Segunda setup
        
    Returns:
        Diccionario with las diferencias
    """
    differences = {
        "added": {},
        "removed": {},
        "modified": {},
        "unchanged": {}
    }
    
    def _compare_recursive(c1: Any, c2: Any, path: str = "") -> None:
        if isinstance(c1, dict) and isinstance(c2, dict):
            # Claves añadidas
            for key in c2.keys() - c1.keys():
                current_path = f"{path}.{key}" if path else key
                differences["added"][current_path] = c2[key]
            
            # Claves removidas
            for key in c1.keys() - c2.keys():
                current_path = f"{path}.{key}" if path else key
                differences["removed"][current_path] = c1[key]
            
            # Claves comunes
            for key in c1.keys() & c2.keys():
                current_path = f"{path}.{key}" if path else key
                _compare_recursive(c1[key], c2[key], current_path)
        
        else:
            if c1 != c2:
                differences["modified"][path] = {"old": c1, "new": c2}
            else:
                differences["unchanged"][path] = c1
    
    _compare_recursive(config1, config2)
    return differences


def normalize_config_keys(config: Dict[str, Any], case: str = "lower") -> Dict[str, Any]:
    """
    Normaliza las claves de configuración.
    
    Args:
        config: Configuración a normalizar
        case: Tipo de normalización ("lower", "upper", "snake", "camel")
        
    Returns:
        Configuración con claves normalizadas
    """
    def _normalize_key(key: str) -> str:
        if case == "lower":
            return key.lower()
        elif case == "upper":
            return key.upper()
        elif case == "snake":
            import re
            # Convertir CamelCase a snake_case
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', key)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        elif case == "camel":
            # Convertir snake_case a camelCase
            components = key.split('_')
            return components[0].lower() + ''.join(word.capitalize() for word in components[1:])
        else:
            return key
    
    def _normalize_recursive(data: Any) -> Any:
        if isinstance(data, dict):
            return {_normalize_key(k): _normalize_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_normalize_recursive(item) for item in data]
        else:
            return data
    
    return _normalize_recursive(config)


def get_config_size(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Calcula estadísticas de size de la setup.
    
    Args:
        config: setup a analyze
        
    Returns:
        Diccionario with estadísticas
    """
    stats = {
        "total_keys": 0,
        "nested_levels": 0,
        "string_values": 0,
        "numeric_values": 0,
        "boolean_values": 0,
        "list_values": 0,
        "dict_values": 0
    }
    
    def _analyze_recursive(data: Any, level: int = 0) -> None:
        stats["nested_levels"] = max(stats["nested_levels"], level)
        
        if isinstance(data, dict):
            stats["total_keys"] += len(data)
            stats["dict_values"] += 1
            for value in data.values():
                _analyze_recursive(value, level + 1)
        
        elif isinstance(data, list):
            stats["list_values"] += 1
            for item in data:
                _analyze_recursive(item, level + 1)
        
        elif isinstance(data, bool):
            stats["boolean_values"] += 1
        
        elif isinstance(data, str):
            stats["string_values"] += 1
        
        elif isinstance(data, (int, float)):
            stats["numeric_values"] += 1
    
    _analyze_recursive(config)
    return stats