"""
Utilidades de logging for CapibaraGPT.
"""

import logging
import sys
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json


def setup_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    include_console: bool = True
) -> logging.Logger:
    """
    Configura un logger with format personalizado.
    
    Args:
        name: Nombre del logger
        level: level de logging
        log_file: file de log (optional)
        format_string: format personalizado (optional)
        include_console: if include output a consola
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # clean handlers existentes
    logger.handlers.clear()
    
    # format by defect
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Handler for consola
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler for file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_structured_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Crea un logger que produce logs estructurados en JSON.
    
    Args:
        name: Nombre del logger
        level: level de logging
        log_file: file de log (optional)
        
    Returns:
        Logger configurado for JSON
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # add information de excepción if existe
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_entry, ensure_ascii=False)
    
    formatter = JSONFormatter()
    
    # Handler for consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler for file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_log_level_from_string(level_str: str) -> int:
    """
    Convierte una cadena de level de log a entero.
    
    Args:
        level_str: level how cadena
        
    Returns:
        level how entero
        
    Raises:
        ValueError: if el level not es valid
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level_upper = level_str.upper()
    if level_upper not in level_map:
        raise ValueError(f"Nivel de log inválido: {level_str}")
    
    return level_map[level_upper]


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorador for loggear llamadas a funciones.
    
    Args:
        logger: Logger a use
        level: level de log
        
    Returns:
        Decorador
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(level, f"Llamando a {func_name} con args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func_name} completado exitosamente")
                return result
            except Exception as e:
                logger.error(f"Error en {func_name}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger, level: int = logging.INFO):
    """
    Decorador for loggear tiempo de execution de funciones.
    
    Args:
        logger: Logger a use
        level: level de log
        
    Returns:
        Decorador
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            func_name = func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log(level, f"{func_name} ejecutado en {execution_time:.4f} segundos")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func_name} falló después de {execution_time:.4f} segundos: {e}")
                raise
        
        return wrapper
    return decorator


def create_rotating_logger(
    name: str,
    log_file: Union[str, Path],
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """
    Crea un logger with rotación de archivos.
    
    Args:
        name: Nombre del logger
        log_file: file de log
        max_bytes: size maximum del file
        backup_count: number of archivos de respaldo
        level: level de logging
        
    Returns:
        Logger with rotación
    """
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


def create_timed_rotating_logger(
    name: str,
    log_file: Union[str, Path],
    when: str = 'midnight',
    interval: int = 1,
    backup_count: int = 7,
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """
    Crea un logger with rotación basada en tiempo.
    
    Args:
        name: Nombre del logger
        log_file: file de log
        when: Cuándo rotar ('midnight', 'H', 'D', etc.)
        interval: Intervalo de rotación
        backup_count: number of archivos de respaldo
        level: level de logging
        
    Returns:
        Logger with rotación temporary
    """
    from logging.handlers import TimedRotatingFileHandler
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    handler = TimedRotatingFileHandler(
        log_path,
        when=when,
        interval=interval,
        backupCount=backup_count
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


def filter_logs_by_level(log_file: Union[str, Path], min_level: str) -> List[str]:
    """
    Filtra logs by level minimum.
    
    Args:
        log_file: file de log
        min_level: level minimum a include
        
    Returns:
        list de líneas filtradas
    """
    min_level_int = get_log_level_from_string(min_level)
    filtered_lines = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # search el level en la line
                for level_name, level_int in [
                    ('CRITICAL', logging.CRITICAL),
                    ('ERROR', logging.ERROR),
                    ('WARNING', logging.WARNING),
                    ('INFO', logging.INFO),
                    ('DEBUG', logging.DEBUG)
                ]:
                    if level_name in line and level_int >= min_level_int:
                        filtered_lines.append(line.strip())
                        break
    except FileNotFoundError:
        pass
    
    return filtered_lines


def parse_log_file(log_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Parsea un file de log and extrae information estructurada.
    
    Args:
        log_file: file de log
        
    Returns:
        list de entradas de log parseadas
    """
    import re
    
    log_entries = []
    log_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
    )
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                match = log_pattern.match(line.strip())
                if match:
                    timestamp_str, logger_name, level, message = match.groups()
                    
                    entry = {
                        'line_number': line_num,
                        'timestamp': timestamp_str,
                        'logger': logger_name,
                        'level': level,
                        'message': message
                    }
                    log_entries.append(entry)
    except FileNotFoundError:
        pass
    
    return log_entries


def get_log_statistics(log_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Obtiene estadísticas de un file de log.
    
    Args:
        log_file: file de log
        
    Returns:
        Diccionario with estadísticas
    """
    entries = parse_log_file(log_file)
    
    if not entries:
        return {
            'total_entries': 0,
            'levels': {},
            'loggers': {},
            'first_entry': None,
            'last_entry': None
        }
    
    # tell by level
    level_counts = {}
    logger_counts = {}
    
    for entry in entries:
        level = entry['level']
        logger = entry['logger']
        
        level_counts[level] = level_counts.get(level, 0) + 1
        logger_counts[logger] = logger_counts.get(logger, 0) + 1
    
    return {
        'total_entries': len(entries),
        'levels': level_counts,
        'loggers': logger_counts,
        'first_entry': entries[0]['timestamp'] if entries else None,
        'last_entry': entries[-1]['timestamp'] if entries else None
    }


def cleanup_old_logs(log_directory: Union[str, Path], days_to_keep: int = 30) -> int:
    """
    Limpia logs antiguos de un directory.
    
    Args:
        log_directory: directory de logs
        days_to_keep: Días a maintain
        
    Returns:
        number of archivos eliminados
    """
    from datetime import timedelta
    
    log_dir = Path(log_directory)
    if not log_dir.exists():
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_count = 0
    
    for log_file in log_dir.glob('*.log*'):
        try:
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                log_file.unlink()
                deleted_count += 1
        except (OSError, PermissionError):
            continue
    
    return deleted_count


def create_context_logger(base_logger: logging.Logger, context: Dict[str, Any]) -> logging.Logger:
    """
    Crea un logger que incluye contexto adicional en todos los mensajes.
    
    Args:
        base_logger: Logger base
        context: Contexto a include
        
    Returns:
        Logger with contexto
    """
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            context_str = ', '.join(f"{k}={v}" for k, v in self.extra.items())
            return f"[{context_str}] {msg}", kwargs
    
    return ContextAdapter(base_logger, context)


def log_memory_usage(logger: logging.Logger, message: str = "Uso de memoria") -> None:
    """
    Loggea el uso current de memory.
    
    Args:
        logger: Logger a use
        message: Mensaje personalizado
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"{message}: {memory_mb:.2f} MB")
    except ImportError:
        logger.warning("psutil no disponible para monitoreo de memoria")


def create_multi_level_logger(
    name: str,
    log_files: Dict[str, Union[str, Path]],
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """
    Crea un logger que escribe diferentes niveles a diferentes archivos.
    
    Args:
        name: Nombre del logger
        log_files: Mapeo de level a file
        level: level base del logger
        
    Returns:
        Logger multi-level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    for level_name, log_file in log_files.items():
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        
        # configure filtro by level
        level_int = get_log_level_from_string(level_name)
        handler.setLevel(level_int)
        
        # Filtro for only este level
        class LevelFilter:
            def __init__(self, level):
                self.level = level
            
            def filter(self, record):
                return record.levelno == self.level
        
        handler.addFilter(LevelFilter(level_int))
        logger.addHandler(handler)
    
    return logger