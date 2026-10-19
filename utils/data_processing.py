"""
Módulo for procesamiento de data.
"""

import numpy as np  # type: ignore
from typing import Dict, Any, Union, List, Tuple
from capibara.utils.error_handling import handle_error
from .error_handling import DataProcessingError, handle_error

# Importación lazy de jax for evitar circularidad
def get_jnp():
    from jax import numpy as jnp
    return jnp

@handle_error(DataProcessingError)
def process_data(
    batch: Union[List[str], Dict[str, Any]],
    tokenizer: Any = None,
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True
) -> Dict[str, Any]:
    """
    Procesa un lote de data for entrenamiento or inferencia.
    
    Args:
        batch: Lote de data a process
        tokenizer: Tokenizador preentrenado
        max_length: length máxima de secuencia
        padding: Estrategia de padding
        truncation: if se debe truncar
        
    Returns:
        Dict with tensores procesados
    """
    jnp = get_jnp()
    
    if tokenizer is not None:
        # Tokenizar
        inputs = tokenizer(
            batch,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='np'
        )
        
        # Convertir a JAX
        return {
            'input_ids': jnp.array(inputs['input_ids'], dtype=jnp.int32),
            'attention_mask': jnp.array(inputs['attention_mask'], dtype=jnp.int32)
        }
    else:
        # process directamente
        if isinstance(batch, dict):
            return {
                'input_ids': jnp.array(batch['input_ids'], dtype=jnp.int32),
                'attention_mask': jnp.array(batch['attention_mask'], dtype=jnp.int32)
            }
        else:
            raise DataProcessingError("Se requiere tokenizer o diccionario de datos")

@handle_error(DataProcessingError)
def save_processed_data(
    data: Union[np.ndarray, Any],
    output_path: str,
    format: str = 'numpy'
) -> None:
    """
    Guarda data procesados.
    
    Args:
        data: data a save
        output_path: path de output
        format: format de output (numpy or jax)
    """
    jnp = get_jnp()
    
    if format == 'numpy':
        np.save(output_path, np.array(data))
    elif format == 'jax':
        jnp.save(output_path, jnp.array(data))
    else:
        raise DataProcessingError(f"Formato no soportado: {format}")

@handle_error(DataProcessingError)
def load_processed_data(
    input_path: str,
    format: str = 'numpy'
) -> Union[np.ndarray, Any]:
    """
    load data procesados.
    
    Args:
        input_path: path de input
        format: format de input (numpy or jax)
        
    Returns:
        data cargados
    """
    jnp = get_jnp()
    
    if format == 'numpy':
        return np.load(input_path)
    elif format == 'jax':
        return jnp.load(input_path)
    else:
        raise DataProcessingError(f"Formato no soportado: {format}")

@handle_error(DataProcessingError)
def text_to_bytes(text: str) -> List[int]:
    """
    Convierte texto a bytes.
    
    Args:
        text: Texto a convertir
        
    Returns:
        list de bytes
    """
    return [ord(c) for c in text]

@handle_error(DataProcessingError)
def bytes_to_text(bytes_data: List[int]) -> str:
    """
    Convierte bytes a texto.
    
    Args:
        bytes_data: list de bytes
        
    Returns:
        Texto convertido
    """
    return ''.join(chr(b) for b in bytes_data)

@handle_error(DataProcessingError)
def prepare_training_data(texts: List[str]) -> List[Tuple[List[int], List[int]]]:
    """
    Prepara data for entrenamiento.
    
    Args:
        texts: list de textos
        
    Returns:
        list de tuplas (input_bytes, target_bytes)
    """
    training_data = []
    for text in texts:
        bytes_data = text_to_bytes(text)
        input_bytes = bytes_data[:-1]
        target_bytes = bytes_data[1:]
        training_data.append((input_bytes, target_bytes))
    return training_data