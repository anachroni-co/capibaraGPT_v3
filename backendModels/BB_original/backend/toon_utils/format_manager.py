"""
Gestor de formatos TOON para el proyecto Capibara6
Decide cuándo usar TOON vs JSON para optimizar tokens
"""

from typing import Any, Dict
from .toon_converter import toon_encode, toon_decode
import json

class FormatManager:
    @staticmethod
    def should_use_toon(data: Any) -> bool:
        """
        Determina si TOON es más eficiente que JSON para los datos dados
        Basado en la heurística: arrays grandes de objetos con la misma estructura
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if FormatManager._is_large_uniform_array(value):
                    return True
        elif FormatManager._is_large_uniform_array(data):
            return True
        return False

    @staticmethod
    def _is_large_uniform_array(data: Any) -> bool:
        """
        Verifica si los datos son un array grande de objetos con la misma estructura
        Consideramos 'grande' como > 5 elementos basado en benchmarks de TOON
        """
        if not isinstance(data, list) or len(data) <= 5:
            return False

        if not data:
            return False

        # Verificar que todos los elementos sean diccionarios
        if not all(isinstance(item, dict) for item in data):
            return False

        # Verificar que todos tengan las mismas claves
        first_keys = set(data[0].keys()) if data else set()
        return all(set(item.keys()) == first_keys for item in data)

    @staticmethod
    def encode(data: Any, preferred_format: str = 'auto') -> tuple[str, str]:
        """
        Codifica los datos en el formato más eficiente
        Devuelve (contenido_codificado, tipo_formato)
        """
        if preferred_format == 'toon' or (preferred_format == 'auto' and FormatManager.should_use_toon(data)):
            try:
                toon_content = toon_encode(data)
                return toon_content, 'toon'
            except:
                # Si falla TOON, usar JSON
                json_content = json.dumps(data, ensure_ascii=False)
                return json_content, 'json'
        else:
            json_content = json.dumps(data, ensure_ascii=False)
            return json_content, 'json'

    @staticmethod
    def decode(content: str, format_type: str = 'json') -> Any:
        """
        Decodifica contenido desde el formato especificado
        """
        if format_type == 'toon':
            return toon_decode(content)
        else:
            return json.loads(content)