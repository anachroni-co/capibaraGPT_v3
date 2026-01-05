#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from toon_utils import ToonEncoderUltraOptimized, ToonParserUltraOptimized

# Probar el comportamiento real con versiones ultra optimizadas
test_data = {
    "id": 1,
    "name": "Test",
    "values": [1, 2, 3],  # Este es un array como valor de campo
    "nested": {"a": 1, "b": 2}
}

print("Datos originales:")
print(test_data)

toon_str = ToonEncoderUltraOptimized.encode(test_data)
print(f"\nCodificado en TOON (Ultra):")
print(repr(toon_str))

decoded = ToonParserUltraOptimized.parse(toon_str)
print(f"\nDecodificado (Ultra):")
print(decoded)

print(f"\n¿Son iguales? {test_data == decoded}")
print(f"¿Tiene el mismo tipo de 'values'? {type(test_data.get('values')) == type(decoded.get('values'))}")
print(f"¿values es un array en origen? {isinstance(test_data.get('values'), list)}")
print(f"¿values es un array en destino? {isinstance(decoded.get('values'), list)}")
print(f"Contenido de 'values' en origen: {test_data.get('values')}")
print(f"Contenido de 'values' en destino: {decoded.get('values')}")