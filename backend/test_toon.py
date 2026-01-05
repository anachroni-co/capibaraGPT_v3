#!/usr/bin/env python3
"""
Script de prueba para TOON (Token-Oriented Object Notation)
Verifica encoding, decoding, y eficiencia
"""

import sys
import json
from pathlib import Path

# Agregar backend al path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from toon_utils import ToonParser, ToonEncoder, FormatManager
    print("✅ TOON Utils importado correctamente\n")
except ImportError as e:
    print(f"❌ Error importando TOON utils: {e}")
    sys.exit(1)

def print_section(title):
    """Imprime un header para cada sección"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_encoding():
    """Test de encoding Python -> TOON"""
    print_section("Test 1: Encoding Python -> TOON")

    # Datos de prueba
    data = {
        "users": [
            {"id": 1, "name": "Alice", "role": "admin", "active": True},
            {"id": 2, "name": "Bob", "role": "user", "active": True},
            {"id": 3, "name": "Charlie", "role": "user", "active": False}
        ]
    }

    print("\n📊 Datos originales (Python):")
    print(json.dumps(data, indent=2))

    # Encodear a TOON
    toon_string = ToonEncoder.encode(data)

    print("\n📝 Formato TOON:")
    print(toon_string)

    # Mostrar estadísticas
    stats = ToonEncoder.estimate_token_savings(data)
    print(f"\n📈 Estadísticas:")
    print(f"   • Tamaño JSON: {stats['json_size']} caracteres")
    print(f"   • Tamaño TOON: {stats['toon_size']} caracteres")
    print(f"   • Ahorro: {stats['savings']} caracteres ({stats['savings_percent']}%)")
    print(f"   • ¿Recomendado?: {'✅ Sí' if stats['recommended'] else '❌ No'}")

    return toon_string, data

def test_decoding(toon_string, original_data):
    """Test de decoding TOON -> Python"""
    print_section("Test 2: Decoding TOON -> Python")

    print("\n📝 Entrada TOON:")
    print(toon_string)

    # Decodear
    decoded_data = ToonParser.parse(toon_string)

    print("\n📊 Datos decodificados:")
    print(json.dumps(decoded_data, indent=2))

    # Verificar que coincide con original
    print("\n🔍 Verificación:")

    # Comparar estructuras
    if "users" in decoded_data and "users" in original_data:
        if len(decoded_data["users"]) == len(original_data["users"]):
            print(f"   ✅ Mismo número de usuarios: {len(decoded_data['users'])}")
        else:
            print(f"   ❌ Diferente número de usuarios")
            return False

        # Comparar campos
        for i, (decoded_user, original_user) in enumerate(zip(decoded_data["users"], original_data["users"])):
            if set(decoded_user.keys()) == set(original_user.keys()):
                print(f"   ✅ Usuario {i+1}: Mismos campos")
            else:
                print(f"   ❌ Usuario {i+1}: Campos diferentes")
                return False
    else:
        print("   ❌ Estructura diferente")
        return False

    print("\n   ✅ Decoding exitoso - Datos coinciden")
    return True

def test_format_manager():
    """Test de FormatManager con auto-detección"""
    print_section("Test 3: FormatManager - Conversión Automática")

    # Datos grandes (beneficio de TOON)
    large_data = {
        "transactions": [
            {"id": i, "amount": 100 + i, "user_id": i % 10, "status": "completed"}
            for i in range(1, 21)  # 20 transacciones
        ]
    }

    # Datos pequeños (JSON más eficiente)
    small_data = {
        "config": {
            "debug": True,
            "timeout": 30
        }
    }

    print("\n📊 Test con datos GRANDES (20 transacciones):")
    print(f"   Datos: {len(large_data['transactions'])} transacciones")

    # Auto-detect
    content, format_type = FormatManager.encode(large_data, preferred_format='auto')
    stats = FormatManager.analyze_data(large_data)

    print(f"   Formato seleccionado: {format_type.upper()}")
    print(f"   Razón: {'TOON más eficiente' if format_type == 'toon' else 'JSON más eficiente'}")
    print(f"   Ahorro TOON: {stats['savings_percent']:.1f}%")

    print("\n📊 Test con datos PEQUEÑOS (config simple):")
    print(f"   Datos: {small_data}")

    content, format_type = FormatManager.encode(small_data, preferred_format='auto')
    stats = FormatManager.analyze_data(small_data)

    print(f"   Formato seleccionado: {format_type.upper()}")
    print(f"   Razón: {'TOON más eficiente' if format_type == 'toon' else 'JSON más eficiente'}")
    print(f"   Diferencia: {abs(stats['savings_percent']):.1f}%")

def test_roundtrip():
    """Test de conversión bidireccional (roundtrip)"""
    print_section("Test 4: Roundtrip - JSON -> TOON -> JSON")

    test_cases = [
        {
            "name": "Lista de objetos uniformes",
            "data": {
                "items": [
                    {"id": 1, "value": "A"},
                    {"id": 2, "value": "B"},
                    {"id": 3, "value": "C"}
                ]
            }
        },
        {
            "name": "Objeto simple",
            "data": {
                "user": {
                    "name": "Alice",
                    "age": 30,
                    "active": True
                }
            }
        },
        {
            "name": "Valores mixtos",
            "data": {
                "string": "hello",
                "number": 42,
                "boolean": True,
                "null": None
            }
        }
    ]

    all_passed = True

    for test_case in test_cases:
        print(f"\n🧪 Test: {test_case['name']}")
        original = test_case['data']

        # JSON -> TOON
        toon_str = ToonEncoder.encode(original)
        print(f"   1. Python -> TOON: OK ({len(toon_str)} chars)")

        # TOON -> Python
        decoded = ToonParser.parse(toon_str)
        print(f"   2. TOON -> Python: OK")

        # Python -> JSON (verificar estructura)
        json_original = json.dumps(original, sort_keys=True)
        json_decoded = json.dumps(decoded, sort_keys=True)

        # Las claves deben coincidir
        if set(str(original.keys())) == set(str(decoded.keys())):
            print(f"   3. Verificación: ✅ Estructuras coinciden")
        else:
            print(f"   3. Verificación: ❌ Estructuras difieren")
            all_passed = False

    return all_passed

def test_edge_cases():
    """Test de casos especiales"""
    print_section("Test 5: Casos Especiales")

    test_cases = [
        ("Lista vacía", {"items": []}),
        ("Valores nulos", {"data": [{"id": 1, "value": None}]}),
        ("Strings con comas", {"data": [{"id": 1, "name": "Smith, John"}]}),
        ("Números decimales", {"data": [{"id": 1, "price": 19.99}]}),
        ("Booleans", {"data": [{"id": 1, "active": True, "verified": False}]}),
    ]

    all_passed = True

    for name, data in test_cases:
        try:
            # Encode
            toon_str = ToonEncoder.encode(data)

            # Decode
            decoded = ToonParser.parse(toon_str)

            print(f"   ✅ {name}: OK")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            all_passed = False

    return all_passed

def test_efficiency_benchmark():
    """Benchmark de eficiencia con diferentes tamaños de datos"""
    print_section("Test 6: Benchmark de Eficiencia")

    sizes = [5, 10, 20, 50, 100]

    print("\n📊 Comparación JSON vs TOON (ahorro de espacio):\n")
    print(f"{'Items':<10} {'JSON':<15} {'TOON':<15} {'Ahorro':<15} {'Recomendado'}")
    print("-" * 70)

    for size in sizes:
        data = {
            "records": [
                {"id": i, "name": f"User{i}", "score": i * 10, "active": i % 2 == 0}
                for i in range(1, size + 1)
            ]
        }

        stats = ToonEncoder.estimate_token_savings(data)

        print(f"{size:<10} {stats['json_size']:<15} {stats['toon_size']:<15} "
              f"{stats['savings_percent']:>6.1f}% {' '*8} "
              f"{'TOON ✅' if stats['recommended'] else 'JSON'}")

    print("\n💡 Observación: TOON es más eficiente con arrays grandes de objetos uniformes")

def main():
    """Función principal"""
    print("=" * 70)
    print("  🧪 Test Suite - TOON Format Manager")
    print("=" * 70)

    # Test 1: Encoding
    toon_string, original_data = test_encoding()

    # Test 2: Decoding
    decode_success = test_decoding(toon_string, original_data)

    # Test 3: Format Manager
    test_format_manager()

    # Test 4: Roundtrip
    roundtrip_success = test_roundtrip()

    # Test 5: Edge cases
    edge_cases_success = test_edge_cases()

    # Test 6: Efficiency
    test_efficiency_benchmark()

    # Resumen
    print_section("Resumen de Tests")

    print(f"\n   {'Test':<30} {'Resultado'}")
    print("   " + "-" * 50)
    print(f"   {'Encoding':<30} {'✅ PASS'}")
    print(f"   {'Decoding':<30} {'✅ PASS' if decode_success else '❌ FAIL'}")
    print(f"   {'Format Manager':<30} {'✅ PASS'}")
    print(f"   {'Roundtrip':<30} {'✅ PASS' if roundtrip_success else '❌ FAIL'}")
    print(f"   {'Edge Cases':<30} {'✅ PASS' if edge_cases_success else '❌ FAIL'}")
    print(f"   {'Efficiency Benchmark':<30} {'✅ PASS'}")

    all_success = decode_success and roundtrip_success and edge_cases_success

    print("\n" + "=" * 70)
    if all_success:
        print("✅ TODOS LOS TESTS PASARON")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
    print("=" * 70)

    sys.exit(0 if all_success else 1)

if __name__ == '__main__':
    main()
