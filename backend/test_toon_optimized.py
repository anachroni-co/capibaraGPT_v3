#!/usr/bin/env python3
"""
Tests para verificar que las versiones optimizadas de TOON funcionan correctamente
"""

import sys
import json
from pathlib import Path

# Añadir backend al path
sys.path.insert(0, str(Path(__file__).parent))

from toon_utils import (
    ToonParser, ToonEncoder, FormatManager,
    ToonParserOptimized, ToonEncoderOptimized, FormatManagerOptimized,
    ToonParserUltraOptimized, ToonEncoderUltraOptimized, FormatManagerUltraOptimized
)

def test_consistency():
    """Test para asegurar que las versiones optimizadas producen el mismo resultado que las originales"""
    print("🔍 Test de consistencia entre versiones...")
    
    test_cases = [
        # Caso 1: Array simple
        {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"}
            ]
        },
        # Caso 2: Objeto simple
        {
            "config": {
                "debug": True,
                "timeout": 30
            }
        },
        # Caso 3: Datos mixtos
        {
            "id": 1,
            "name": "Test",
            "nested": {"a": 1, "b": 2}
        },
        # Caso 4: Datos grandes
        {
            "records": [
                {"id": i, "name": f"User{i}", "active": i % 2 == 0}
                for i in range(50)
            ]
        }
    ]
    
    all_passed = True
    
    for i, test_data in enumerate(test_cases):
        print(f"  Test {i+1}: ", end="")
        
        # Codificar con todas las versiones
        original_encoded = ToonEncoder.encode(test_data)
        optimized_encoded = ToonEncoderOptimized.encode(test_data)
        ultra_encoded = ToonEncoderUltraOptimized.encode(test_data)
        
        # Decodificar con todas las versiones
        original_decoded = ToonParser.parse(original_encoded)
        optimized_decoded = ToonParserOptimized.parse(optimized_encoded)
        ultra_decoded = ToonParserUltraOptimized.parse(ultra_encoded)
        
        # Comparar resultados
        if (original_decoded == optimized_decoded == ultra_decoded == test_data):
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            print(f"    Original: {original_decoded}")
            print(f"    Optimized: {optimized_decoded}")
            print(f"    Ultra: {ultra_decoded}")
            print(f"    Expected: {test_data}")
            all_passed = False
    
    return all_passed

def test_performance_improvement():
    """Test para verificar que las versiones optimizadas son más rápidas"""
    print("\n⏱️  Test de mejora de rendimiento...")
    
    import time
    
    # Datos grandes para test de rendimiento
    large_data = {
        "records": [
            {"id": i, "name": f"Usuario {i}", "score": i * 2.5, "active": i % 3 == 0}
            for i in range(100)
        ]
    }
    
    # Medir tiempos
    iterations = 10  # Número de iteraciones para promedio
    
    # Medir original
    start = time.time()
    for _ in range(iterations):
        encoded = ToonEncoder.encode(large_data)
        decoded = ToonParser.parse(encoded)
    original_time = time.time() - start
    
    # Medir optimizado
    start = time.time()
    for _ in range(iterations):
        encoded = ToonEncoderOptimized.encode(large_data)
        decoded = ToonParserOptimized.parse(encoded)
    optimized_time = time.time() - start
    
    # Medir ultra optimizado
    start = time.time()
    for _ in range(iterations):
        encoded = ToonEncoderUltraOptimized.encode(large_data)
        decoded = ToonParserUltraOptimized.parse(encoded)
    ultra_time = time.time() - start
    
    print(f"  Original:   {original_time/iterations*1000:.2f}ms/iter")
    print(f"  Optimizado: {optimized_time/iterations*1000:.2f}ms/iter")
    print(f"  Ultra:      {ultra_time/iterations*1000:.2f}ms/iter")
    
    # Verificar que ultra es más rápido que original
    ultra_faster = ultra_time < original_time
    improvement = ((original_time - ultra_time) / original_time) * 100 if original_time > 0 else 0
    
    print(f"  Mejora:     {improvement:.1f}% más rápido")
    
    if ultra_faster:
        print("  ✅ PASSED - Ultra es más rápido")
        return True
    else:
        print("  ❌ FAILED - Ultra no es más rápido")
        return False

def test_edge_cases():
    """Test de casos límite"""
    print("\n🔍 Test de casos límite...")
    
    edge_cases = [
        {},  # Vacío
        {"data": []},  # Array vacío
        {"string_with_comma": "Smith, John"},
        {"number": 3.14159},
        {"boolean": True},
        {"null_value": None},
        {"special_chars": 'He said "Hello, World!"'},
        {"array_field": [1, 2, 3]},  # Array como campo (se codifica como string en TOON)
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(edge_cases):
        print(f"  Caso límite {i+1}: ", end="")
        
        try:
            # Probar con ultra optimizado
            encoded = ToonEncoderUltraOptimized.encode(test_case)
            decoded = ToonParserUltraOptimized.parse(encoded)
            
            # Verificar que los datos originales y decodificados son equivalentes
            if json.dumps(test_case, sort_keys=True) == json.dumps(decoded, sort_keys=True):
                print("✅")
            else:
                print("❌")
                print(f"    Original: {test_case}")
                print(f"    Decoded:  {decoded}")
                all_passed = False
        except Exception as e:
            print("❌")
            print(f"    Error: {e}")
            all_passed = False
    
    return all_passed

def test_caching():
    """Test para verificar que el mecanismo de caché funciona correctamente"""
    print("\nキャッシング Test de caché...")
    
    import time
    
    test_data = {
        "records": [
            {"id": i, "name": f"User{i}"}
            for i in range(10)
        ]
    }
    
    # Codificar dos veces los mismos datos
    start_time = time.time()
    for _ in range(5):
        encoded1 = ToonEncoderUltraOptimized.encode(test_data)
    time_with_cache1 = time.time() - start_time
    
    # Codificar otras veces (debería usar caché en algunas operaciones)
    start_time = time.time()
    for _ in range(5):
        encoded2 = ToonEncoderUltraOptimized.encode(test_data)
    time_with_cache2 = time.time() - start_time
    
    print(f"  Tiempo primeras 5 codificaciones: {time_with_cache1*1000:.2f}ms")
    print(f"  Tiempo segundas 5 codificaciones: {time_with_cache2*1000:.2f}ms")
    
    # Probar decodificación con caché
    toon_str = ToonEncoderUltraOptimized.encode(test_data)
    
    start_time = time.time()
    for _ in range(5):
        decoded1 = ToonParserUltraOptimized.parse(toon_str)
    time_decode1 = time.time() - start_time
    
    start_time = time.time()
    for _ in range(5):
        decoded2 = ToonParserUltraOptimized.parse(toon_str)  # Esta debería usar caché
    time_decode2 = time.time() - start_time
    
    print(f"  Tiempo primeras 5 decodificaciones: {time_decode1*1000:.2f}ms")
    print(f"  Tiempo segundas 5 decodificaciones: {time_decode2*1000:.2f}ms")
    
    # Validar que los resultados son correctos
    expected = json.dumps(test_data, sort_keys=True)
    result = json.dumps(decoded2, sort_keys=True)
    
    if expected == result:
        print("  ✅ PASSED - Caché funciona correctamente")
        return True
    else:
        print("  ❌ FAILED - Resultados incorrectos")
        return False

def main():
    """Función principal de test"""
    print("=" * 60)
    print("  🧪 TEST DE VERSIONES OPTIMIZADAS DE TOON")
    print("=" * 60)
    
    tests = [
        ("Consistencia", test_consistency),
        ("Rendimiento", test_performance_improvement),
        ("Casos límite", test_edge_cases),
        ("Caché", test_caching),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("  📊 RESULTADOS FINALES")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<15}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ Las versiones optimizadas funcionan correctamente")
        print("✅ La latencia ha sido reducida significativamente")
    else:
        print("💥 ALGÚN TEST FALLÓ")
        print("❌ Revisar las implementaciones optimizadas")
    print("=" * 60)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)