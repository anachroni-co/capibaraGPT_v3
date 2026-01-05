#!/usr/bin/env python3
"""
Script de prueba para Semantic Router
Prueba diferentes tipos de queries y muestra qué modelo se selecciona
"""
import sys
import json
from pathlib import Path

# Agregar backend al path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from semantic_model_router import get_router
    print("✅ Semantic Router importado correctamente\n")
except ImportError as e:
    print(f"❌ Error importando semantic_model_router: {e}")
    print("   Asegúrate de haber instalado: pip install semantic-router")
    sys.exit(1)

# Queries de prueba por categoría
TEST_QUERIES = {
    "Programming": [
        "cómo crear una función en Python",
        "ayúdame con este error de JavaScript",
        "explica qué es un array en programación",
        "cómo hacer un loop for en Java",
        "debug este código HTML",
    ],
    "Creative Writing": [
        "escribe un cuento sobre un robot que descubre emociones",
        "crea un poema sobre el mar",
        "redacta una historia de terror",
        "inventa un personaje para una novela de fantasía",
        "escribe un artículo sobre viajes espaciales",
    ],
    "Quick Facts": [
        "qué es Python",
        "cuántos habitantes tiene Madrid",
        "quién descubrió América",
        "define inteligencia artificial",
        "en qué año comenzó la Segunda Guerra Mundial",
    ],
    "Analysis": [
        "analiza las diferencias entre React y Vue",
        "compara los sistemas operativos Windows y Linux",
        "evalúa las ventajas del trabajo remoto",
        "explica en detalle cómo funciona la fotosíntesis",
        "cuáles son las implicaciones éticas de la IA",
    ],
    "Conversation": [
        "hola, cómo estás",
        "háblame de ti",
        "qué puedes hacer",
        "buenos días",
        "gracias por tu ayuda",
    ],
    "Math": [
        "resuelve 25 + 37",
        "calcula la raíz cuadrada de 144",
        "cuánto es 15 por 8",
        "resuelve esta ecuación: 2x + 5 = 15",
        "calcula el área de un círculo con radio 5",
    ],
    "Translation": [
        "traduce 'hola' al inglés",
        "cómo se dice 'gracias' en francés",
        "traduce esta frase al alemán: buenos días",
    ],
    "Mixed/Ambiguous": [
        "esto es una prueba general",
        "cuéntame algo interesante",
        "no sé qué preguntar",
    ]
}

def print_section_header(title):
    """Imprime un header bonito para cada sección"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_query_result(query, decision, index):
    """Imprime el resultado de una query de forma legible"""
    print(f"\n{index}. Query: \"{query}\"")
    print(f"   ├─ Ruta detectada: {decision['route_name']}")
    print(f"   ├─ Modelo: {decision['model_id']}")
    print(f"   ├─ Confianza: {decision['confidence']:.0%}")
    print(f"   ├─ Fallback: {'Sí' if decision['fallback'] else 'No'}")
    print(f"   └─ Razón: {decision['reasoning']}")

def test_single_query(router, query):
    """Prueba una query individual"""
    result = router.test_query(query)
    return result['decision']

def test_category(router, category_name, queries):
    """Prueba todas las queries de una categoría"""
    print_section_header(f"Categoría: {category_name}")

    results = []
    for idx, query in enumerate(queries, 1):
        decision = test_single_query(router, query)
        print_query_result(query, decision, idx)
        results.append({
            'query': query,
            'decision': decision
        })

    return results

def generate_statistics(all_results):
    """Genera estadísticas de todas las pruebas"""
    print_section_header("📊 Estadísticas Globales")

    total_queries = sum(len(results) for results in all_results.values())

    # Contar rutas detectadas
    route_counts = {}
    model_counts = {}
    fallback_count = 0

    for category, results in all_results.items():
        for result in results:
            decision = result['decision']
            route = decision['route_name']
            model = decision['model_id']

            route_counts[route] = route_counts.get(route, 0) + 1
            model_counts[model] = model_counts.get(model, 0) + 1

            if decision['fallback']:
                fallback_count += 1

    print(f"\n📝 Total de queries probadas: {total_queries}")
    print(f"🎯 Queries con ruta específica: {total_queries - fallback_count}")
    print(f"⚠️  Queries con fallback: {fallback_count}")

    print(f"\n🗺️  Distribución por Rutas:")
    for route, count in sorted(route_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_queries) * 100
        print(f"   • {route:<20} {count:>3} queries ({percentage:>5.1f}%)")

    print(f"\n🤖 Distribución por Modelos:")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_queries) * 100
        print(f"   • {model:<20} {count:>3} queries ({percentage:>5.1f}%)")

def interactive_mode(router):
    """Modo interactivo para probar queries manualmente"""
    print_section_header("🎮 Modo Interactivo")
    print("\nEscribe queries para probar el router.")
    print("Comandos especiales:")
    print("  • 'quit' o 'exit' - Salir")
    print("  • 'info' - Ver información del router")
    print("  • 'routes' - Ver todas las rutas")
    print()

    while True:
        try:
            query = input("Query > ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'salir']:
                print("👋 ¡Hasta luego!")
                break

            if query.lower() == 'info':
                routes = router.get_available_routes()
                models = router.get_model_mapping()
                print(f"\n📋 Información del Router:")
                print(f"   • Rutas disponibles: {len(routes)}")
                print(f"   • Modelos configurados: {len(models)}")
                continue

            if query.lower() == 'routes':
                print(f"\n🗺️  Rutas disponibles:")
                for route_name in router.get_available_routes():
                    route_info = router.get_route_info(route_name)
                    if route_info:
                        print(f"   • {route_name:<20} → {route_info['assigned_model']}")
                continue

            # Probar la query
            decision = test_single_query(router, query)
            print_query_result(query, decision, "→")

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Función principal"""
    print("=" * 70)
    print("  🧪 Test Suite - Semantic Router Capibara6")
    print("=" * 70)

    # Inicializar router
    print("\n🚀 Inicializando Semantic Router...")
    try:
        router = get_router()
        print("✅ Router inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando router: {e}")
        sys.exit(1)

    # Mostrar información del router
    routes = router.get_available_routes()
    models = router.get_model_mapping()

    print(f"\n📋 Configuración:")
    print(f"   • Rutas: {len(routes)}")
    print(f"   • Modelos: {len(models)}")
    print(f"   • Encoder: FastEmbed (local)")

    # Modo de ejecución
    import argparse
    parser = argparse.ArgumentParser(description='Test Semantic Router')
    parser.add_argument('--interactive', '-i', action='store_true',
                      help='Modo interactivo')
    parser.add_argument('--category', '-c', type=str,
                      help='Probar solo una categoría específica')
    parser.add_argument('--query', '-q', type=str,
                      help='Probar una query específica')

    args = parser.parse_args()

    # Query individual
    if args.query:
        print_section_header("Query Individual")
        decision = test_single_query(router, args.query)
        print_query_result(args.query, decision, "→")
        return

    # Modo interactivo
    if args.interactive:
        interactive_mode(router)
        return

    # Test completo o por categoría
    all_results = {}

    categories_to_test = TEST_QUERIES.keys()
    if args.category:
        if args.category in TEST_QUERIES:
            categories_to_test = [args.category]
        else:
            print(f"❌ Categoría '{args.category}' no encontrada")
            print(f"   Categorías disponibles: {', '.join(TEST_QUERIES.keys())}")
            return

    # Ejecutar tests
    for category_name in categories_to_test:
        queries = TEST_QUERIES[category_name]
        results = test_category(router, category_name, queries)
        all_results[category_name] = results

    # Mostrar estadísticas
    if len(all_results) > 1:
        generate_statistics(all_results)

    print("\n" + "=" * 70)
    print("✅ Tests completados")
    print("=" * 70)
    print("\n💡 Tip: Usa --interactive para modo interactivo")
    print("   Ejemplo: python test_semantic_router.py --interactive")

if __name__ == '__main__':
    main()
