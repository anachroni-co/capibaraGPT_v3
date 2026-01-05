#!/usr/bin/env python3
"""
Semantic Router para selección automática de modelos en Capibara6
Usa semantic-router para clasificar consultas y elegir el modelo óptimo
"""
from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.routers import SemanticRouter
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CapibaraModelRouter:
    """Router semántico para seleccionar modelos automáticamente"""

    def __init__(self):
        """Inicializa el router con rutas y encoder"""
        logger.info("🎯 Inicializando Semantic Router...")

        # Definir rutas semánticas para diferentes tipos de consultas
        self.routes = [
            Route(
                name="programming",
                utterances=[
                    "cómo programar en Python",
                    "ayúdame con este código JavaScript",
                    "debug este error de sintaxis",
                    "escribe una función que calcule",
                    "explica este algoritmo de ordenamiento",
                    "qué hace este código",
                    "cómo crear una clase en Java",
                    "error en mi código HTML",
                    "implementa un bucle for",
                    "qué es una API REST",
                    "cómo usar git",
                    "explica la recursividad",
                ]
            ),
            Route(
                name="creative_writing",
                utterances=[
                    "escribe un cuento sobre dragones",
                    "crea un poema romántico",
                    "redacta una historia de ciencia ficción",
                    "inventa un diálogo entre dos amigos",
                    "escribe una carta formal de presentación",
                    "genera un eslogan para mi empresa",
                    "crea una historia corta de terror",
                    "escribe un artículo sobre viajes",
                    "redacta un discurso motivacional",
                    "inventa un personaje para una novela",
                ]
            ),
            Route(
                name="quick_facts",
                utterances=[
                    "qué es Python",
                    "define inteligencia artificial",
                    "cuántos habitantes tiene Madrid",
                    "quién descubrió América",
                    "en qué año fue la Segunda Guerra Mundial",
                    "cuál es la capital de Francia",
                    "qué significa IA",
                    "quién inventó el teléfono",
                    "cuánto mide el Everest",
                    "qué es un átomo",
                    "define fotosíntesis",
                    "cuál es la velocidad de la luz",
                ]
            ),
            Route(
                name="analysis",
                utterances=[
                    "analiza las diferencias entre React y Vue",
                    "compara estos dos enfoques arquitectónicos",
                    "evalúa las ventajas de usar microservicios",
                    "explica en detalle el proceso de fotosíntesis",
                    "cuáles son las implicaciones de la IA en la sociedad",
                    "analiza las causas de la inflación",
                    "compara los sistemas operativos Linux y Windows",
                    "evalúa los pros y contras de trabajar remoto",
                    "explica detalladamente cómo funciona blockchain",
                    "analiza el impacto del cambio climático",
                ]
            ),
            Route(
                name="conversation",
                utterances=[
                    "hola cómo estás",
                    "qué tal el día",
                    "cuéntame algo interesante",
                    "háblame de ti",
                    "buenos días",
                    "cómo te llamas",
                    "qué puedes hacer",
                    "quién eres",
                    "me siento triste hoy",
                    "gracias por tu ayuda",
                    "hasta luego",
                    "cuál es tu color favorito",
                ]
            ),
            Route(
                name="math",
                utterances=[
                    "resuelve esta ecuación",
                    "calcula la raíz cuadrada de 144",
                    "cuánto es 25 por 4",
                    "deriva esta función",
                    "integra x al cuadrado",
                    "resuelve este problema de geometría",
                    "calcula el área de un círculo",
                    "explica el teorema de Pitágoras",
                    "resuelve este sistema de ecuaciones",
                    "calcula la probabilidad",
                ]
            ),
            Route(
                name="translation",
                utterances=[
                    "traduce esto al inglés",
                    "cómo se dice hola en francés",
                    "traduce esta frase al alemán",
                    "qué significa hello en español",
                    "traduce este texto al italiano",
                    "cómo se escribe gracias en japonés",
                ]
            )
        ]

        # Mapeo de rutas a modelos (Solo modelos activos en Backend BB)
        # Modelos disponibles: mixtral, phi, gpt-oss-20b
        self.model_mapping = {
            "programming": "gpt-oss-20b",      # Modelo grande para código complejo
            "creative_writing": "mixtral",      # Excelente para creatividad
            "quick_facts": "phi",               # Modelo pequeño y rápido (phi-mini)
            "analysis": "gpt-oss-20b",          # Usar gpt-oss-20b para análisis
            "conversation": "phi",              # Phi para conversación rápida
            "math": "gpt-oss-20b",              # Bueno para matemáticas
            "translation": "mixtral",           # Multilingüe
            "default": "gpt-oss-20b"            # Fallback a modelo más versátil
        }

        try:
            # Usar encoder local (sin API keys necesarias)
            logger.info("📦 Cargando FastEmbed encoder...")
            self.encoder = FastEmbedEncoder(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Crear router semántico
            logger.info("🔧 Creando Semantic Router...")
            self.router = SemanticRouter(
                encoder=self.encoder,
                routes=self.routes,
                auto_sync="local"  # Mantener todo local
            )

            logger.info("✅ Semantic Router inicializado correctamente")
            logger.info(f"   📋 Rutas disponibles: {len(self.routes)}")
            logger.info(f"   🤖 Modelos configurados: {len(self.model_mapping)}")

        except Exception as e:
            logger.error(f"❌ Error inicializando Semantic Router: {e}")
            raise

    def select_model(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Selecciona el modelo óptimo basado en la consulta del usuario

        Args:
            user_query: Consulta del usuario
            context: Contexto adicional (opcional)

        Returns:
            {
                'model_id': str,           # ID del modelo seleccionado
                'route_name': str,         # Nombre de la ruta detectada
                'confidence': float,       # Confianza en la decisión (0-1)
                'reasoning': str,          # Explicación de la decisión
                'fallback': bool           # True si se usó modelo por defecto
            }
        """
        try:
            logger.debug(f"🔍 Analizando query: {user_query[:100]}...")

            # Obtener ruta semántica
            route = self.router(user_query)

            if route and route.name:
                # Ruta encontrada
                model_id = self.model_mapping.get(route.name, self.model_mapping["default"])

                result = {
                    'model_id': model_id,
                    'route_name': route.name,
                    'confidence': 0.9,  # Alta confianza cuando hay match
                    'reasoning': f"Query clasificada como '{route.name}' → usando {model_id}",
                    'fallback': False
                }

                logger.info(f"✅ Ruta detectada: {route.name} → Modelo: {model_id}")

            else:
                # Sin ruta clara, usar modelo por defecto
                model_id = self.model_mapping["default"]

                result = {
                    'model_id': model_id,
                    'route_name': 'default',
                    'confidence': 0.5,  # Baja confianza, usando fallback
                    'reasoning': f"No se encontró ruta específica → usando modelo por defecto ({model_id})",
                    'fallback': True
                }

                logger.info(f"⚠️ Sin ruta específica → usando modelo por defecto: {model_id}")

            return result

        except Exception as e:
            logger.error(f"❌ Error en select_model: {e}")
            # En caso de error, retornar modelo por defecto
            return {
                'model_id': self.model_mapping["default"],
                'route_name': 'error',
                'confidence': 0.0,
                'reasoning': f"Error en routing: {str(e)} → usando modelo por defecto",
                'fallback': True
            }

    def get_available_routes(self) -> list:
        """Retorna los nombres de las rutas disponibles"""
        return [route.name for route in self.routes]

    def get_model_mapping(self) -> Dict[str, str]:
        """Retorna el mapeo completo de rutas a modelos"""
        return self.model_mapping.copy()

    def get_route_info(self, route_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de una ruta específica"""
        for route in self.routes:
            if route.name == route_name:
                return {
                    'name': route.name,
                    'utterances_count': len(route.utterances),
                    'examples': route.utterances[:3],  # Primeros 3 ejemplos
                    'assigned_model': self.model_mapping.get(route.name, 'unknown')
                }
        return None

    def test_query(self, query: str) -> Dict[str, Any]:
        """
        Prueba una query sin hacer request al modelo
        Útil para testing y debugging
        """
        decision = self.select_model(query)
        route_info = self.get_route_info(decision['route_name'])

        return {
            'query': query,
            'decision': decision,
            'route_details': route_info,
            'all_routes': self.get_available_routes()
        }


# ============================================
# INSTANCIA SINGLETON
# ============================================

_router_instance: Optional[CapibaraModelRouter] = None

def get_router() -> CapibaraModelRouter:
    """
    Obtiene la instancia singleton del router
    Lazy initialization para cargar solo cuando se necesita
    """
    global _router_instance

    if _router_instance is None:
        logger.info("🚀 Inicializando Semantic Router por primera vez...")
        _router_instance = CapibaraModelRouter()

    return _router_instance

def reset_router():
    """Reinicia el router (útil para testing)"""
    global _router_instance
    _router_instance = None
    logger.info("🔄 Router reiniciado")


# ============================================
# TESTING
# ============================================

if __name__ == '__main__':
    # Configurar logging para testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("🧪 Testing Semantic Router")
    print("=" * 60)

    # Inicializar router
    router = get_router()

    # Queries de prueba
    test_queries = [
        "cómo programar en Python",
        "escribe un cuento sobre un viaje espacial",
        "qué es la fotosíntesis",
        "analiza las diferencias entre React y Angular",
        "hola, cómo estás hoy",
        "resuelve 25 + 37",
        "traduce esto al inglés: buenos días",
        "esto no debería matchear con nada específico"
    ]

    print("\n📋 Rutas disponibles (Backend BB):")
    for route in router.get_available_routes():
        model = router.model_mapping.get(route, 'unknown')
        print(f"  • {route:<20} → {model}")

    print("\n🤖 Modelos activos:")
    print(f"  • gpt-oss-20b  - Programación, Matemáticas, Análisis")
    print(f"  • mixtral      - Creatividad, Traducción")
    print(f"  • phi          - Facts rápidos, Conversación")

    print("\n" + "=" * 60)
    print("🔍 Probando queries...")
    print("=" * 60)

    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        result = router.test_query(query)
        decision = result['decision']

        print(f"   ✓ Ruta: {decision['route_name']}")
        print(f"   ✓ Modelo: {decision['model_id']}")
        print(f"   ✓ Confianza: {decision['confidence']:.1%}")
        print(f"   ✓ Razón: {decision['reasoning']}")
        if decision['fallback']:
            print(f"   ⚠️ Usando fallback")

    print("\n" + "=" * 60)
    print("✅ Testing completado")
    print("=" * 60)
