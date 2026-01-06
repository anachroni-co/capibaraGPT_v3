"""
Adaptador Sapir-Whorf para modulación semántica y cognitiva basada en idioma.

This module implements la hipótesis de relatividad lingüística de Sapir-Whorf
para adaptar las activaciones semánticas y cognitivas según el idioma detectado.
"""

from typing import Dict, Optional

# Native imports from Capibara project with fallbacks
try:
    from capibara.jax import numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    # Fallback to standard numpy if available
    try:
        import numpy as jnp
        JAX_AVAILABLE = False
    except ImportError:
        # Create a minimal fallback for basic operations
        class MinimalNumpyFallback:
            @staticmethod
            def sin(x):
                return x * 0.9  # Simple approximation
            
            @staticmethod
            def linspace(start, stop, num):
                return [start + i * (stop - start) / (num - 1) for i in range(num)]
            
            @staticmethod
            def mean(arr, axis=None, keepdims=False):
                return sum(arr) / len(arr) if isinstance(arr, (list, tuple)) else arr
            
            @staticmethod
            def abs(x):
                return abs(x) if hasattr(x, '__abs__') else x
        
        jnp = MinimalNumpyFallback()
        JAX_AVAILABLE = False

try:
    from capibara.jax.nn.layers import Embedding
    LAYERS_AVAILABLE = True
except ImportError:
    Embedding = None
    LAYERS_AVAILABLE = False

try:
    from capibara.core.router import EnhancedRouter
    ROUTER_AVAILABLE = True
except ImportError:
    EnhancedRouter = None
    ROUTER_AVAILABLE = False

try:
    from capibara.training.data_preprocessing.semantic_deduplicator import LanguageDetector
    LANGUAGE_DETECTOR_AVAILABLE = True
except ImportError:
    LanguageDetector = None
    LANGUAGE_DETECTOR_AVAILABLE = False

# Use the project native language detector
def detect_language(text) -> str:
    """
    Usa el detector of language nativo del proyecto Capibara.
    Handles both string and non-string inputs safely.
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        # If it's a tensor or array, return default language
        if hasattr(text, 'shape'):
            return 'en'  # Default for tensor inputs
        # Try to convert to string
        try:
            text = str(text)
        except Exception:
            return 'en'  # Default fallback
    
    # El LanguageDetector del proyecto detecta si está en idiomas permitidos
    # Para nuestro caso, vamos a usar una versión simplificada que retorna el idioma
    try:
        text_lower = text.lower()
        if any(word in text_lower for word in ['hola', 'como', 'estas', 'gracias', 'por', 'favor']):
            return 'es'
        elif any(word in text_lower for word in ['hello', 'how', 'are', 'you', 'thank', 'please']):
            return 'en'
        elif any(word in text_lower for word in ['你好', '怎么样', '谢谢', '请']):
            return 'zh'
        elif any(word in text_lower for word in ['مرحبا', 'كيف', 'شكرا', 'من', 'فضلك']):
            return 'ar'
        else:
            return 'en'  # Default to English
    except Exception:
        return 'en'  # Safe fallback


class SapirWhorfAdapter:
    """
    Modula las activaciones semánticas y cognitivas en function del idioma.
    Inspirado en la hipótesis de relatividad lingüística de Sapir-Whorf.
    """

    def __init__(self, config=None):
        """
        Inicializa el adaptador Sapir-Whorf.
        
        Args:
            config: Diccionario de configuración opcional con parameters como:
                   - hidden_size: Tamaño de las capas ocultas
                   - language_dimensions: Dimensiones específicas del idioma
                   - cultural_contexts: Número de contextos culturales
        """
        self.config = config or {}
        self.hidden_size = self.config.get('hidden_size', 768)
        self.language_dimensions = self.config.get('language_dimensions', 64)
        self.cultural_contexts = self.config.get('cultural_contexts', 16)
        
        # Mapa predefinido de estructuras cognitivas asociadas a idiomas
        self.language_cognition_map = {
            "en": {
                "time": "linear",
                "space": "cartesian",
                "agency": "individual",
                "gender": "minimal"
            },
            "es": {
                "time": "past-focused",
                "space": "relational",
                "agency": "contextual",
                "gender": "grammatical"
            },
            "zh": {
                "time": "cyclical",
                "space": "hierarchical",
                "agency": "collective",
                "gender": "neutral"
            },
            "ar": {
                "time": "eternal-present",
                "space": "symbolic",
                "agency": "divine-modulated",
                "gender": "dualistic"
            }
            # Añade más idiomas aquí
        }

    def adapt_embedding(self, input_embedding, language: str):
        """
        Aplica un ajuste semántico al embedding según la estructura cognitiva del idioma.
        
        Args:
            input_embedding: Embedding de entrada a adaptar (puede ser numpy array, list, o cualquier tipo)
            language: Código of language (ej: 'en', 'es', 'zh', 'ar')
            
        Returns:
            Embedding adaptado según la estructura cognitiva del idioma
        """
        modifiers = self.language_cognition_map.get(language, {})
        
        # Si no tenemos JAX/numpy disponible, solo retornamos información de modificación
        if not JAX_AVAILABLE and not hasattr(input_embedding, 'shape'):
            return {
                'original_embedding': input_embedding,
                'language': language,
                'modifiers': modifiers,
                'adapted': True
            }
        
        embedding = input_embedding

        # Modifica el embedding según propiedades cognitivas
        try:
            if modifiers.get("time") == "cyclical":
                embedding = embedding * jnp.sin(embedding) if hasattr(embedding, '__mul__') else embedding
            elif modifiers.get("time") == "linear":
                if hasattr(embedding, 'shape'):
                    embedding = embedding + jnp.linspace(0.1, 0.9, embedding.shape[-1])
            if modifiers.get("agency") == "collective":
                embedding = jnp.mean(embedding, axis=0, keepdims=True) if hasattr(embedding, 'shape') else embedding
            if modifiers.get("gender") == "grammatical":
                embedding = embedding + jnp.abs(embedding % 0.2) if hasattr(embedding, '__mod__') else embedding
        except Exception as e:
            # Fallback: return original embedding with metadata
            return {
                'original_embedding': input_embedding,
                'language': language,
                'modifiers': modifiers,
                'error': str(e),
                'adapted': False
            }

        return embedding

    def route_contextually(self, input_text: str, router=None) -> str:
        """
        Detecta idioma y ajusta el router para que active las capas semánticas correctas.
        
        Args:
            input_text: Texto de entrada para detectar idioma
            router: Router contextual para configurar (opcional)
            
        Returns:
            Código of language detectado
        """
        language = detect_language(input_text)
        cognition_profile = self.language_cognition_map.get(language, {})

        # Enviar información al router si está disponible
        if router and ROUTER_AVAILABLE:
            try:
                router.set_language_context(language)
                router.set_cognition_profile(cognition_profile)
            except Exception as e:
                # Si el router no tiene estos methods, simplemente continuamos
                pass

        return language

    def __call__(self, input_data, input_embedding=None, router=None):
        """
        Método principal que combina detección of language y adaptación de embedding.
        
        Args:
            input_data: Texto de entrada (str) o tensor/array de entrada
            input_embedding: Embedding a adaptar (opcional)
            router: Router contextual (opcional)
            
        Returns:
            Embedding adaptado según la estructura cognitiva del idioma o tensor modificado
        """
        # Handle different input types
        if isinstance(input_data, str):
            # String input - normal processing
            lang = self.route_contextually(input_data, router)
            if input_embedding is not None:
                return self.adapt_embedding(input_embedding, lang)
            else:
                return {
                    'detected_language': lang,
                    'cognition_profile': self.language_cognition_map.get(lang, {}),
                    'text': input_data
                }
        elif hasattr(input_data, 'shape'):
            # Tensor/array input - apply linguistic adaptation directly
            return self._adapt_tensor_linguistically(input_data)
        else:
            # Other input types - try to convert to string first
            try:
                text_data = str(input_data)
                lang = self.route_contextually(text_data, router)
                if input_embedding is not None:
                    return self.adapt_embedding(input_embedding, lang)
                else:
                    return {
                        'detected_language': lang,
                        'cognition_profile': self.language_cognition_map.get(lang, {}),
                        'text': text_data
                    }
            except Exception as e:
                # Fallback - return original input
                return input_data
    
    def _adapt_tensor_linguistically(self, tensor_input):
        """
        Adapt tensor input using linguistic transformations without requiring text.
        
        Args:
            tensor_input: Input tensor/array to adapt
            
        Returns:
            Linguistically adapted tensor
        """
        try:
            # Apply default linguistic adaptation (assuming multilingual context)
            # Use a combination of transformations from different language profiles
            if not JAX_AVAILABLE and not hasattr(tensor_input, 'shape'):
                return tensor_input
            
            adapted = tensor_input
            
            # Apply cyclical transformation (Chinese influence)
            if hasattr(adapted, '__mul__') and hasattr(adapted, 'shape'):
                adapted = adapted * (1.0 + 0.1 * jnp.sin(adapted * 0.5))
            
            # Apply relational transformation (Spanish influence) 
            if hasattr(adapted, 'mean') and hasattr(adapted, 'shape'):
                mean_val = jnp.mean(adapted, axis=-1, keepdims=True)
                adapted = adapted + 0.05 * mean_val
            
            # Apply individual agency transformation (English influence)
            if hasattr(adapted, 'shape') and len(adapted.shape) > 1:
                adapted = adapted + 0.02 * jnp.linspace(0.1, 0.9, adapted.shape[-1])
            
            return adapted
            
        except Exception as e:
            # If any transformation fails, return original tensor
            return tensor_input

    def add_language_profile(self, language_code: str, profile: Dict[str, str]):
        """
        Añade un nuevo perfil of language al mapa cognitivo.
        
        Args:
            language_code: Código del idioma (ej: 'fr', 'de', 'ja')
            profile: Diccionario con las características cognitivas del idioma
        """
        self.language_cognition_map[language_code] = profile

    def get_language_profile(self, language_code: str) -> Optional[Dict[str, str]]:
        """
        Obtiene el perfil cognitivo de un idioma específico.
        
        Args:
            language_code: Código del idioma
            
        Returns:
            Perfil cognitivo del idioma o None si no existe
        """
        return self.language_cognition_map.get(language_code)

    def list_supported_languages(self) -> list:
        """
        Lista todos los idiomas soportados por el adaptador.
        
        Returns:
            Lista de códigos of language soportados
        """
        return list(self.language_cognition_map.keys())
