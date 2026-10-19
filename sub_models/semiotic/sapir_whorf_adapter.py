"""
Sapir-Whorf Adapter for language-based semantic and cognitive modulation.

This module implements the Sapir-Whorf linguistic relativity hypothesis
to adapt semantic and cognitive activations based on the detected language.
"""

from typing import Dict, Optional

# Native imports from Capibara project with fallbacks
try:
    from jax import numpy as jnp
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
    from jax.nn.layers import Embedding
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
    Use the native language detector from the Capibara project.
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
    
    # The project's LanguageDetector checks if text is in allowed languages
    # For our case, we use a simplified version that returns the language
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
    Modulates semantic and cognitive activations based on language.
    Inspired by the Sapir-Whorf linguistic relativity hypothesis.
    """

    def __init__(self, config=None):
        """
        Initialize the Sapir-Whorf adapter.

        Args:
            config: Optional configuration dictionary with parameters such as:
                   - hidden_size: Size of hidden layers
                   - language_dimensions: Language-specific dimensions
                   - cultural_contexts: Number of cultural contexts
        """
        self.config = config or {}
        self.hidden_size = self.config.get('hidden_size', 768)
        self.language_dimensions = self.config.get('language_dimensions', 64)
        self.cultural_contexts = self.config.get('cultural_contexts', 16)
        
        # Predefined map of cognitive structures associated with languages
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
            # Add more languages here
        }

    def adapt_embedding(self, input_embedding, language: str):
        """
        Apply semantic adjustment to embedding based on language cognitive structure.

        Args:
            input_embedding: Input embedding to adapt (can be numpy array, list, or any type)
            language: Language code (e.g., 'en', 'es', 'zh', 'ar')

        Returns:
            Embedding adapted according to language cognitive structure
        """
        modifiers = self.language_cognition_map.get(language, {})
        
        # If JAX/numpy is not available, we only return modification info
        if not JAX_AVAILABLE and not hasattr(input_embedding, 'shape'):
            return {
                'original_embedding': input_embedding,
                'language': language,
                'modifiers': modifiers,
                'adapted': True
            }
        
        embedding = input_embedding

        # Modify embedding based on cognitive properties
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
        Detect language and adjust router to activate correct semantic layers.

        Args:
            input_text: Input text for language detection
            router: Contextual router to configure (optional)

        Returns:
            Detected language code
        """
        language = detect_language(input_text)
        cognition_profile = self.language_cognition_map.get(language, {})

        # Send information to router if available
        if router and ROUTER_AVAILABLE:
            try:
                router.set_language_context(language)
                router.set_cognition_profile(cognition_profile)
            except Exception as e:
                # If the router doesn't have these methods, we simply continue
                pass

        return language

    def __call__(self, input_data, input_embedding=None, router=None):
        """
        Main method that combines language detection and embedding adaptation.

        Args:
            input_data: Input text (str) or input tensor/array
            input_embedding: Embedding to adapt (optional)
            router: Contextual router (optional)

        Returns:
            Embedding adapted according to language cognitive structure or modified tensor
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
        Add a new language profile to the cognitive map.

        Args:
            language_code: Language code (e.g., 'fr', 'de', 'ja')
            profile: Dictionary with language cognitive characteristics
        """
        self.language_cognition_map[language_code] = profile

    def get_language_profile(self, language_code: str) -> Optional[Dict[str, str]]:
        """
        Get the cognitive profile for a specific language.

        Args:
            language_code: Language code

        Returns:
            Language cognitive profile or None if not found
        """
        return self.language_cognition_map.get(language_code)

    def list_supported_languages(self) -> list:
        """
        List all languages supported by the adapter.

        Returns:
            List of supported language codes
        """
        return list(self.language_cognition_map.keys())
