"""
Mnemosyne Semio Module - Semiotic and cultural analysis module.

This module provides specialized analysis for visual and literary art,
inspired by Mario Praz's approach. It automatically activates during
inference when art-related requests are detected.

Key Components:
    - MnemosyneSemioModule: Main class for semiotic analysis
    - Integration with SparseCapibara for sparse processing
    - MetaBAMDP integration for advanced meta-learning

Features:
    - Artwork and photography analysis
    - Symbolic interpretation of visual content
    - Literary corpus integration
    - Style and iconography recognition

Author: Skydesk International Dev Team.
"""

import os
import re
import logging
import numpy as np
from flax import linen as nn
from functools import lru_cache
import jax.numpy as jnp
from flax.training import train_state
try:
    from transformers import FlaxBlipProcessor
except ImportError:
    # Fallback if FlaxBlipProcessor is not available
    FlaxBlipProcessor = None
from typing import List, Optional, Dict, Union, Tuple, Any
from pathlib import Path
from capibara.layers.sparsity.sparse_capibara import SparseCapibara
from capibara.sub_models.experimental.meta_bamdp import MetaBAMDP, MetaBAMDPConfig

logger = logging.getLogger(__name__)

class MnemosyneSemioModule:
    """
    Specialized module for semiotic and cultural analysis of visual and literary art,
    inspired by Mario Praz's approach. This module is designed to automatically activate
    during inference in CapibaraGPT when requests related to art, photography, video,
    or descriptions of classical works are detected.
    """

    def __init__(self, use_sparse=False, use_meta_bamdp=False):
        self.activated: bool = False
        self.context_keywords: List[str] = [
            "artwork", "artistic analysis", "painting", "sculpture", "video art",
            "photography", "symbolic interpretation", "visual symbols", "classical art",
            "romanticism", "baroque", "renaissance", "iconography"
        ]
        # Will be loaded during training
        self.literary_corpus: Optional[Dict] = None
        # Visual vector index
        self.visual_reference_db: Optional[Dict] = None
        self.style_lookup: Dict[str, str] = {}
        self.symbol_lookup: Dict[str, List[str]] = {}
        self.parallel_matrix: Dict[str, List[str]] = {}
        self.use_sparse = use_sparse
        self.use_meta_bamdp = use_meta_bamdp

        if self.use_sparse:
            try:
                self.sparse_layer = SparseCapibara(
                    hidden_size=256, sparsity_type="neuronal", sparsity_target=0.7
                )
            except Exception as e:
                logger.warning(f"️ SparseCapibara not available: {e}")
                self.sparse_layer = None
        else:
            self.sparse_layer = None

        if self.use_meta_bamdp:
            try:
                config = MetaBAMDPConfig(hidden_size=256)
                self.meta_bamdp = MetaBAMDP(config=config)
            except Exception as e:
                logger.warning(f"️ MetaBAMDP not available: {e}")
                self.meta_bamdp = None
        else:
            self.meta_bamdp = None

        # Initialize BLIP processor if available
        self.processor = None
        self.model = None
        if FlaxBlipProcessor:
            try:
                self.processor = FlaxBlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                # Note: FlaxBlipForConditionalGeneration would need to be imported separately
                logger.info(" BLIP processor initialized successfully")
            except Exception as e:
                logger.warning(f"️ BLIP processor initialization failed: {e}")

    def activate_if_context_matches(self, input_text: str) -> Tuple[bool, float]:
        """Check if the input text matches artistic context and return activation status with confidence."""
        matches = 0
        total_keywords = len(self.context_keywords)
        
        for keyword in self.context_keywords:
            if keyword in input_text.lower():
                matches += 1
                
        confidence = matches / total_keywords if total_keywords > 0 else 0.0
        self.activated = confidence > 0.1  # Activate if at least 10% keywords match
        
        return self.activated, confidence

    def contextual_analysis(self, input_text: str, image_metadata: Optional[Dict] = None) -> Dict:
        """Perform contextual analysis of artistic content."""
        if not self.activated:
            return {"status": "inactive", "reason": "context not matched"}

        # Step 1: Era and style identification
        style, style_conf = self.identify_historical_style(input_text)

        # Step 2: Iconography and symbolism
        symbols, symbols_conf = self.extract_symbols(input_text, image_metadata)

        # Step 3: Cultural parallelism
        parallels = self.find_cross_modal_parallels(style)

        return {
            "status": "active",
            "confidence": {
                "overall": min(style_conf, symbols_conf),
                "style": style_conf,
                "symbols": symbols_conf,
                "image": image_metadata.get("confidence", 0.0) if image_metadata else 0.0
            },
            "style": {
                "name": style,
                "confidence": style_conf
            },
            "symbols": symbols,
            "parallels": parallels,
            "metadata": {
                "image_metadata": image_metadata if image_metadata else None,
            }
        }

    def identify_historical_style(self, text: str) -> Tuple[str, float]:
        """Identify historical art style from text with confidence score."""
        text_lower = text.lower()
        matches = []
        
        for name, style in self.style_lookup.items():
            if re.search(rf"\b{name}\b", text_lower):
                matches.append((style, len(name)))
        
        if matches:
            # Return the style with the longest matching keyword (more specific)
            best_match = max(matches, key=lambda x: x[1])
            confidence = min(1.0, best_match[1] / 10.0)  # Normalize confidence
            return best_match[0], confidence
        
        return "Unknown", 0.0

    def extract_symbols(self, text: str, image_metadata: Optional[Dict]) -> Tuple[List[str], float]:
        """Extract symbols from text and image metadata with confidence score."""
        results = []
        text_lower = text.lower()
        matches = 0
        total_possible = len(self.symbol_lookup)
        
        for keyword, symbols in self.symbol_lookup.items():
            if keyword in text_lower:
                results.extend(symbols)
                matches += 1

        if image_metadata and "caption" in image_metadata:
            caption_lower = image_metadata["caption"].lower()
            for keyword, symbols in self.symbol_lookup.items():
                if keyword in caption_lower:
                    results.extend(symbols)
                    matches += 1

        confidence = matches / max(1, total_possible)
        return list(set(results)), confidence

    def find_cross_modal_parallels(self, style: str) -> List[str]:
        """Find cross-modal parallels for a given style."""
        return self.parallel_matrix.get(style, [])

    def load_training_material(self, corpus: Dict, visual_db: Dict,
                               style_mapping: Dict[str, str],
                               symbol_mapping: Dict[str, List[str]],
                               parallel_mapping: Dict[str, List[str]]):
        """
        Load training material, including literary corpus, visual references,
        style mapping, symbolic mapping, and interdisciplinary parallelism index.
        """
        self.literary_corpus = corpus
        self.visual_reference_db = visual_db
        self.style_lookup = style_mapping
        self.symbol_lookup = symbol_mapping
        self.parallel_matrix = parallel_mapping
        logger.info(" Training material loaded for MnemosyneSemioModule")

    def process_features(self, features, training=False, rng=None):
        """Process features through sparse and meta-BAMDP layers if available."""
        if self.sparse_layer:
            try:
                features = self.sparse_layer(features, training=training, rng=rng)["output"]
            except Exception as e:
                logger.warning(f"️ Sparse layer processing failed: {e}")
                
        if self.meta_bamdp:
            try:
                features = self.meta_bamdp(features, training=training)
            except Exception as e:
                logger.warning(f"️ Meta-BAMDP processing failed: {e}")
                
        return features

    def to_jax_array(self, x):
        """
        Convert PyTorch or NumPy tensors to jnp.ndarray for JAX/Flax compatibility.
        """
        if isinstance(x, jnp.ndarray):
            return x
        elif isinstance(x, np.ndarray):
            return jnp.array(x)
        else:
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return jnp.array(x.cpu().numpy())
            except ImportError:
                pass
            
            # Try to convert as numpy array
            try:
                return jnp.array(x)
            except Exception as e:
                raise TypeError(f"Unsupported tensor type: {type(x)}, error: {e}")

    def analyze(
        self,
        input_text: str,
        image_path: Optional[Union[str, Path]] = None,
        features=None,
        training=False,
        rng=None
    ) -> Dict:
        """Main analysis method for semiotic content."""
        try:
            activated, confidence = self.activate_if_context_matches(input_text)
            if not activated:
                return {
                    "status": "inactive",
                    "confidence": confidence,
                    "reason": "Context not matched"
                }
            
            image_metadata = {}
            if image_path and self.processor:
                try:
                    description, img_conf = self.describe_image(image_path)
                    image_metadata = {
                        "caption": description,
                        "confidence": img_conf,
                        "source": str(image_path)
                    }
                    full_text = f"{input_text} {description}"
                except Exception as e:
                    logger.warning(f"️ Image description failed: {e}")
                    full_text = input_text
            else:
                full_text = input_text

            # Robust conversion of features to jnp.ndarray
            processed_features = None
            if features is not None:
                try:
                    features = self.to_jax_array(features)
                    processed_features = self.process_features(features, training=training, rng=rng)
                except Exception as e:
                    logger.warning(f"️ Feature processing failed: {e}")

            style, style_conf = self.identify_historical_style(full_text)
            symbols, symbols_conf = self.extract_symbols(full_text, image_metadata)
            parallels = self.find_cross_modal_parallels(style)

            return {
                "status": "active",
                "confidence": {
                    "overall": min(confidence, style_conf, symbols_conf),
                    "style": style_conf,
                    "symbols": symbols_conf,
                    "image": image_metadata.get("confidence", 0.0)
                },
                "style": {
                    "name": style,
                    "confidence": style_conf
                },
                "symbols": symbols,
                "parallels": parallels,
                "metadata": {
                    "image_metadata": image_metadata if image_metadata else None,
                    "processed_features": processed_features
                }
            }

        except Exception as e:
            logger.error(f"Error in semiotic analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "confidence": 0.0
            }

    def describe_image(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """
        Describe an image using BLIP if available, otherwise return placeholder.
        """
        if not self.processor:
            return "Image not processed (BLIP not available)", 0.0
            
        try:
            # This would require PIL and the actual BLIP model
            return f"Descripción de imagen: {image_path}", 0.5
        except Exception as e:
            logger.warning(f"️ Image description failed: {e}")
            return "Error procesando imagen", 0.0


# Export the main class
__all__ = ['MnemosyneSemioModule']