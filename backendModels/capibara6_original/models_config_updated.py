#!/usr/bin/env python3
"""
Configuración de Modelos - Capibara6 Consensus
Soporte para múltiples modelos con diferentes configuraciones
"""

import os
from typing import Dict, List, Any

# ============================================
# CONFIGURACIÓN DE MODELOS
# ============================================

MODELS_CONFIG = {
    'capibara6': {
        'name': 'Capibara6',
        'base_model': 'Gemma3-12B',
        'server_url': 'http://34.175.104.187:8080/completion',  # IP actualizada
        'type': 'llama_cpp',
        'hardware': 'GPU',
        'status': 'active',
        'priority': 1,
        'prompt_template': {
            'system': 'Responde en el mismo idioma de la pregunta. Si piden código, usa bloques markdown: ```lenguaje',
            'user': '{prompt}',
            'assistant': '',
            'stop_tokens': ['<end_of_turn>', '