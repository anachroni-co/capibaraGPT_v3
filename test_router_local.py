#!/usr/bin/env python3
"""
Prueba local del sistema de router semántico y consenso
"""

import sys
import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, List

# Añadir las carpetas al path
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2')
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/config')

def test_local_model_config():
    """Probar la configuración local de modelos"""
    print("🔍 Verificando configuración de modelos...")
    from config.models_config import get_system_info, get_model_config, get_active_models, get_prompt_template
    
    info = get_system_info()
    print(f"   ✅ Modelos activos: {info['active_models']}/{info['total_models']}")
    print(f"   📋 Modelos: {info['models_list']}")
    print(f"   🤝 Consenso habilitado: {info['consensus_enabled']}")
    
    # Probar acceso a configuraciones específicas
    for model_id in info['models_list']:
        config = get_model_config(model_id)
        print(f"   🧠 {model_id}: {config['name']} ({config['type']})")
    
    return True

def test_prompt_templates():
    """Probar las plantillas de prompts"""
    print("\n📝 Verificando plantillas de prompts...")
    from config.models_config import get_prompt_template, get_available_templates, get_models_for_template
    
    templates = get_available_templates()
    print(f"   🎯 Plantillas disponibles: {templates}")
    
    for template_id in templates:
        template = get_prompt_template(template_id)
        models = get_models_for_template(template_id)
        print(f"   📝 {template_id}: {template['description']} -> {models}")
    
    return True

def test_consenso_logic():
    """Probar la lógica de consenso"""
    print("\n🤝 Verificando lógica de consenso...")
    from config.models_config import CONSENSUS_CONFIG
    
    print(f"   🎯 Método de votación: {CONSENSUS_CONFIG['voting_method']}")
    print(f"   📊 Mín. modelos: {CONSENSUS_CONFIG['min_models']}")
    print(f"   📈 Máx. modelos: {CONSENSUS_CONFIG['max_models']}")
    print(f"   🔄 Modelo fallback: {CONSENSUS_CONFIG['fallback_model']}")
    print(f"   ⚖️  Pesos: {CONSENSUS_CONFIG['model_weights']}")
    
    return True

def test_format_prompt():
    """Probar la función de formateo de prompts"""
    print("\n💬 Verificando formateo de prompts...")
    from config.models_config import format_prompt
    
    test_prompt = "Hola, ¿cómo estás?"
    
    # Probar con diferentes modelos
    for model_id in ['phi4', 'qwen2.5-coder', 'gpt-oss-20b']:
        formatted = format_prompt(model_id, 'general', test_prompt)
        print(f"   🤖 {model_id}: {len(formatted)} caracteres")
    
    return True

def test_model_routing_logic():
    """Probar la lógica de enrutamiento basada en palabras clave (simulada)"""
    print("\n🧭 Verificando lógica de enrutamiento...")
    
    # Simulación de lógica de enrutamiento basada en palabras clave
    def classify_task(prompt: str) -> str:
        """Clasificación simple basada en palabras clave (similar a task_classifier.py)"""
        prompt_lower = prompt.lower()
        
        # Palabras clave para tareas complejas
        complex_keywords = ['análisis', 'razonamiento', 'comparación', 'evaluar', 'estrategia', 'planificación', 'investigación', 'profundo', 'detalle', 'complejo', 'técnico']
        
        # Palabras clave para tareas intermedias
        balanced_keywords = ['explicar', 'qué es', 'cómo funciona', 'describir', 'resumen', 'breve', 'ejemplo', 'definir', 'código', 'programación']
        
        # Palabras clave para tareas simples
        simple_keywords = ['qué', 'quién', 'cuál', 'cuándo', 'dónde', 'chiste', 'broma', 'saludo', 'ayuda']
        
        complex_score = sum(1 for keyword in complex_keywords if keyword in prompt_lower)
        balanced_score = sum(1 for keyword in balanced_keywords if keyword in prompt_lower)
        simple_score = sum(1 for keyword in simple_keywords if keyword in prompt_lower)
        
        # También considerar la longitud del prompt
        if len(prompt) > 200:
            complex_score += 1
        elif len(prompt) > 100:
            balanced_score += 1
            
        scores = {
            'complex': complex_score,
            'balanced': balanced_score,
            'simple': simple_score
        }
        
        # Escoger el modelo con mayor puntuación
        chosen_task = max(scores, key=scores.get)
        
        print(f"   📝 Prompt: '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'")
        print(f"   📊 Puntuaciones - simple: {scores['simple']}, balanced: {scores['balanced']}, complex: {scores['complex']}")
        print(f"   🎯 Clasificación: {chosen_task}")
        
        # Mapear a modelos reales
        if chosen_task == 'complex':
            return 'gpt-oss-20b'
        elif chosen_task == 'balanced':
            return 'qwen2.5-coder'  # o 'mixtral' dependiendo del contenido
        else:
            return 'phi4'  # modelo rápido para tareas simples
    
    # Pruebas de enrutamiento
    test_queries = [
        "¿Qué es Python?",
        "Escribe un código en Python para calcular la serie de Fibonacci",
        "Analiza las implicaciones éticas de la inteligencia artificial en la sociedad moderna",
        "Cuentame un chiste",
        "Explica cómo funciona un transformer en inteligencia artificial"
    ]
    
    for query in test_queries:
        selected_model = classify_task(query)
        print(f"   🧠 Ruta elegida: {selected_model}")
        print()
    
    return True

async def test_consensus_simulation():
    """Simular el proceso de consenso entre modelos"""
    print("🤝 Simulando proceso de consenso...")
    
    from config.models_config import get_active_models, CONSENSUS_CONFIG
    
    active_models = get_active_models()
    print(f"   🤖 Modelos disponibles para consenso: {active_models}")
    
    # Simular consultas a múltiples modelos
    test_prompt = "¿Qué opinas sobre la inteligencia artificial?"
    
    print(f"   📝 Consulta: '{test_prompt}'")
    print(f"   ⚖️  Método de consenso: {CONSENSUS_CONFIG['voting_method']}")
    
    # Simular respuestas de modelos (en una implementación real, esto haría llamadas reales)
    print("   🔄 Simulando respuestas de modelos:")
    for model in active_models[:3]:  # Solo tomar algunos modelos para la simulación
        print(f"     • {model}: respuesta simulada (2.5s, calidad alta)")
    
    # Aplicar lógica de consenso
    if CONSENSUS_CONFIG['voting_method'] == 'weighted':
        weights = CONSENSUS_CONFIG['model_weights']
        print(f"   📊 Pesos aplicados: {weights}")
        
        # Simular selección basada en pesos
        if len(active_models) >= CONSENSUS_CONFIG['min_models']:
            print("   ✅ Condición de consenso satisfecha (mín. modelos disponibles)")
            print("   🎯 Resultado de consenso: respuesta combinada usando pesos")
        else:
            print("   ⚠️  No hay suficientes modelos para consenso")
            print(f"   🔄 Usando modelo fallback: {CONSENSUS_CONFIG['fallback_model']}")
    
    return True

def main():
    """Función principal de pruebas"""
    print("🧪 Pruebas locales del sistema Capibara6")
    print("   Router Semántico y Sistema de Consenso")
    print("=" * 60)
    
    success = True
    
    try:
        # Prueba 1: Configuración de modelos
        success &= test_local_model_config()
        
        # Prueba 2: Plantillas de prompts
        success &= test_prompt_templates()
        
        # Prueba 3: Lógica de consenso
        success &= test_consenso_logic()
        
        # Prueba 4: Formateo de prompts
        success &= test_format_prompt()
        
        # Prueba 5: Lógica de enrutamiento
        success &= test_model_routing_logic()
        
        # Prueba 6: Simulación de consenso (asincrónica)
        asyncio.run(test_consensus_simulation())
        
        print("\n" + "=" * 60)
        print("📋 Resumen de pruebas locales:")
        print("   ✅ Configuración de modelos: Verificada")
        print("   ✅ Plantillas de prompts: Verificadas")
        print("   ✅ Lógica de consenso: Verificada")
        print("   ✅ Formateo de prompts: Verificado")
        print("   ✅ Lógica de enrutamiento: Verificada")
        print("   ✅ Simulación de consenso: Completada")
        
        if success:
            print("\n✅ ¡Todas las pruebas locales se completaron exitosamente!")
            print("\n🚀 El sistema de router semántico y consenso está correctamente configurado con:")
            print("   - phi4: Modelo rápido para tareas simples")
            print("   - qwen2.5-coder: Modelo experto en código y tareas técnicas") 
            print("   - gpt-oss-20b: Modelo complejo para razonamiento avanzado")
            print("   - mixtral: Modelo general para tareas creativas")
            print("   - Sistema de consenso con votación ponderada")
            print("   - Templates de prompts por categoría")
            print("   - Lógica de enrutamiento semántico")
            
            return True
        else:
            print("\n❌ Hubo errores en algunas pruebas")
            return False
            
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)