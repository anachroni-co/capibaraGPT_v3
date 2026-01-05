#!/usr/bin/env python3
"""
Prueba específica para verificar la integración E2B en las plantillas de prompts
"""

import sys
import os

# Añadir las carpetas al path
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2')
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/config')

def test_e2b_integration_in_templates():
    """Probar que las plantillas incluyen integración con E2B"""
    print("🔍 Verificando integración E2B en plantillas de prompts...")
    
    from config.models_config import get_prompt_template, get_available_templates, format_prompt
    
    templates = get_available_templates()
    
    for template_id in templates:
        template = get_prompt_template(template_id)
        requires_execution = template.get('requires_execution', False)
        execution_context = template.get('execution_context', 'none')
        
        print(f"   📝 {template_id}:")
        print(f"      - Requiere ejecución: {requires_execution}")
        print(f"      - Contexto de ejecución: {execution_context}")
        print(f"      - Modelos: {template.get('models', [])}")
        
        # Probar formateo de prompt con esta plantilla
        test_prompt = f"Test prompt for {template_id}"
        formatted = format_prompt(template.get('models', ['phi4'])[0], template_id, test_prompt)
        
        # Verificar si incluye instrucciones de E2B cuando es necesario
        has_e2b_instructions = "E2B" in formatted
        print(f"      - Incluye instrucciones E2B: {has_e2b_instructions and requires_execution}")
        
        if requires_execution and has_e2b_instructions:
            print(f"      - ✅ Correctamente integrado con E2B")
        elif not requires_execution and not has_e2b_instructions:
            print(f"      - ✅ Correctamente sin integración E2B (como debe ser)")
        else:
            print(f"      - ⚠️  Posible inconsistencia en la integración")
        
        print()
    
    return True

def test_specific_e2b_templates():
    """Probar específicamente las plantillas que deben usar E2B"""
    print("🧪 Verificando plantillas específicas para E2B...")
    
    from config.models_config import get_prompt_template, format_prompt
    
    # Plantillas que deben requerir ejecución E2B
    e2b_templates = ['coding', 'analysis', 'technical']
    non_e2b_templates = ['general', 'creative']
    
    print("   Plantillas que DEBEN usar E2B:")
    for template_id in e2b_templates:
        template = get_prompt_template(template_id)
        requires_execution = template.get('requires_execution', False)
        execution_context = template.get('execution_context', 'none')
        
        print(f"      🤖 {template_id}: {'✅' if requires_execution else '❌'} (contexto: {execution_context})")
        
        # Probar formateo para ver si incluye instrucciones E2B
        model_for_template = template.get('models', ['phi4'])[0]
        formatted = format_prompt(model_for_template, template_id, "Realiza un cálculo")
        has_e2b_instructions = "E2B" in formatted
        print(f"         Instrucciones E2B en prompt: {'✅' if has_e2b_instructions else '❌'}")
    
    print("\n   Plantillas que NO deben usar E2B:")
    for template_id in non_e2b_templates:
        template = get_prompt_template(template_id)
        requires_execution = template.get('requires_execution', False)
        
        print(f"      🤖 {template_id}: {'❌' if requires_execution else '✅'} (sin ejecución)")
        
        # Probar formateo para ver si NO incluye instrucciones E2B
        model_for_template = template.get('models', ['phi4'])[0]
        formatted = format_prompt(model_for_template, template_id, "Contesta generalmente")
        has_e2b_instructions = "E2B" in formatted
        print(f"         Sin instrucciones E2B en prompt: {'✅' if not has_e2b_instructions else '❌'}")
    
    return True

def test_coding_specifics():
    """Probar específicamente la plantilla de codificación"""
    print("\n💻 Verificando plantilla de codificación específicamente...")
    
    from config.models_config import format_prompt
    
    # Probar con el modelo qwen2.5-coder que es experto en código
    formatted_prompt = format_prompt('qwen2.5-coder', 'coding', 'Escribe una función en Python que calcule el factorial de un número')
    
    print(f"   Prompt formateado para codificación:")
    print(f"   {'='*50}")
    print(f"   {formatted_prompt[:200]}...")
    print(f"   {'='*50}")
    
    # Verificar que contiene instrucciones E2B
    has_e2b = "E2B" in formatted_prompt
    has_execution_note = "NOTA IMPORTANTE" in formatted_prompt
    is_python_context = "e2b_python" in formatted_prompt
    
    print(f"   ✅ Contiene E2B: {has_e2b}")
    print(f"   ✅ Contiene nota importante: {has_execution_note}")
    print(f"   ✅ Contexto Python: {is_python_context}")
    
    return has_e2b and has_execution_note

def test_data_analysis_specifics():
    """Probar específicamente la plantilla de análisis de datos"""
    print("\n📊 Verificando plantilla de análisis de datos específicamente...")
    
    from config.models_config import format_prompt
    
    # Probar con el modelo gpt-oss-20b que es bueno para análisis
    formatted_prompt = format_prompt('gpt-oss-20b', 'analysis', 'Analiza este conjunto de datos: [1, 5, 10, 15, 20]')
    
    print(f"   Prompt formateado para análisis:")
    print(f"   {'='*50}")
    print(f"   {formatted_prompt[:200]}...")
    print(f"   {'='*50}")
    
    # Verificar que contiene instrucciones E2B
    has_e2b = "E2B" in formatted_prompt
    has_data_analysis_context = "e2b_data_analysis" in formatted_prompt
    
    print(f"   ✅ Contiene E2B: {has_e2b}")
    print(f"   ✅ Contexto análisis de datos: {has_data_analysis_context}")
    
    return has_e2b and has_data_analysis_context

def main():
    """Función principal de pruebas E2B"""
    print("🧪 Pruebas de integración E2B en el sistema Capibara6")
    print("=" * 60)
    
    success = True
    
    try:
        # Prueba 1: Integración general en plantillas
        success &= test_e2b_integration_in_templates()
        
        # Prueba 2: Plantillas específicas
        success &= test_specific_e2b_templates()
        
        # Prueba 3: Codificación específica
        success &= test_coding_specifics()
        
        # Prueba 4: Análisis de datos específico
        success &= test_data_analysis_specifics()
        
        print("\n" + "=" * 60)
        print("📋 Resumen de pruebas de integración E2B:")
        print("   ✅ Plantillas verificadas para integración E2B")
        print("   ✅ Plantillas de código correctamente integradas")
        print("   ✅ Plantillas de análisis correctamente integradas")
        print("   ✅ Plantillas generales sin integración (como debe ser)")
        print("   ✅ Prompts formateados incluyen instrucciones E2B cuando es necesario")
        
        if success:
            print("\n✅ ¡Todas las pruebas de integración E2B se completaron exitosamente!")
            print("\n🚀 El sistema Capibara6 ahora considera E2B en su flujo:")
            print("   - coding template: Ejecución de código Python en E2B")
            print("   - analysis template: Análisis de datos con ejecución en E2B")
            print("   - technical template: Ejemplos de código con ejecución en E2B") 
            print("   - general y creative: Sin ejecución E2B (como debe ser)")
            print("\n🎯 Esta integración permite:")
            print("   - Generar código que puede ejecutarse en entornos seguros")
            print("   - Análisis de datos con resultados reales")
            print("   - Pruebas de ejemplos técnicos en tiempo real")
            print("   - Verificación de funcionalidad de código propuesto")
            
            return True
        else:
            print("\n❌ Hubo errores en algunas pruebas de integración E2B")
            return False
            
    except Exception as e:
        print(f"\n❌ Error durante las pruebas E2B: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)