#!/usr/bin/env python3
"""
Demo de consenso por turnos ultra ligero - Solo un turno con modelo ya cargado
"""

import requests
import time
import json
import psutil

def get_ram_usage_percent():
    """Obtiene el porcentaje de uso de RAM"""
    return psutil.virtual_memory().percent

def single_turn_demo(question: str):
    """
    Demo de un solo turno del consenso por turnos para evitar problemas de RAM
    """
    print("🚀 DEMOSTRACIÓN DE CONSENSO POR TURNOS (Ultra Ligero)")
    print("="*70)
    print(f"Pregunta: {question}")
    print("="*70)
    
    initial_ram = get_ram_usage_percent()
    print(f"📊 RAM inicial: {initial_ram:.1f}%")
    
    # Solo usar el modelo ya cargado para seguridad RAM
    model = "aya_expanse_multilingual"
    specific_question = f"[Perspectiva Global e Integradora] {question} Considera aspectos técnicos, éticos, sociales y económicos."
    
    print(f"\\n🔄 TURNO ÚNICO: {model.upper()}")
    print(f"   Pregunta especializada: '{specific_question[:60]}...'")
    
    ram_before = get_ram_usage_percent()
    print(f"   📊 RAM antes de turno: {ram_before:.1f}%")
    
    if ram_before > 95.0:
        print("   ⚠️  RAM crítica, intentando ejecución ultra ligera...")
    
    start_time = time.time()
    
    try:
        # Solicitud ultra ligera al modelo ya cargado
        response = requests.post(
            "http://localhost:8082/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": specific_question}
                ],
                "max_tokens": 60,  # Muy limitado por seguridad RAM
                "temperature": 0.7
            },
            timeout=90  # Tiempo suficiente para respuesta completa
        )
        
        turn_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            tokens = result['usage']['completion_tokens']
            
            print(f"   ✅ RESPUESTA OBTENIDA CON ÉXITO")
            print(f"   ⏱️  Duración: {turn_time:.2f}s")
            print(f"   🔢 Tokens generados: {tokens}")
            print(f"   ⚡ Velocidad: {tokens/turn_time:.2f} tokens/seg")
            
            print(f"\\n   📄 RESPUESTA DEL ESPECIALISTA:")
            print("   " + "-" * 60)
            print(f"   {content}")
            print("   " + "-" * 60)
            
            # Mostrar métricas del modelo usado
            used_model = result['model']
            print(f"\\n   🤖 Modelo especialista usado: {used_model}")
            
            # Simular cómo sería si otros modelos hubieran participado
            print(f"\\n🎯 SIMULACIÓN DE CONSENSO COMPLETO (conceptual):")
            print("   " + "-" * 60)
            print("   [phi4_fast - Visión General]: 'Análisis rápido del tema...'")  
            print("   [mistral_balanced - Técnica]: 'Capacidades técnicas actuales...'") 
            print("   [qwen_coder - Ingeniería]: 'Aspectos de desarrollo e IA aplicada...'") 
            print(f"   [{used_model} - Global]: '{content[:80]}...'") 
            print("   " + "-" * 60)
            
            print(f"\\n📋 SÍNTESIS PROYECTADA:")
            print("   - Perspectiva Técnica: Modelos especializados evalúan capacidades")
            print("   - Perspectiva Ética: Consideraciones sobre rol humano")
            print("   - Perspectiva Económica: Factibilidad y transformación gradual")
            print("   - Perspectiva Social: Impacto en sociedad y trabajo")
            print("   - Conclusión Global: Integración de todas las perspectivas")
            
            return {
                "success": True,
                "turn_time": turn_time,
                "tokens": tokens,
                "content": content,
                "model": used_model
            }
        else:
            print(f"   ❌ Error HTTP {response.status_code}")
            return {"success": False, "response": response.text[:200]}
    
    except Exception as e:
        print(f"   ❌ Error en turno: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def main():
    """Función principal de la demo"""
    print("🦫 Demo: Consenso por Turnos ARM-Axion")
    print("   Versión ultra ligera con control de RAM")
    print("="*70)
    
    # La pregunta específica
    question = "¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay de ese hecho?"
    
    result = single_turn_demo(question)
    
    final_ram = get_ram_usage_percent()
    print(f"\\n📊 RAM final: {final_ram:.1f}%")
    
    if result.get("success"):
        print("✅ Demostración de consenso por turnos completada exitosamente")
        print("   Aunque solo con un modelo, se ilustra el concepto del sistema")
    else:
        print("⚠️  Demostración incompleta por problemas técnicos o de recursos")

if __name__ == "__main__":
    main()