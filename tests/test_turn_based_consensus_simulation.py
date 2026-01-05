#!/usr/bin/env python3
"""
Prueba de concepto: Consenso por turnos ARM-Axion
Simulando múltiples modelos especialistas que responden en secuencia
"""

import requests
import time
import json
import psutil
from typing import Dict, List

def get_ram_usage_percent():
    """Obtiene el porcentaje de uso de RAM"""
    return psutil.virtual_memory().percent

def simulate_turn_based_consensus():
    """
    Simula un sistema de consenso por turnos con modelos especialistas
    """
    print("🚀 SIMULACIÓN DE CONSENSO POR TURNOS ARM-Axion")
    print("="*70)
    print("Pregunta: ¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y")
    print("por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay?")
    print("="*70)
    
    ram_initial = get_ram_usage_percent()
    print(f"📊 RAM inicial: {ram_initial:.1f}%")
    
    # La pregunta principal
    main_question = "¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay?"
    
    # Definir perspectivas por modelo especialista
    specialist_questions = {
        "phi4_fast": f"[Visión General] {main_question} Da una respuesta general concisa.",
        "mistral_balanced": f"[Análisis Técnico] {main_question} Considera capacidades y limitaciones técnicas actuales.",
        "qwen_coder": f"[Perspectiva de Ingeniería] {main_question} Considera aspectos de desarrollo tecnológico y automatización.",
        "aya_expanse_multilingual": f"[Perspectiva Global] {main_question} Considera aspectos culturales, éticos y sociales internacionales."
    }
    
    # Resultados del consenso por turnos
    turn_results = {}
    total_time = 0
    total_tokens = 0
    
    print(f"\\n🔄 INICIANDO CONSENSO POR TURNOS...")
    print("-" * 70)
    
    for idx, (model, question) in enumerate(specialist_questions.items(), 1):
        print(f"\\nTURNO {idx}: {model.upper()}")
        print(f"   Pregunta: '{question[:50]}...'")
        
        # Verificar RAM antes de cada turno
        ram_before = get_ram_usage_percent()
        print(f"   📊 RAM antes: {ram_before:.1f}%")
        
        if ram_before > 95.0:
            print(f"   ⚠️  RAM muy alta, abortando turno {idx}")
            continue
            
        start_time = time.time()
        
        try:
            # Intentar usar el modelo disponible
            response = requests.post(
                "http://localhost:8082/v1/chat/completions",  # Servidor estándar
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 40,  # Limitar para seguridad RAM
                    "temperature": 0.7
                },
                timeout=60
            )
            
            turn_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                tokens = result['usage']['completion_tokens']
                
                turn_results[model] = {
                    "response": content,
                    "tokens": tokens,
                    "time": turn_time,
                    "speed": tokens / turn_time if turn_time > 0 else 0
                }
                
                total_time += turn_time
                total_tokens += tokens
                
                print(f"   ✅ Éxito: {turn_time:.2f}s ({tokens} tokens, {tokens/turn_time:.2f} tok/s)")
                print(f"   📄 Resumen: {content[:80]}...")
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error en turno {idx}: {e}")
        
        ram_after = get_ram_usage_percent()
        print(f"   📊 RAM después: {ram_after:.1f}% (+{ram_after-ram_before:+.1f}%)")
        
        # Pequeño delay entre turnos para no sobrecargar
        time.sleep(1)
    
    print("\\n" + "="*70)
    print("📊 RESULTADOS DEL CONSENSO POR TURNOS")
    print("="*70)
    
    if turn_results:
        print(f"⏱️  Tiempo total: {total_time:.2f}s")
        print(f"🔢 Tokens totales: {total_tokens}")
        print(f"⚡ Velocidad promedio: {total_tokens/total_time:.2f} tokens/segundo")
        print(f"👥 Modelos participantes: {len(turn_results)}")
        
        print("\\n📝 PERSPECTIVAS POR ESPECIALISTA:")
        print("-" * 70)
        
        for model, data in turn_results.items():
            print(f"\\n🔹 {model.upper()}:")
            print(f"   Duración: {data['time']:.2f}s | Tokens: {data['tokens']} | Vel.: {data['speed']:.2f} tok/s")
            print(f"   Vista: {data['response'][:120]}...")
            
        print("\\n🎯 SÍNTESIS DE CONSENSO:")
        print("-" * 70)
        
        # Crear una síntesis de las perspectivas
        perspectives = []
        for model, data in turn_results.items():
            model_short = model.split('_')[0].upper()
            perspectives.append(f"- {model_short}: {data['response'][:60]}...")
        
        for perspective in perspectives:
            print(f"  {perspective}")
            
        print(f"\\n🔍 CONCLUSIONES PRELIMINARES:")
        has_technical_limitations = any("limitación" in data["response"].lower() or "difícil" in data["response"].lower() 
                                       for data in turn_results.values())
        has_ethical_concerns = any("ético" in data["response"].lower() or "social" in data["response"].lower() 
                                  or "humano" in data["response"].lower())
        
        print(f"   • ¿Reemplazo total es factible?: {'Posible pero con limitaciones' if has_technical_limitations else 'Potencialmente factible'}")
        print(f"   • ¿Consideraciones éticas presentes?: {'Sí' if has_ethical_concerns else 'No evidentes aún'}")
        print(f"   • ¿Plazo de 20 años razonable?: {'Variable según especialista' if len(turn_results) > 1 else 'Requiere múltiples perspectivas'}")
        
    else:
        print("❌ No se obtuvieron resultados de ningún modelo")
    
    final_ram = get_ram_usage_percent()
    print(f"\\n📊 RAM final: {final_ram:.1f}% (cambio total: {final_ram - ram_initial:+.1f}%)")
    print("✅ Simulación de consenso por turnos completada")


def main():
    """Función principal"""
    print("🦫 Simulación de Consenso por Turnos - Sistema ARM-Axion")
    print("   Evaluando múltiples perspectivas con control de recursos")
    print("="*70)
    
    simulate_turn_based_consensus()


if __name__ == "__main__":
    main()