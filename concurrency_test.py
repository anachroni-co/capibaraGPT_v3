#!/usr/bin/env python3
"""
Script para probar concurrencia en el servidor multi_model_server
"""
import asyncio
import time
import json
import requests
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading


def check_memory_usage():
    """Verificar el uso actual de memoria"""
    memory = psutil.virtual_memory()
    return memory.percent, memory.available


def send_request(conversation_id, model="phi4_fast", user_id=1):
    """Enviar una solicitud al servidor"""
    try:
        url = "http://localhost:8082/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Usuario {user_id} - Hola, esta es una prueba de concurrencia #{conversation_id}. ¿Cómo estás? Por favor dime cómo puedes ayudarme."
                }
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            tokens = result.get('usage', {}).get('total_tokens', 0)
            return {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "status": "success",
                "response_time": round(end_time - start_time, 2),
                "tokens": tokens,
                "memory_usage": check_memory_usage()[0]
            }
        else:
            return {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "status": "error",
                "error_code": response.status_code,
                "memory_usage": check_memory_usage()[0]
            }
    
    except Exception as e:
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "status": "exception",
            "error": str(e),
            "memory_usage": check_memory_usage()[0]
        }


def run_concurrency_test(num_users, duration=30):
    """Ejecutar prueba de concurrencia con un número específico de usuarios"""
    print(f"\n{'='*60}")
    print(f"INICIANDO PRUEBA DE CONCURRENCIA: {num_users} usuarios")
    print(f"Duración: {duration} segundos")
    print(f"{'='*60}")
    
    initial_memory, _ = check_memory_usage()
    print(f"Memoria inicial: {initial_memory:.1f}%")
    
    # Configurar tareas concurrentes
    # Cada usuario enviará solicitudes cada 2 segundos aproximadamente
    tasks = []
    results = []
    stop_event = threading.Event()
    
    def user_task(user_id):
        conversation_counter = 0
        while not stop_event.is_set():
            conversation_id = f"user_{user_id}_conv_{conversation_counter}"
            result = send_request(conversation_id, "phi4_fast", user_id)
            results.append(result)
            
            # Verificar si hemos alcanzado el límite de memoria
            current_memory, available_memory = check_memory_usage()
            if current_memory > 90:
                print(f"⚠️  Límite de memoria >90% alcanzado: {current_memory:.1f}%")
                stop_event.set()
                break
            
            conversation_counter += 1
            time.sleep(2)  # Esperar 2 segundos entre solicitudes
    
    # Crear hilos para cada usuario
    threads = []
    for user_id in range(1, num_users + 1):
        thread = threading.Thread(target=user_task, args=(user_id,))
        threads.append(thread)
        thread.start()
    
    # Esperar durante la duración especificada o hasta que se alcance el límite de memoria
    start_time = time.time()
    while time.time() - start_time < duration and not stop_event.is_set():
        time.sleep(1)
    
    # Detener todas las tareas
    stop_event.set()
    
    # Esperar a que terminen todos los hilos
    for thread in threads:
        thread.join(timeout=5)  # Timeout de 5 segundos
    
    # Analizar resultados
    successful_requests = [r for r in results if r["status"] == "success"]
    error_requests = [r for r in results if r["status"] == "error"]
    exception_requests = [r for r in results if r["status"] == "exception"]
    
    final_memory, _ = check_memory_usage()
    
    print(f"\nResultados para {num_users} usuarios:")
    print(f"  - Solicitudes exitosas: {len(successful_requests)}")
    print(f"  - Solicitudes con error: {len(error_requests)}")
    print(f"  - Excepciones: {len(exception_requests)}")
    print(f"  - Memoria inicial: {initial_memory:.1f}%")
    print(f"  - Memoria final: {final_memory:.1f}%")
    
    if successful_requests:
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        avg_tokens = sum(r["tokens"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        print(f"  - Tiempo promedio de respuesta: {avg_response_time:.2f}s")
        print(f"  - Promedio de tokens: {avg_tokens:.0f}")
    
    # Verificar si se alcanzó el límite de memoria
    memory_limit_reached = final_memory > 90
    server_responsive = final_memory <= 95  # Consideramos servidor bloqueado si >95%
    
    return {
        "num_users": num_users,
        "successful_requests": len(successful_requests),
        "error_requests": len(error_requests),
        "exception_requests": len(exception_requests),
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "memory_limit_reached": memory_limit_reached,
        "server_responsive": server_responsive,
        "results": results
    }


def main():
    """Función principal para ejecutar pruebas de concurrencia progresiva"""
    print("🧪 Prueba de concurrencia para multi_model_server")
    print("Objetivo: Encontrar el límite de usuarios concurrentes antes del 90% de RAM")
    
    # Verificar que el servidor esté disponible
    try:
        response = requests.get("http://localhost:8082/health", timeout=10)
        if response.status_code == 200:
            print("✅ Servidor disponible")
        else:
            print("❌ Servidor no responde")
            return
    except Exception as e:
        print(f"❌ Servidor no responde: {e}")
        return
    
    results_summary = []
    max_users_to_test = 16  # Máximo número de usuarios a probar
    user_counts = [1, 2, 4, 6, 8, 10, 12, 16]  # Conteos progresivos
    
    for num_users in user_counts:
        if num_users > max_users_to_test:
            break
            
        result = run_concurrency_test(num_users, duration=20)  # 20 segundos por prueba
        results_summary.append(result)
        
        # Verificar si alcanzamos el límite de memoria
        if result["memory_limit_reached"]:
            print(f"\n🚨 Límite de memoria alcanzado con {num_users} usuarios")
            break
        
        # Breve pausa entre pruebas
        time.sleep(5)
    
    # Mostrar resumen
    print(f"\n{'='*80}")
    print("RESUMEN DE PRUEBAS DE CONCURRENCIA")
    print(f"{'='*80}")
    
    for result in results_summary:
        status = "⚠️ MEM-LIMIT" if result["memory_limit_reached"] else "✅ OK"
        print(f"Usuarios: {result['num_users']:2d} | "
              f"Éxito: {result['successful_requests']:3d} | "
              f"Errores: {result['error_requests']:2d} | "
              f"Mem: {result['initial_memory']:5.1f}%→{result['final_memory']:5.1f}% | "
              f"{status}")
    
    # Determinar el máximo número seguro de usuarios
    safe_users = 0
    for result in results_summary:
        if not result["memory_limit_reached"] and result["server_responsive"]:
            safe_users = result["num_users"]
        else:
            break
    
    print(f"\n🎯 CONCLUSIÓN:")
    print(f"   Máximo número seguro de usuarios concurrentes: {safe_users}")
    print(f"   (Sin exceder el 90% de uso de RAM)")
    
    # Guardar resultados detallados
    with open("/tmp/concurrency_test_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📊 Resultados detallados guardados en: /tmp/concurrency_test_results.json")


if __name__ == "__main__":
    main()