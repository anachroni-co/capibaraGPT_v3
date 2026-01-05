#!/usr/bin/env python3
"""
Script para probar concurrencia en el servidor multi_model_server por modelo individual
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


def send_request(conversation_id, model, user_id=1):
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
            "max_tokens": 30
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
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
                "error_msg": response.text[:200],  # First 200 chars of error
                "memory_usage": check_memory_usage()[0]
            }
    
    except requests.exceptions.Timeout:
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "status": "timeout",
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


def load_model_first(model_name):
    """Cargar un modelo primero con una solicitud simple"""
    print(f"📡 Cargando modelo {model_name}...")
    try:
        url = "http://localhost:8082/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Carga el modelo por favor."
                }
            ],
            "temperature": 0.1,
            "max_tokens": 5
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            print(f"✅ Modelo {model_name} cargado exitosamente")
            return True
        else:
            print(f"⚠️ Error al cargar {model_name}: {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"⚠️ Excepción al cargar {model_name}: {str(e)}")
        return False


def run_concurrency_test_per_model(model_name, num_users, duration=30):
    """Ejecutar prueba de concurrencia para un modelo específico"""
    print(f"\n{'='*60}")
    print(f"INICIANDO PRUEBA DE CONCURRENCIA: {num_users} usuarios para modelo {model_name}")
    print(f"Duración: {duration} segundos")
    print(f"{'='*60}")
    
    initial_memory, _ = check_memory_usage()
    print(f"Memoria inicial: {initial_memory:.1f}%")
    
    # Configurar tareas concurrentes para un modelo específico
    tasks = []
    results = []
    stop_event = threading.Event()
    
    def user_task(user_id):
        conversation_counter = 0
        while not stop_event.is_set():
            conversation_id = f"user_{user_id}_conv_{conversation_counter}"
            result = send_request(conversation_id, model_name, user_id)
            results.append(result)
            
            # Mostrar estado periódicamente
            if conversation_counter % 5 == 0 and conversation_counter > 0:
                success_count = len([r for r in results if r["user_id"] == user_id and r["status"] == "success"])
                print(f"  Usuario {user_id}: {success_count} solicitudes exitosas hasta ahora")
            
            # Verificar si hemos alcanzado el límite de memoria
            current_memory, available_memory = check_memory_usage()
            if current_memory > 90:
                print(f"⚠️  Límite de memoria >90% alcanzado: {current_memory:.1f}%")
                stop_event.set()
                break
            
            conversation_counter += 1
            time.sleep(3)  # Esperar 3 segundos entre solicitudes
    
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
    timeout_requests = [r for r in results if r["status"] == "timeout"]
    exception_requests = [r for r in results if r["status"] == "exception"]
    
    final_memory, _ = check_memory_usage()
    
    print(f"\nResultados para {model_name} con {num_users} usuarios:")
    print(f"  - Solicitudes exitosas: {len(successful_requests)}")
    print(f"  - Solicitudes con error: {len(error_requests)}")
    print(f"  - Timeouts: {len(timeout_requests)}")
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
        "model": model_name,
        "num_users": num_users,
        "successful_requests": len(successful_requests),
        "error_requests": len(error_requests),
        "timeout_requests": len(timeout_requests),
        "exception_requests": len(exception_requests),
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "memory_limit_reached": memory_limit_reached,
        "server_responsive": server_responsive,
        "avg_response_time": sum(r["response_time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0,
        "results": results
    }


def test_model_concurrency(model_name):
    """Función principal para probar concurrencia por modelo"""
    print(f"\n🧪 Prueba de concurrencia para el modelo: {model_name}")
    
    # Cargar el modelo primero
    load_model_first(model_name)
    
    # Verificar que el servidor esté disponible
    try:
        response = requests.get("http://localhost:8082/health", timeout=10)
        if response.status_code == 200:
            print("✅ Servidor disponible")
        else:
            print("❌ Servidor no responde")
            return []
    except Exception as e:
        print(f"❌ Servidor no responde: {e}")
        return []
    
    results_summary = []
    user_counts = [1, 2, 4, 8, 12, 16, 20]  # Conteos progresivos
    
    for num_users in user_counts:
        result = run_concurrency_test_per_model(model_name, num_users, duration=25)  # 25 segundos por prueba
        results_summary.append(result)
        
        # Verificar si alcanzamos el límite de memoria
        if result["memory_limit_reached"]:
            print(f"\n🚨 Límite de memoria alcanzado con {num_users} usuarios para {model_name}")
            break
        
        # Breve pausa entre pruebas
        time.sleep(3)
    
    # Mostrar resumen para este modelo
    print(f"\n{'='*60}")
    print(f"RESUMEN PARA EL MODELO: {model_name}")
    print(f"{'='*60}")
    
    for result in results_summary:
        status = "⚠️ MEM-LIMIT" if result["memory_limit_reached"] else "✅ OK"
        total_requests = (result['successful_requests'] + result['error_requests'] + 
                         result['timeout_requests'] + result['exception_requests'])
        print(f"Usuarios: {result['num_users']:2d} | "
              f"Total req: {total_requests:3d} | "
              f"Éxito: {result['successful_requests']:3d} | "
              f"Errores: {result['error_requests']:2d} | "
              f"Timeout: {result['timeout_requests']:2d} | "
              f"Resp_Tiempo: {result['avg_response_time']:.2f}s | "
              f"Mem: {result['initial_memory']:5.1f}%→{result['final_memory']:5.1f}% | "
              f"{status}")
    
    print(f"\n🎯 CONCLUSIÓN PARA {model_name}:")
    safe_users = 0
    for result in results_summary:
        if not result["memory_limit_reached"] and result["server_responsive"] and result["successful_requests"] > 0:
            safe_users = result["num_users"]
        else:
            # Si hay errores pero se mantiene por debajo del 90%, podría ser aceptable
            if not result["memory_limit_reached"] and result["server_responsive"]:
                safe_users = result["num_users"]
            else:
                break
    
    if safe_users > 0:
        print(f"   Número seguro máximo de usuarios concurrentes: {safe_users}")
        avg_success_rate = sum(r["successful_requests"] for r in results_summary if r["num_users"] <= safe_users) / len([r for r in results_summary if r["num_users"] <= safe_users])
        print(f"   Tasa promedio de éxito: {avg_success_rate:.1f} solicitudes por prueba")
    
    return results_summary


def main():
    """Función principal para probar todos los modelos disponibles"""
    print("🧪 Prueba de concurrencia por modelo para multi_model_server")
    print("Objetivo: Evaluar la capacidad de cada modelo individualmente")
    
    # Obtener la lista de modelos disponibles
    try:
        response = requests.get("http://localhost:8082/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model["id"] for model in models_data["data"]]
            print(f"✅ Servidor disponible - Modelos encontrados: {available_models}")
        else:
            print("❌ No se pudieron obtener los modelos")
            available_models = ["phi4_fast", "qwen_coder"]  # modelos por defecto
    except Exception as e:
        print(f"❌ Error al obtener modelos: {e}")
        available_models = ["phi4_fast", "qwen_coder"]  # modelos por defecto
    
    all_results = {}
    
    for model_name in available_models:
        model_results = test_model_concurrency(model_name)
        all_results[model_name] = model_results
    
    # Guardar resultados generales
    with open("/tmp/model_concurrency_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Resultados detallados guardados en: /tmp/model_concurrency_results.json")
    
    # Crear archivo de reporte Markdown
    create_markdown_report(all_results)
    
    print(f"\n📄 Reporte de pruebas guardado en: /tmp/model_concurrency_report.md")


def create_markdown_report(results):
    """Crear un archivo de reporte en formato Markdown"""
    with open("/tmp/model_concurrency_report.md", "w") as f:
        f.write("# Pruebas de Concurrencia por Modelo\n\n")
        f.write("Este informe detalla las pruebas de concurrencia realizadas para cada modelo disponible en el servidor `multi_model_server`.\n\n")
        
        for model_name, model_results in results.items():
            f.write(f"## Modelo: {model_name}\n\n")
            
            if model_results:
                f.write("| Usuarios | Solicitudes | Éxito | Errores | Timeouts | Tiempo Resp. (s) | Memoria Inicial | Memoria Final |\n")
                f.write("|----------|-------------|-------|---------|----------|-------------------|-----------------|---------------|\n")
                
                for result in model_results:
                    total_requests = (result['successful_requests'] + result['error_requests'] + 
                                     result['timeout_requests'] + result['exception_requests'])
                    
                    f.write(f"| {result['num_users']} | {total_requests} | {result['successful_requests']} | "
                            f"{result['error_requests']} | {result['timeout_requests']} | "
                            f"{result['avg_response_time']:.2f} | {result['initial_memory']:.1f}% | "
                            f"{result['final_memory']:.1f}% |\n")
                
                # Encontrar el máximo número seguro de usuarios
                safe_users = 0
                for result in model_results:
                    if not result["memory_limit_reached"] and result["server_responsive"] and result["successful_requests"] > 0:
                        safe_users = result["num_users"]
                    else:
                        if not result["memory_limit_reached"] and result["server_responsive"]:
                            safe_users = result["num_users"]
                        else:
                            break
                
                f.write(f"\n### Conclusión para {model_name}:\n")
                if safe_users > 0:
                    f.write(f"- Número máximo seguro de usuarios concurrentes: **{safe_users}**\n")
                    avg_success_rate = sum(r["successful_requests"] for r in model_results if r["num_users"] <= safe_users) / len([r for r in model_results if r["num_users"] <= safe_users])
                    f.write(f"- Tasa promedio de éxito: **{avg_success_rate:.1f}** solicitudes por prueba\n")
                else:
                    f.write("- No se pudo determinar un número seguro de usuarios concurrentes\n")
                
                f.write("\n")
            else:
                f.write("No se obtuvieron resultados para este modelo.\n\n")
        
        f.write("## Resumen General\n\n")
        f.write("Las pruebas de concurrencia se realizaron con el objetivo de determinar:\n\n")
        f.write("- El número máximo de usuarios concurrentes que puede manejar cada modelo sin exceder el 90% de uso de RAM\n")
        f.write("- El rendimiento del modelo bajo carga concurrente\n")
        f.write("- El tiempo promedio de respuesta bajo diferentes niveles de concurrencia\n\n")
        
        f.write("### Metodología\n\n")
        f.write("- Duración de cada prueba: 25 segundos\n")
        f.write("- Intervalo entre solicitudes: 3 segundos\n")
        f.write("- Timeout de solicitud: 60 segundos\n")
        f.write("- Cada usuario envía solicitudes concurrentes al mismo modelo\n")
        f.write("- Se monitorea el uso de memoria durante todas las pruebas\n\n")
        
        f.write("### Configuración del Servidor\n\n")
        f.write("- Configuración ligera con lazy loading activado\n")
        f.write("- Optimizaciones ARM Axion aplicadas\n")
        f.write("- Uso de CPU (no GPU) para inferencia\n\n")


if __name__ == "__main__":
    main()