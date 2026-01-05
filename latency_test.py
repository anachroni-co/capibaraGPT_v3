import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración del servidor
SERVER_URL = "http://localhost:8081/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

# Modelos disponibles
MODELS = [
    "phi4_fast",
    "mistral_balanced", 
    "qwen_coder",
    "gemma3_multimodal",
    "aya_expanse_multilingual"
]

# Prompt de prueba
TEST_PROMPT = {
    "messages": [
        {"role": "user", "content": "Hola, ¿cómo estás? Responde brevemente."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
}

def test_model_latency(model_name, request_data):
    """Función para probar la latencia de un modelo individual"""
    try:
        # Añadir el modelo a los datos de la solicitud
        data = request_data.copy()
        data["model"] = model_name
        
        # Medir tiempo de inicio
        start_time = time.time()
        
        # Realizar la solicitud
        response = requests.post(SERVER_URL, headers=HEADERS, json=data)
        
        # Medir tiempo final
        end_time = time.time()
        
        # Calcular latencia
        latency = end_time - start_time
        
        # Verificar si la solicitud fue exitosa
        if response.status_code == 200:
            response_data = response.json()
            # Extraer texto de la respuesta (primera opción)
            response_text = response_data['choices'][0]['message']['content']
            return {
                'model': model_name,
                'latency': latency,
                'status': 'success',
                'response_length': len(response_text),
                'status_code': response.status_code
            }
        else:
            return {
                'model': model_name,
                'latency': latency,
                'status': 'error',
                'status_code': response.status_code,
                'error': response.text
            }
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time if 'start_time' in locals() else 0
        return {
            'model': model_name,
            'latency': latency,
            'status': 'exception',
            'error': str(e)
        }

def run_latency_tests():
    """Función principal para correr pruebas de latencia"""
    print("🚀 Iniciando pruebas de latencia para 5 modelos...")
    print("="*80)
    
    results = []
    
    # Probar cada modelo individualmente
    for model in MODELS:
        print(f"🔄 Probando modelo: {model}")
        result = test_model_latency(model, TEST_PROMPT)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ {model}: {result['latency']:.3f}s (respuesta: {result['response_length']} chars)")
        else:
            print(f"❌ {model}: Error (latencia: {result['latency']:.3f}s)")
        print("-" * 40)
    
    return results

def run_concurrent_latency_tests():
    """Función para correr pruebas de latencia concurrentemente"""
    print("🚀 Iniciando pruebas de latencia concurrentes para 5 modelos...")
    print("="*80)
    
    results = []
    
    # Usar ThreadPoolExecutor para probar todos los modelos al mismo tiempo
    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        # Crear future para cada modelo
        future_to_model = {
            executor.submit(test_model_latency, model, TEST_PROMPT): model 
            for model in MODELS
        }
        
        # Recoger resultados a medida que se completan
        for future in as_completed(future_to_model):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ {result['model']}: {result['latency']:.3f}s (respuesta: {result['response_length']} chars)")
            else:
                print(f"❌ {result['model']}: Error (latencia: {result['latency']:.3f}s)")
    
    return results

if __name__ == "__main__":
    print("Ejecutando pruebas de latencia secuenciales...")
    results = run_latency_tests()

    # Imprimir resumen
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("="*80)
    for result in results:
        if result['status'] == 'success':
            print(f"{result['model']:<25} | Latencia: {result['latency']:.3f}s | {result['response_length']} chars")
        else:
            print(f"{result['model']:<25} | ERROR: {result['latency']:.3f}s | {result.get('error', 'Unknown error')}")

    # Guardar resultados en archivo
    with open('/tmp/latency_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Resultados guardados en /tmp/latency_test_results.json")