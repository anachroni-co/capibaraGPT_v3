# 1. Verificar archivos del modelo AWQ
cd /home/elect/models/gemma-3-27b-it-awq
ls -lh

# 2. Verificar configuración AWQ
cat config.json | grep -E "(quantization|awq)" | head -10

# 3. Test de carga y generación con el modelo AWQ
cd ~/capibara6/arm-axion-optimizations/vllm-integration

python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import time

print("╔════════════════════════════════════════════════════════════════════╗")
print("║  Test de Gemma 3 27B AWQ INT4 en CPU ARM Axion                    ║")
print("║  Comparación vs modelo sin quantizar                               ║")
print("╚════════════════════════════════════════════════════════════════════╝")
print()

model_path = "/home/elect/models/gemma-3-27b-it-awq"

print(f"🔄 Cargando modelo AWQ desde: {model_path}")
print("   Esperado: ~2-3s (similar al anterior)")
start = time.time()

try:
    # Cargar modelo AWQ
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    load_time = time.time() - start
    print(f"✅ Modelo cargado en {load_time:.1f}s")
    
    # Test de generación
    print("\n🧪 Test de generación (100 tokens)...")
    prompt = "Explica qué es ARM Axion:"
    inputs = processor(text=prompt, return_tensors="pt")
    
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    gen_time = time.time() - start
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    tokens_per_sec = 100 / gen_time
    
    print(f"✅ Generado en {gen_time:.1f}s")
    print(f"\n📝 Respuesta: {response[len(prompt):].strip()[:200]}...")
    
    # Calcular memoria
    import psutil
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3
    
    print("\n📊 Estadísticas AWQ INT4:")
    print(f"   - Tiempo de carga: {load_time:.1f}s")
    print(f"   - Tiempo de generación: {gen_time:.1f}s")
    print(f"   - Velocidad: {tokens_per_sec:.1f} tokens/s")
    print(f"   - Memoria usada: {mem_gb:.1f} GB")
    
    print("\n📈 Comparación vs Modelo Original:")
    print(f"   - Velocidad: 0.7 tok/s → {tokens_per_sec:.1f} tok/s ({tokens_per_sec/0.7:.1f}x más rápido)")
    print(f"   - Memoria: 51.4 GB → {mem_gb:.1f} GB ({51.4/mem_gb:.1f}x menos)")
    print(f"   - Tamaño disco: 52 GB → 18 GB (2.9x menos)")
    
    # Evaluar si es aceptable para producción
    if tokens_per_sec >= 3:
        print("\n✅ Performance ACEPTABLE para producción")
    elif tokens_per_sec >= 1.5:
        print("\n⚠️  Performance MARGINAL - considerar optimizaciones adicionales")
    else:
        print("\n❌ Performance INSUFICIENTE - necesita más optimización")
    
    print("\n✅ Test exitoso!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
