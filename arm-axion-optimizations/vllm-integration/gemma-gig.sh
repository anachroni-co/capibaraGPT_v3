# Usar el modelo original que ya funciona
cd ~/capibara6/arm-axion-optimizations/vllm-integration

# Actualizar config para usar el modelo original
sed -i 's|/home/elect/models/gemma-3-27b-it-fp16|/home/elect/models/gemma-3-27b-it|g' config.gemma3.json

# Verificar
grep "model_path.*gemma" config.gemma3.json

# Test con el modelo original
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import time

print("╔════════════════════════════════════════════════════════════════════╗")
print("║  Test de Gemma 3 27B Multimodal en CPU ARM Axion                  ║")
print("╚════════════════════════════════════════════════════════════════════╝")
print()

model_path = "/home/elect/models/gemma-3-27b-it"

print(f"🔄 Cargando modelo desde: {model_path}")
print("   Esto puede tomar 3-5 minutos...")
start = time.time()

try:
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Cargar processor (para multimodal)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    load_time = time.time() - start
    print(f"✅ Modelo cargado en {load_time:.1f}s")
    
    # Test de generación solo texto
    print("\n🧪 Test 1: Generación de texto...")
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
    print(f"✅ Generado en {gen_time:.1f}s")
    print(f"\n📝 Respuesta: {response[len(prompt):].strip()[:200]}...")
    
    print("\n📊 Estadísticas:")
    print(f"   - Tiempo de carga: {load_time:.1f}s")
    print(f"   - Tiempo de generación (100 tokens): {gen_time:.1f}s")
    print(f"   - Velocidad: {100/gen_time:.1f} tokens/s")
    print(f"   - TTFT estimado: ~{gen_time/100:.2f}s")
    
    # Calcular memoria usada
    import psutil
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3
    print(f"   - Memoria usada: {mem_gb:.1f} GB")
    
    print("\n✅ Test exitoso!")
    print("\n💡 Modelo funcionando correctamente en CPU ARM Axion")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
