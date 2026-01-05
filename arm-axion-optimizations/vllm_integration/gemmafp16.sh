# 1. Verificar el modelo FP16 convertido
du -sh /home/elect/models/gemma-3-27b-it-fp16/
ls -lh /home/elect/models/gemma-3-27b-it-fp16/

# 2. Actualizar la configuración para usar el modelo FP16
cd ~/capibara6/arm-axion-optimizations/vllm-integration

# 3. Editar config para apuntar al modelo FP16
sed -i 's|/home/elect/models/gemma-3-27b-it-awq|/home/elect/models/gemma-3-27b-it-fp16|g' config.gemma3.json

# 4. Verificar el cambio
grep "model_path.*gemma" config.gemma3.json

# 5. Test de carga con Transformers (más compatible que vLLM para CPU)
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

print("╔════════════════════════════════════════════════════════════════════╗")
print("║  Test de Gemma 3 27B FP16 en CPU ARM Axion                        ║")
print("╚════════════════════════════════════════════════════════════════════╝")
print()

model_path = "/home/elect/models/gemma-3-27b-it-fp16"

print(f"🔄 Cargando modelo desde: {model_path}")
print("   Esto puede tomar 3-5 minutos...")
start = time.time()

try:
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    load_time = time.time() - start
    print(f"✅ Modelo cargado en {load_time:.1f}s")
    
    # Test de generación
    print("\n🧪 Test de generación...")
    prompt = "Explica qué es ARM Axion en una frase:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )
    gen_time = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Respuesta generada en {gen_time:.1f}s")
    print(f"\n📝 Prompt: {prompt}")
    print(f"📝 Respuesta: {response}")
    
    print("\n📊 Estadísticas:")
    print(f"   - Tiempo de carga: {load_time:.1f}s")
    print(f"   - Tiempo de generación: {gen_time:.1f}s")
    print(f"   - Tokens/segundo: {50/gen_time:.1f}")
    
    print("\n✅ Test exitoso!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
