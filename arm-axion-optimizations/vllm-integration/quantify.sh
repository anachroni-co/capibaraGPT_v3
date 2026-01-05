# Alternativa: Usar llama.cpp para quantizar a GGUF
pip3 install llama-cpp-python

python3 << 'EOF'
from transformers import AutoModelForCausalLM
import torch

print("Convirtiendo a formato compatible...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/elect/models/gemma-3-27b-it",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)

# Guardar en formato compatible
model.save_pretrained(
    "/home/elect/models/gemma-3-27b-it-fp16",
    safe_serialization=True
)
print("✅ Modelo convertido a FP16")
EOF
