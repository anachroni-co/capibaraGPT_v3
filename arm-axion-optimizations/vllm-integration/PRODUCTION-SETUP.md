# Production Setup - ARM Axion VM

Guía para deployment en tu VM ARM Axion de Google Cloud con los modelos ya descargados.

## 📦 Modelos Disponibles

Según tu configuración actual, tienes estos 4 modelos en `/home/user/models/`:

| Modelo | Tamaño | Uso | Quantización |
|--------|--------|-----|--------------|
| **phi4-mini-instruct** | ~1.5GB (AWQ) | Respuestas rápidas, chat simple | AWQ (4-bit) |
| **mistral-7b-v0.2** | ~3.5GB (AWQ) | Código, explicaciones técnicas | AWQ (4-bit) |
| **qwen2.5-7b-instruct** | ~4GB (Q4) | Multilingüe, análisis de texto | Q4_0 (4-bit) |
| **gpt-oss-20b** | ~10GB (Q4) | Razonamiento complejo, research | Q4_0 (4-bit) |

**Total memoria**: ~19GB → Caben todos en tu C4A-standard-32 (128GB RAM)

## 🚀 Quick Start (Producción)

### 1. Conectar a VM ARM Axion

```bash
# Conectar a tu VM
gcloud compute ssh [NOMBRE_VM_AXION] --zone [ZONA]

# Navegar al directorio
cd ~/capibara6/arm-axion-optimizations/vllm-integration
```

### 2. Verificar Modelos

```bash
# Verificar que los modelos estén descargados
ls -lah /home/user/models/

# Deberías ver:
# phi4-mini-instruct/
# mistral-7b-v0.2-instruct/  (o similar)
# qwen2.5-7b-instruct/
# gpt-oss-20b/
```

### 3. Deployment Automático

```bash
# Ejecutar script de deployment
./deploy-production.sh

# El script:
# ✅ Verifica arquitectura ARM
# ✅ Compila kernels NEON
# ✅ Instala vLLM
# ✅ Configura systemd service
# ✅ Optimiza sistema
```

### 4. Iniciar Servidor

**Opción A: Manualmente** (para testing)

```bash
python3 inference_server.py --host 0.0.0.0 --port 8080
```

**Opción B: Systemd** (para producción)

```bash
# Iniciar
sudo systemctl start vllm-capibara6

# Ver estado
sudo systemctl status vllm-capibara6

# Ver logs en tiempo real
sudo journalctl -u vllm-capibara6 -f

# Habilitar auto-start en boot
sudo systemctl enable vllm-capibara6
```

### 5. Verificar Funcionamiento

```bash
# Health check
curl http://localhost:8080/health

# Listar expertos
curl http://localhost:8080/experts

# Stats
curl http://localhost:8080/stats

# Test completion
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain ARM Axion processors",
    "max_tokens": 100
  }'
```

## 📊 Configuración de Routing

El sistema automáticamente enruta requests a los expertos apropiados:

```python
# Routing automático por dominio:

"Hola, ¿cómo estás?"
  → phi4_fast (respuesta simple)

"Implementa binary search en Python"
  → mistral_balanced (código técnico)

"翻译这句话到英语"  (Traducir al inglés)
  → qwen_multilingual (multilingüe)

"Analiza las implicaciones económicas del cambio climático"
  → gptoss_complex (análisis profundo)
```

## ⚙️ Ajustar Configuración

### Editar Paths de Modelos

Si tus modelos están en ubicaciones diferentes:

```bash
nano config.production.json
```

Actualizar paths:

```json
{
  "experts": [
    {
      "expert_id": "phi4_fast",
      "model_path": "/ruta/real/a/phi4-mini-instruct",
      ...
    }
  ]
}
```

### Ajustar Memoria

Si tienes menos RAM disponible:

```json
{
  "experts": [
    {
      "gpu_memory_utilization": 0.70,  // Reducir de 0.85
      "max_num_seqs": 128,              // Reducir de 256
      ...
    }
  ]
}
```

### Deshabilitar Expertos

Si quieres usar solo algunos modelos:

```json
{
  "experts": [
    // Comentar o eliminar expertos que no quieras cargar
    {
      "expert_id": "phi4_fast",
      ...
    },
    {
      "expert_id": "mistral_balanced",
      ...
    }
    // qwen y gptoss comentados = no se cargan
  ]
}
```

## 🔧 Troubleshooting

### Error: "Model not found"

```bash
# Verificar path exacto de modelos
ls -la /home/user/models/

# Actualizar config.production.json con paths correctos
```

### Error: "Out of memory"

```bash
# Opción 1: Usar menos expertos
# Editar config.production.json y cargar solo 2-3 modelos

# Opción 2: Reducir batch size
# En config: "max_num_seqs": 64  (en lugar de 256)

# Opción 3: Usar más agresiva quantización
# Cambiar "awq" → "q4_0" para más compression
```

### Performance lento

```bash
# Verificar CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Debe ser "performance"

# Si no:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verificar NEON kernels compilados
cd ~/capibara6/arm-axion-optimizations
make info
```

### vLLM no inicia

```bash
# Ver logs detallados
sudo journalctl -u vllm-capibara6 -n 100 --no-pager

# Verificar vLLM instalado
python3 -c "import vllm; print(vllm.__version__)"

# Reinstalar si necesario
pip3 install --upgrade vllm
```

## 📈 Monitoring

### Logs

```bash
# Logs del servidor
sudo journalctl -u vllm-capibara6 -f

# Logs de sistema
dmesg | tail -50

# Uso de memoria
watch -n 1 free -h

# Uso de CPU
htop
```

### Métricas

```bash
# Stats de vLLM
curl http://localhost:8080/stats | jq

# Info de expertos
curl http://localhost:8080/experts | jq

# Health check continuo
watch -n 5 'curl -s http://localhost:8080/health | jq'
```

## 🔄 Updates

### Actualizar código

```bash
cd ~/capibara6
git pull origin main

# Recompilar kernels
cd arm-axion-optimizations
make clean && make all

# Reiniciar servicio
sudo systemctl restart vllm-capibara6
```

### Actualizar vLLM

```bash
pip3 install --upgrade vllm

# Reiniciar
sudo systemctl restart vllm-capibara6
```

## 🎯 Casos de Uso Optimizados

### RAG con Multi-Expert

```python
import openai

openai.api_base = "http://[TU-VM-IP]:8080/v1"
openai.api_key = "dummy"

# El router automáticamente selecciona el mejor experto
response = openai.ChatCompletion.create(
    model="default",
    messages=[
        {"role": "system", "content": "Eres un asistente experto"},
        {"role": "user", "content": "Analiza este código Python: ..."}
    ]
)
# → Usa mistral_balanced (experto en código)
```

### High-Throughput

```python
# Múltiples requests en paralelo
# vLLM continuous batching los procesa eficientemente

import asyncio

async def generate_many():
    tasks = [
        generate_async(prompt)
        for prompt in prompts_list
    ]
    return await asyncio.gather(*tasks)

# Throughput esperado: 150-200 req/min
```

## 📞 Soporte

- **Issues**: https://github.com/anacronic-io/capibara6/issues
- **Email**: marco@anachroni.co
- **Docs**: arm-axion-optimizations/vllm-integration/README.md

---

**Optimizado para**: Google Cloud C4A (ARM Axion)
**Última actualización**: 2025-11-19
