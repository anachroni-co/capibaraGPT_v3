# models_config.py
# Configuración de modelos para el proyecto Capibara6

MODEL_CONFIGS = {
    "capibara6": {
        "name": "capibara6",
        "description": "Gemma3-12B en GPU",
        "endpoint": "http://34.175.104.187:8080",
        "type": "gemma3-12b",
        "location": "gpu-server"
    },
    "oss_120b": {
        "name": "oss_120b", 
        "description": "OSS-120B en TPU",
        "endpoint": "http://tpu-server:8080",
        "type": "oss-120b",
        "location": "tpu-server"
    },
    "gpt_oss_20b": {
        "name": "gpt_oss_20b",
        "description": "GPT-OSS-20B en GPU",
        "endpoint": "http://34.175.215.109:8080",
        "type": "gpt-oss-20b", 
        "location": "gpu-server-2"
    }
}

DEFAULT_MODEL = "gpt_oss_20b"
FALLBACK_ENABLED = True
MAX_RETRIES = 3
TIMEOUT = 120000  # 2 minutos