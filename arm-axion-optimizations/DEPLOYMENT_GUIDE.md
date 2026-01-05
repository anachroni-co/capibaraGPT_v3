# ARM-Axion vLLM Deployment Guide

## Prerequisites
- Google Cloud ARM Axion instance (C4A-standard-32 or higher)
- Ubuntu 22.04 LTS or similar
- Python 3.10+
- vLLM 0.11.1+ with ARM optimizations

## Step-by-Step Deployment

### 1. Clone and Prepare the Repository
```bash
cd /home/elect/capibara6
```

### 2. Verify ARM Compute Library Installation
```bash
# Check that ACL is available in vLLM source
ls -la /home/elect/vllm-source/.deps/arm_compute-src/
```

### 3. Install Dependencies
```bash
# vLLM should already be installed
pip list | grep vllm  # Should show vllm 0.11.1

# Install additional dependencies if needed
pip install fastapi uvicorn pydantic requests
```

### 4. Configure Models
Ensure model files are available at the configured paths:
- `/home/elect/models/phi-4-mini`
- `/home/elect/models/qwen2.5-coder-1.5b`
- `/home/elect/models/gemma-3-27b-it-awq`
- `/home/elect/models/mistral-7b-instruct-v0.2`

### 5. Update Configuration
The optimized configuration is located at:
`/home/elect/capibara6/arm-axion-optimizations/vllm-integration/config.production.json`

This configuration includes:
- ARM-specific optimizations enabled
- Quantization settings optimized
- Memory utilization tuned
- Performance parameters set

### 6. Start the Server
```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm-integration
python3 multi_model_server.py --config config.production.json --host 0.0.0.0 --port 8080
```

### 7. Verify Deployment
```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Test inference
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4_fast",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

## Performance Tuning

### Adjusting for Your Hardware
For different ARM configurations, adjust these parameters in config.production.json:

- `gpu_memory_utilization`: Adjust based on available RAM (0.70-0.90 typical)
- `max_num_seqs`: Higher for more concurrent requests
- `max_num_batched_tokens`: Higher for better throughput
- `tensor_parallel_size`: Only change if using multiple GPUs

### Monitoring
- Track memory usage with `nvidia-smi` or `free -h`
- Monitor request rates and response times
- Check logs for optimization warnings

## Troubleshooting

### If Server Fails to Start
1. Check dependencies: `pip list | grep -E "(vllm|torch|transformers)"`
2. Verify model paths exist and are accessible
3. Check available memory: `free -h`
4. Review configuration for syntax errors

### Performance Issues
1. Verify ACL integration in vLLM build
2. Check if NEON optimizations are enabled
3. Monitor CPU utilization: `htop`
4. Consider adjusting batch sizes based on workload

## Maintenance

### Backup Configuration
Always backup the production configuration before making changes:
```bash
cp config.production.json config.production.json.backup.$(date +%Y%m%d_%H%M%S)
```

### Updating Optimizations
To update ARM optimizations:
1. Update the NEON kernels in `/home/elect/capibara6/arm-axion-optimizations/kernels/`
2. Rebuild with the Makefile: `cd kernels && make acl`
3. Restart the server