# Capibara6 Router System - User Guide

## Overview
The Capibara6 system uses intelligent routing to direct your queries to the most appropriate AI model based on content analysis, complexity, and model specialization.

## Available Models

1. **fast_response** (`phi4:mini`)
   - Purpose: Fast responses for simple queries
   - Max tokens: 512
   - Use cases: Simple questions, quick responses, greetings, basic calculations
   - Response time: ~0.2s

2. **balanced** (`qwen2.5-coder-1.5b`)
   - Purpose: Technical questions and coding tasks
   - Max tokens: 1024
   - Use cases: Programming help, technical explanations, debugging
   - Response time: ~0.5s

3. **complex** (`gpt-oss:20b`)
   - Purpose: Complex analysis and deep reasoning
   - Max tokens: 2048
   - Use cases: Analysis, research, strategic planning, complex comparisons
   - Response time: ~1.0s+

## How Routing Works

The system analyzes your query using:
- Keyword matching (e.g., "code", "python", "debug", "analyze")
- Query length and complexity
- Technical terminology detection
- Specialized routing rules for each model

## Testing the Router

To test the router system with your own queries:

```bash
cd /home/elect/capibara6
python3 router_system_demo_fixed.py
```

## API Integration

The system is configured to use the vLLM endpoint at:
- URL: `http://34.12.166.76:8000/v1`
- Models are accessible via the OpenAI-compatible API

## Configuration

- Main configuration: `/home/elect/capibara6/model_config.json`
- Backend routing: `/home/elect/capibara6/backend/`
- ARM-Axion optimizations: `/home/elect/capibara6/arm-axion-optimizations/`

## Troubleshooting

If the router doesn't respond:
1. Verify the vLLM server is running
2. Check network connectivity to `http://34.12.166.76:8000`
3. Confirm model services are operational

## Performance

The system is optimized for ARM Axion processors with:
- NEON-optimized kernels
- ARM Compute Library integration
- Quantized models for efficiency
- Continuous batching for throughput