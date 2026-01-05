#!/usr/bin/env python3
"""
Quick test to verify models are available and responding
"""

import requests
import json
import time

def test_models():
    """Test if the vLLM server is running and models are available"""
    base_url = "http://localhost:8082"  # Main vLLM endpoint for models-europe VM
    
    print("Testing vLLM server connectivity...")
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Server health: OK")
        else:
            print(f"❌ Server health: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return
    
    # Test models endpoint
    try:
        models_response = requests.get(f"{base_url}/v1/models", timeout=10)
        if models_response.status_code == 200:
            models_data = models_response.json()
            print(f"✅ Available models: {len(models_data['data'])}")
            for model in models_data['data']:
                print(f"  - {model['id']}")
        else:
            print(f"❌ Models endpoint: {models_response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint check failed: {e}")
        return

    # Test a simple query with each model
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"
    }
    
    test_prompts = [
        {"name": "Simple Question", "prompt": "What is 2+2?", "model": "phi4:mini"},
        {"name": "Tech Question", "prompt": "What is a neural network?", "model": "qwen2.5-coder:1.5b"}
    ]
    
    for test in test_prompts:
        print(f"\nTesting {test['name']} with {test['model']}:")
        payload = {
            "model": test['model'],
            "messages": [{"role": "user", "content": test['prompt']}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{base_url}/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content']
                print(f"✅ Success in {end_time - start_time:.2f}s")
                print(f"Response: {response_text[:100]}...")
            else:
                print(f"❌ Request failed: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_models()