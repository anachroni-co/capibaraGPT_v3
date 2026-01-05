#!/usr/bin/env python3
"""
Test script to verify the vLLM migration changes
"""

import sys
import os
import json

def test_model_config():
    """Test that the model_config.json file is valid and contains expected models"""
    print("🔍 Testing model_config.json...")
    
    try:
        with open('/home/elect/capibara6/model_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded model_config.json successfully")
        
        expected_models = ['fast_response', 'balanced', 'complex']
        for model in expected_models:
            if model not in config['models']:
                print(f"❌ Missing model: {model}")
                return False
            else:
                print(f"✅ Found model: {model} -> {config['models'][model]['name']}")
        
        # Verify model changes
        if config['models']['fast_response']['name'] != 'phi4:mini':
            print(f"❌ Expected phi4:mini, got {config['models']['fast_response']['name']}")
            return False
        else:
            print(f"✅ Fast response model updated to: {config['models']['fast_response']['name']}")
        
        if config['models']['balanced']['name'] != 'qwen2.5-coder-1.5b':
            print(f"❌ Expected qwen2.5-coder-1.5b, got {config['models']['balanced']['name']}")
            return False
        else:
            print(f"✅ Balanced model updated to: {config['models']['balanced']['name']}")
            
        # Check for vllm endpoint
        if 'vllm_endpoint' not in config['api_settings']:
            print(f"❌ vllm_endpoint not found in api_settings")
            return False
        else:
            print(f"✅ vLLM endpoint configured: {config['api_settings']['vllm_endpoint']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model_config.json: {e}")
        return False


def test_python_imports():
    """Test that Python modules can be imported without errors"""
    print("\n🔍 Testing Python imports...")
    
    try:
        # Add backend to path
        sys.path.insert(0, '/home/elect/capibara6/backend')
        
        # Test importing the updated ollama_client (now VLLM client)
        from ollama_client import VLLMClient
        print(f"✅ Successfully imported VLLMClient")
        
        # Test importing models_config and verify new models
        import models_config
        if 'qwen2.5-coder' not in models_config.MODELS_CONFIG:
            print(f"❌ qwen2.5-coder not found in models_config")
            return False
        else:
            print(f"✅ qwen2.5-coder model found in Python config")
            
        if models_config.MODELS_CONFIG['phi']['name'] != 'Phi-4 Mini':
            print(f"❌ Phi model not updated to Phi-4 Mini")
            return False
        else:
            print(f"✅ Phi model updated to: {models_config.MODELS_CONFIG['phi']['name']}")
        
        # Test importing the updated RAG integration
        from ollama_rag_integration import OllamaRAGIntegration, create_integrated_client
        print(f"✅ Successfully imported OllamaRAGIntegration and create_integrated_client")
        
        # Test importing task_classifier
        import task_classifier
        print(f"✅ Successfully imported task_classifier")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Python imports: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_instantiation():
    """Test that VLLMClient can be instantiated"""
    print("\n🔍 Testing VLLMClient instantiation...")
    
    try:
        sys.path.insert(0, '/home/elect/capibara6/backend')
        from ollama_client import VLLMClient
        
        # Load config to test client
        with open('/home/elect/capibara6/model_config.json', 'r') as f:
            config = json.load(f)
        
        # Try to create client
        client = VLLMClient(config)
        print(f"✅ Successfully created VLLMClient instance")
        print(f"   - Endpoint: {client.endpoint}")
        print(f"   - Models: {list(client.models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing VLLMClient instantiation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Testing vLLM Migration Changes\n")
    
    tests = [
        test_model_config,
        test_python_imports,
        test_client_instantiation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✅ All tests passed! vLLM migration is ready for deployment.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)