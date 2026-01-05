#!/usr/bin/env python3
"""
Interactive Router Test for Capibara6 System
Connects to the backend and tests queries against different models
"""

import sys
import os
import json
import time

# Add backend path to import the client
sys.path.insert(0, '/home/elect/capibara6/backend')

try:
    from ollama_client import VLLMClient
    from models_config import get_active_models, get_model_config
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("This indicates the vLLM client and model configuration are properly set up in the system.")
    print("\nThe Capibara6 system has the following models configured:")
    print("- phi4:mini (fast/simple responses)")
    print("- qwen2.5-coder-1.5b (technical/coding responses)")
    print("- gpt-oss:20b (complex analysis)")
    sys.exit(1)

class InteractiveRouterTester:
    def __init__(self):
        # Load model configuration
        with open('/home/elect/capibara6/model_config.json', 'r') as f:
            config = json.load(f)
        
        # Create VLLM client
        self.client = VLLMClient(config)
        self.models = config['models']
        
    def test_single_model(self, prompt: str, model_key: str):
        """Test a single model with the given prompt"""
        try:
            print(f"Sending query to {model_key}...")
            
            start_time = time.time()
            result = self.client.generate(prompt, model_key)
            end_time = time.time()
            
            return {
                "status": "success",
                "response": result,
                "model": model_key,
                "response_time": end_time - start_time,
                "model_config": self.models[model_key]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": model_key,
                "response_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def test_with_all_models(self, prompt: str):
        """Test the same prompt with all available models"""
        results = {}
        
        for model_key in self.models.keys():
            result = self.test_single_model(prompt, model_key)
            results[model_key] = result
            print(f"Model {model_key}: {result['status']}")
            
        return results
    
    def run_interactive_test(self):
        """Run the interactive test session"""
        print("=" * 80)
        print("Capibara6 Router Test - Interactive Mode")
        print("=" * 80)
        print(f"Available models: {list(self.models.keys())}")
        print("\nModel descriptions:")
        for key, config in self.models.items():
            print(f"  - {key}: {config['description']} (use_case: {config['use_case'][:2]})")
        
        print("\n" + "=" * 80)
        
        while True:
            print("\nOptions:")
            print("1. Test single query with all models")
            print("2. Test specific model")
            print("3. Run predefined test cases")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                prompt = input("Enter your query: ").strip()
                if not prompt:
                    print("Empty query, skipping...")
                    continue
                
                print(f"\nTesting query with all models: '{prompt}'")
                results = self.test_with_all_models(prompt)
                
                print("\n" + "="*60)
                print("RESULTS:")
                print("="*60)
                
                for model_key, result in results.items():
                    print(f"\n--- {model_key} ---")
                    if result['status'] == 'success':
                        print(f"✅ Success in {result['response_time']:.2f}s")
                        print(f"Response: {result['response'][:200]}...")
                        if len(result['response']) > 200:
                            print(f"(truncated from {len(result['response'])} chars)")
                    else:
                        print(f"❌ Error: {result['error']}")
            
            elif choice == "2":
                print(f"Available models: {list(self.models.keys())}")
                model_choice = input("Enter model name: ").strip()
                
                if model_choice not in self.models:
                    print(f"Invalid model. Available: {list(self.models.keys())}")
                    continue
                
                prompt = input("Enter your query: ").strip()
                if not prompt:
                    print("Empty query, skipping...")
                    continue
                
                result = self.test_single_model(prompt, model_choice)
                
                print(f"\n--- {model_choice} ---")
                if result['status'] == 'success':
                    print(f"✅ Success in {result['response_time']:.2f}s")
                    print(f"Response: {result['response']}")
                else:
                    print(f"❌ Error: {result['error']}")
            
            elif choice == "3":
                test_cases = [
                    ("Simple math", "What is 2+2?"),
                    ("Technical question", "How do I reverse a string in Python?"),
                    ("Complex analysis", "Explain the differences between various deep learning frameworks and their use cases."),
                    ("Creative writing", "Write a short story about an AI assistant helping a developer.")
                ]
                
                for test_name, test_query in test_cases:
                    print(f"\n--- Testing: {test_name} -> '{test_query}' ---")
                    results = self.test_with_all_models(test_query)
                    
                    for model_key, result in results.items():
                        print(f"  {model_key}: ", end="")
                        if result['status'] == 'success':
                            print(f"✅ ({result['response_time']:.2f}s)")
                            print(f"    Preview: {result['response'][:100]}...")
                        else:
                            print(f"❌ Error: {result['error']}")
            
            elif choice == "4":
                print("\nExiting router test...")
                break
            
            else:
                print("Invalid choice, please try again.")

def main():
    print("Initializing Capibara6 Router Test...")
    
    try:
        tester = InteractiveRouterTester()
        print("✅ Router test initialized successfully!")
        print(f"Available models: {list(tester.models.keys())}")
        
        tester.run_interactive_test()
        
    except Exception as e:
        print(f"❌ Error initializing router test: {e}")
        print("\nThis could be due to:")
        print("1. vLLM server not running at the configured endpoint")
        print("2. Network connectivity issues")
        print("3. Model service unavailable")
        print(f"\nThe system is configured to use endpoint: http://34.12.166.76:8000/v1")
        print("\nTo start the vLLM server, run:")
        print("  cd /home/elect/capibara6/arm-axion-optimizations/vllm-integration")
        print("  python3 multi_model_server.py --config config.production.json --host 0.0.0.0 --port 8080")

if __name__ == "__main__":
    main()