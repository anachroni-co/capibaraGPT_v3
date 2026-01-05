#!/usr/bin/env python3
"""
Interactive Router Test Client for Capibara6
Allows testing the semantic router with different queries and seeing which model responds
"""

import requests
import json
import time
from typing import Dict, Any

class RouterTestClient:
    def __init__(self, base_url="http://34.12.166.76:8080"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY"  # vLLM default
        }
        
    def test_query(self, prompt: str, model: str = None, max_tokens: int = 200) -> Dict[str, Any]:
        """Test a query with the vLLM endpoint"""
        url = f"{self.base_url}/v1/chat/completions"
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": model if model else "phi4:mini",  # Default to fast model
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "response": result['choices'][0]['message']['content'],
                    "model_used": result.get('model', 'unknown'),
                    "tokens_used": result.get('usage', {}).get('total_tokens', 0),
                    "response_time": end_time - start_time,
                    "raw_response": result
                }
            else:
                return {
                    "status": "error",
                    "error_code": response.status_code,
                    "error_message": response.text,
                    "response_time": end_time - start_time
                }
        except requests.exceptions.RequestException as e:
            return {
                "status": "exception",
                "error": str(e),
                "response_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def test_with_all_models(self, prompt: str) -> Dict[str, Any]:
        """Test the same prompt with all available models"""
        models_to_test = [
            "phi4:mini",           # Fast/simple
            "qwen2.5-coder:1.5b",  # Coding/technical
            "gpt-oss:20b"          # Complex reasoning
        ]
        
        results = {}
        for model in models_to_test:
            print(f"Testing with {model}...")
            result = self.test_query(prompt, model=model)
            results[model] = result
            time.sleep(1)  # Brief pause between requests
        
        return results

def main():
    print("=" * 70)
    print("Capibara6 Router Test Client")
    print("=" * 70)
    print("This tool allows you to test queries and see which model responds")
    print("Available models: phi4:mini, qwen2.5-coder:1.5b, gpt-oss:20b")
    print("=" * 70)
    
    client = RouterTestClient()
    
    while True:
        print("\nChoose an option:")
        print("1. Test single query")
        print("2. Test with all models")
        print("3. Predefined test queries")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            prompt = input("\nEnter your query: ").strip()
            if not prompt:
                print("Empty query, skipping...")
                continue
                
            print(f"\nSending query: {prompt}")
            result = client.test_query(prompt)
            
            if result["status"] == "success":
                print(f"\n✅ SUCCESS")
                print(f"Model used: {result['model_used']}")
                print(f"Response time: {result['response_time']:.2f}s")
                print(f"Tokens used: {result['tokens_used']}")
                print(f"Response:\n{result['response']}")
            else:
                print(f"\n❌ ERROR: {result['error_message'] if 'error_message' in result else result['error']}")
        
        elif choice == "2":
            prompt = input("\nEnter your query: ").strip()
            if not prompt:
                print("Empty query, skipping...")
                continue
                
            print(f"\nTesting query with all models: {prompt}")
            results = client.test_with_all_models(prompt)
            
            print(f"\n📊 RESULTS COMPARISON:")
            for model, result in results.items():
                print(f"\n--- {model} ---")
                if result["status"] == "success":
                    print(f"✅ Success in {result['response_time']:.2f}s")
                    print(f"Tokens: {result['tokens_used']}")
                    print(f"Response: {result['response'][:100]}...")
                else:
                    print(f"❌ Error: {result.get('error_message', result.get('error', 'Unknown error'))}")
        
        elif choice == "3":
            print("\nPredefined test queries:")
            print("a) Simple question: 'What is 2+2?'")
            print("b) Technical query: 'Explain how a neural network works'")
            print("c) Coding query: 'Write a Python function to reverse a string'")
            print("d) Complex query: 'Compare the pros and cons of different deep learning frameworks'")
            
            sub_choice = input("\nChoose a test (a-d): ").strip().lower()
            
            queries = {
                'a': 'What is 2+2?',
                'b': 'Explain how a neural network works',
                'c': 'Write a Python function to reverse a string',
                'd': 'Compare the pros and cons of different deep learning frameworks'
            }
            
            if sub_choice in queries:
                prompt = queries[sub_choice]
                print(f"\nTesting: {prompt}")
                
                # Test with all models
                results = client.test_with_all_models(prompt)
                
                print(f"\n📊 RESULTS FOR: {prompt}")
                for model, result in results.items():
                    print(f"\n--- {model} ---")
                    if result["status"] == "success":
                        print(f"✅ Success in {result['response_time']:.2f}s")
                        print(f"Tokens: {result['tokens_used']}")
                        print(f"Response: {result['response'][:200]}...")
                    else:
                        print(f"❌ Error: {result.get('error_message', result.get('error', 'Unknown error'))}")
            else:
                print("Invalid choice!")
        
        elif choice == "4":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()