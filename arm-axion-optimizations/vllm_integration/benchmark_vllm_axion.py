#!/usr/bin/env python3
"""
Benchmark script for vLLM with ARM Axion optimizations
Measures performance improvements for Qwen2.5, Phi4-mini, Gemma3-27b and Mistral7B models
"""

import time
import asyncio
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import statistics

# Configuration
CONFIG = {
    "base_url": "http://localhost:8080",  # Default vLLM server
    "models": [
        {
            "name": "phi4_fast",
            "endpoint": "/v1/chat/completions",
            "description": "Phi-4 Mini optimized for ARM"
        },
        {
            "name": "qwen_coder",
            "endpoint": "/v1/chat/completions",
            "description": "Qwen2.5-Coder 1.5B optimized for ARM"
        },
        {
            "name": "gemma3_multimodal",
            "endpoint": "/v1/chat/completions",
            "description": "Gemma3 27B multimodal model"
        },
        {
            "name": "mistral_balanced",
            "endpoint": "/v1/chat/completions",
            "description": "Mistral 7B balanced model"
        }
    ],
    "test_prompts": [
        {
            "name": "simple_query",
            "prompt": "What is the capital of France?",
            "max_tokens": 100
        },
        {
            "name": "technical_query", 
            "prompt": "Explain how ARM NEON SIMD instructions improve performance in machine learning inference.",
            "max_tokens": 200
        },
        {
            "name": "complex_query",
            "prompt": "Compare the architectural differences between ARM Neoverse V1 and V2 processors, focusing on improvements in floating-point performance and memory bandwidth.",
            "max_tokens": 300
        },
        {
            "name": "coding_query",
            "prompt": "Write a Python function that implements ARM NEON optimized vector addition using ctypes.",
            "max_tokens": 250
        }
    ],
    "num_iterations": 10,
    "concurrent_requests": [1, 2, 4, 8]  # Different concurrency levels to test
}


def run_single_request(model_config: Dict[str, Any], prompt_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single inference request and measure performance"""
    url = f"{CONFIG['base_url']}{model_config['endpoint']}"
    
    payload = {
        "model": model_config['name'],
        "messages": [
            {"role": "user", "content": prompt_config['prompt']}
        ],
        "max_tokens": prompt_config['max_tokens'],
        "temperature": 0.7
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            total_time = end_time - start_time
            
            # Calculate tokens per second
            if 'usage' in result:
                output_tokens = result['usage']['completion_tokens']
            else:
                # Estimate tokens from response text
                output_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                output_tokens = len(output_text.split())
                
            tokens_per_second = output_tokens / total_time if total_time > 0 else 0
            
            return {
                "status": "success",
                "total_time": total_time,
                "tokens_generated": output_tokens,
                "tokens_per_second": tokens_per_second,
                "response": result
            }
        else:
            return {
                "status": "error",
                "error_code": response.status_code,
                "error_message": response.text,
                "total_time": end_time - start_time
            }
    except Exception as e:
        end_time = time.time()
        return {
            "status": "exception",
            "error": str(e),
            "total_time": end_time - start_time
        }


def benchmark_model_concurrent(model_config: Dict[str, Any], prompt_config: Dict[str, Any], 
                               concurrency: int) -> List[Dict[str, Any]]:
    """Run benchmark for a model with specified concurrency"""
    print(f"  Testing {model_config['name']} with {concurrency} concurrent requests...")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_single_request, model_config, prompt_config) 
                   for _ in range(concurrency)]
        
        results = [future.result() for future in futures]
    
    return results


def run_benchmark():
    """Run comprehensive benchmark for all models and configurations"""
    print("="*80)
    print("vLLM ARM Axion Optimization Benchmark")
    print("="*80)
    print(f"Models: {[m['name'] for m in CONFIG['models']]}")
    print(f"Prompts: {[p['name'] for p in CONFIG['test_prompts']]}")
    print(f"Iterations per test: {CONFIG['num_iterations']}")
    print(f"Concurrency levels: {CONFIG['concurrent_requests']}")
    print("="*80)
    
    all_results = {}
    
    for model_config in CONFIG['models']:
        print(f"\nBenchmarking model: {model_config['name']} ({model_config['description']})")
        print("-" * 60)
        
        model_results = {}
        
        for prompt_config in CONFIG['test_prompts']:
            print(f"  Prompt: {prompt_config['name']} ({prompt_config['prompt'][:50]}...)")
            
            prompt_results = {}
            
            for concurrency in CONFIG['concurrent_requests']:
                times = []
                throughput_values = []
                
                # Run multiple iterations for statistical significance
                for i in range(min(CONFIG['num_iterations'], 5)):  # Limit iterations for time
                    results = benchmark_model_concurrent(model_config, prompt_config, concurrency)
                    
                    successful_results = [r for r in results if r['status'] == 'success']
                    
                    if successful_results:
                        avg_time = statistics.mean([r['total_time'] for r in successful_results])
                        avg_throughput = statistics.mean([r['tokens_per_second'] for r in successful_results if r['tokens_per_second'] > 0])
                        
                        times.append(avg_time)
                        throughput_values.append(avg_throughput)
                
                if times and throughput_values:
                    prompt_results[f"concurrency_{concurrency}"] = {
                        "avg_response_time": statistics.mean(times),
                        "std_response_time": statistics.stdev(times) if len(times) > 1 else 0,
                        "avg_throughput": statistics.mean(throughput_values),
                        "std_throughput": statistics.stdev(throughput_values) if len(throughput_values) > 1 else 0,
                        "samples": len(times)
                    }
            
            model_results[prompt_config['name']] = prompt_results
        
        all_results[model_config['name']] = model_results
    
    return all_results


def print_benchmark_summary(results: Dict[str, Any]):
    """Print a formatted summary of benchmark results"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 40)
        
        for prompt_name, prompt_results in model_results.items():
            print(f"  {prompt_name}:")
            
            for concurrency_key, metrics in prompt_results.items():
                concurrency = concurrency_key.split('_')[1]
                avg_time = metrics['avg_response_time']
                avg_throughput = metrics['avg_throughput']
                
                print(f"    Concurrency {concurrency}: "
                      f"Response Time: {avg_time:.3f}s ± {metrics['std_response_time']:.3f}s, "
                      f"Throughput: {avg_throughput:.2f} tok/s ± {metrics['std_throughput']:.2f}")


def main():
    """Main function to run the benchmark"""
    try:
        results = run_benchmark()
        print_benchmark_summary(results)
        
        # Save detailed results to file
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to 'benchmark_results.json'")
        
        # Calculate and display key metrics
        print("\nKEY PERFORMANCE METRICS:")
        print("-" * 30)
        
        for model_name, model_results in results.items():
            # Calculate average throughput across all prompts and concurrency levels
            all_throughputs = []
            for prompt_results in model_results.values():
                for metrics in prompt_results.values():
                    all_throughputs.append(metrics['avg_throughput'])
            
            if all_throughputs:
                avg_throughput = statistics.mean(all_throughputs)
                print(f"{model_name}: Avg Throughput = {avg_throughput:.2f} tokens/sec")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()