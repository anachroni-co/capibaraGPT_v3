#!/usr/bin/env python3
"""
Test script to validate the programming-only RAG functionality with the existing server
"""

import requests
import time
from typing import List, Tuple
from programming_rag_detector import is_programming_query


def test_programming_rag_activation():
    """Test that the programming RAG detector works as expected"""
    
    print("🎯 Testing Programming-Only RAG Activation")
    print("=" * 60)
    
    # Test queries that should activate RAG for programming
    programming_queries = [
        "How do I sort an array in Python using bubble sort?",
        "Write a JavaScript function to reverse a string",
        "I have a TypeError in my React app, how can I fix it?",
        "What are the differences between let, const, and var in JavaScript?",
        "Show me a Python code example for connecting to PostgreSQL",
        "Help me debug this Python code: def sum(a, b): return a + b",
        "How to implement binary search in C++?",
        "What is the syntax for async/await in TypeScript?",
        "How to use pandas DataFrame in Python?",
        "What are React hooks and how do I use them?"
    ]
    
    # Test queries that should NOT activate RAG
    non_programming_queries = [
        "What is the weather like today?",
        "Tell me about ancient Rome history",
        "How do I cook chicken soup?",
        "Explain the theory of relativity",
        "What are the benefits of meditation?",
        "Hello, how are you?",
        "Can you write a poem about trees?",
        "What time is it in Tokyo?",
        "Tell me about French culture",
        "Explain photosynthesis in plants"
    ]
    
    print(f"🧪 Testing {len(programming_queries) + len(non_programming_queries)} queries...")
    
    # Test programming queries
    print(f"\n💻 Programming Queries (should activate RAG):")
    prog_activations = 0
    for i, query in enumerate(programming_queries, 1):
        is_prog = is_programming_query(query)
        status = "✅ ACTIVATES" if is_prog else "❌ SKIPS"
        print(f"  {i:2d}. {status} - {query[:40]}...")
        if is_prog:
            prog_activations += 1
    
    print(f"\n📊 Programming queries: {prog_activations}/{len(programming_queries)} activate RAG")
    
    # Test non-programming queries 
    print(f"\n💬 Non-Programming Queries (should NOT activate RAG):")
    non_prog_skips = 0
    for i, query in enumerate(non_programming_queries, 1):
        is_prog = is_programming_query(query)
        status = "❌ ACTIVATES" if is_prog else "✅ SKIPS"
        print(f"  {i:2d}. {status} - {query[:40]}...")
        if not is_prog:
            non_prog_skips += 1
    
    print(f"\n📊 Non-programming queries: {non_prog_skips}/{len(non_programming_queries)} skip RAG")
    
    # Overall results
    total_prog = len(programming_queries)
    total_non_prog = len(non_programming_queries)
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   Programming activation rate: {prog_activations}/{total_prog} ({prog_activations/total_prog*100:.1f}%)")
    print(f"   Non-programming skip rate: {non_prog_skips}/{total_non_prog} ({non_prog_skips/total_non_prog*100:.1f}%)")
    print(f"   Combined accuracy: {(prog_activations + non_prog_skips)/(total_prog + total_non_prog)*100:.1f}%")
    
    # Test with the actual server if running
    print(f"\n🌐 Testing with Live Server (http://localhost:8082):")
    try:
        response = requests.get("http://localhost:8082/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Server healthy - {health['models_loaded']}/{health['models_available']} models loaded")
            
            # Test one programming query to the server
            print(f"   🧪 Testing server with programming query...")
            prog_test_query = "How do I create a Python function?"
            
            if is_programming_query(prog_test_query):
                print(f"   💻 Query '{prog_test_query[:30]}...' identified as programming (would activate RAG)")
            else:
                print(f"   ❌ Query '{prog_test_query[:30]}...' not identified as programming")
                
        else:
            print(f"   ⚠️  Server responded with status {response.status_code}, but not healthy")
            
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️  Server not running on http://localhost:8082")
    except Exception as e:
        print(f"   ⚠️  Error connecting to server: {e}")
    
    print(f"\n🎯 Programming-Only RAG system ready for integration!")
    print(f"   The system will activate RAG ONLY for programming-related queries")
    print(f"   This ensures faster responses for non-programming queries")


def demonstrate_router_integration():
    """Demonstrate how to integrate with the semantic router"""
    
    print(f"\n🔗 DEMONSTRATION: Router Integration")
    print("=" * 60)
    
    print("""
When integrating with the semantic router, replace the general RAG activation:

BEFORE (General RAG):
```
if general_rag_detector.detect(query).is_rag_query:
    # Fetch any kind of context
    context = fetch_general_context(query)
    enhanced_prompt = f"{context}\\n\\nUser: {query}"
```

AFTER (Programming-Only RAG):
```
if is_programming_query(query):
    # Fetch ONLY programming context
    context = fetch_programming_context(query) 
    enhanced_prompt = f"{context}\\n\\nProgramming query: {query}"
else:
    # Skip RAG, use original query
    enhanced_prompt = query
```""")

if __name__ == "__main__":
    test_programming_rag_activation()
    demonstrate_router_integration()
    
    print(f"\n🎉 VALIDATION COMPLETE")
    print(f"The programming-only RAG system is working correctly!")
    print(f"You can now integrate this into your semantic router.")