#!/usr/bin/env python3
"""
Test script to demonstrate the routing logic without requiring a running server
Shows how queries would be routed to different models based on content analysis
"""

import re
from typing import Dict, List
import json

class MockRouter:
    def __init__(self):
        self.model_descriptions = {
            "phi4:mini": {
                "name": "Phi-4 Mini",
                "capabilities": ["simple questions", "quick responses", "general knowledge", "basic conversations"],
                "strengths": ["fast", "efficient", "lightweight"],
                "typical_queries": ["simple math", "basic info", "short answers", "greetings"]
            },
            "qwen2.5-coder:1.5b": {
                "name": "Qwen2.5 Coder",
                "capabilities": ["programming", "technical questions", "code generation", "explanations"],
                "strengths": ["coding", "technical", "detailed"],
                "typical_queries": ["code help", "programming", "technical concepts", "algorithm explanations"]
            },
            "gpt-oss:20b": {
                "name": "GPT-OSS 20B",
                "capabilities": ["complex analysis", "long-form content", "deep reasoning", "creative tasks"],
                "strengths": ["knowledge", "reasoning", "complexity"],
                "typical_queries": ["complex analysis", "long answers", "deep reasoning", "creative writing"]
            }
        }
        
        # Keywords to identify query types
        self.code_keywords = [
            'python', 'java', 'javascript', 'code', 'function', 'program', 'algorithm',
            'class', 'method', 'variable', 'debug', 'error', 'syntax', 'library',
            'framework', 'api', 'database', 'sql', 'html', 'css', 'javascript',
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask'
        ]
        
        self.tech_keywords = [
            'computer', 'technology', 'software', 'hardware', 'network', 'system',
            'algorithm', 'data', 'machine learning', 'ai', 'neural', 'model',
            'database', 'server', 'cloud', 'security', 'encryption', 'protocol'
        ]
        
        self.simple_keywords = [
            'hello', 'hi', 'what', 'how', 'why', 'when', 'where', 'who',
            'is', 'are', 'do', 'does', 'can', 'will', 'the', 'a', 'an',
            'simple', 'basic', 'easy', 'quick', 'fast'
        ]
        
        self.complex_keywords = [
            'analyze', 'compare', 'evaluate', 'examine', 'investigate', 'research',
            'complex', 'detailed', 'comprehensive', 'thorough', 'deep', 'advanced',
            'sophisticated', 'nuanced', 'multifaceted', 'challenging'
        ]

    def classify_query(self, query: str) -> str:
        """Classify query to determine appropriate model"""
        query_lower = query.lower()
        scores = {"phi4:mini": 0, "qwen2.5-coder:1.5b": 0, "gpt-oss:20b": 0}
        
        # Score based on keywords
        for keyword in self.code_keywords:
            if keyword in query_lower:
                scores["qwen2.5-coder:1.5b"] += 2
        
        for keyword in self.tech_keywords:
            if keyword in query_lower:
                scores["qwen2.5-coder:1.5b"] += 1
                scores["gpt-oss:20b"] += 1
        
        for keyword in self.simple_keywords:
            if keyword in query_lower:
                scores["phi4:mini"] += 1
        
        for keyword in self.complex_keywords:
            if keyword in query_lower:
                scores["gpt-oss:20b"] += 2
        
        # Score based on query length and complexity
        if len(query) < 20:
            scores["phi4:mini"] += 1
        elif len(query) > 100:
            scores["gpt-oss:20b"] += 1
        
        # Check if it's a math question
        if re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)|math|calculate|sum|multiply|divide', query_lower):
            if len(query) < 50:  # Simple math
                scores["phi4:mini"] += 1
            else:  # Complex math
                scores["gpt-oss:20b"] += 1
        
        # Determine model with highest score
        best_model = max(scores, key=scores.get)
        return best_model, scores

    def generate_mock_response(self, model: str, query: str) -> str:
        """Generate a mock response based on the model and query"""
        if model == "phi4:mini":
            responses = [
                f"I understand you're asking about '{query[:30]}...'. This is a straightforward question that I can answer quickly.",
                f"Regarding '{query[:25]}...', the answer is simple and concise.",
                f"Based on my knowledge, {query[:20]}... can be explained in a few words."
            ]
        elif model == "qwen2.5-coder:1.5b":
            responses = [
                f"Looking at your technical query about '{query[:30]}...', I can provide a detailed explanation.",
                f"For this programming/technical question on '{query[:25]}...', here's a comprehensive answer.",
                f"I'm analyzing your technical request regarding '{query[:20]}...' and here's the solution."
            ]
        else:  # gpt-oss:20b
            responses = [
                f"Your complex query about '{query[:30]}...' requires deep analysis and comprehensive explanation.",
                f"Examining this multifaceted question on '{query[:25]}...', I'll provide a detailed response.",
                f"Analyzing '{query[:20]}...' in depth, here's a thorough exploration of the topic."
            ]
        
        import random
        return random.choice(responses)

    def route_and_respond(self, query: str) -> Dict:
        """Route query to appropriate model and generate response"""
        model, scores = self.classify_query(query)
        
        return {
            "input_query": query,
            "predicted_model": model,
            "model_scores": scores,
            "model_info": self.model_descriptions[model],
            "mock_response": self.generate_mock_response(model, query),
            "confidence": max(scores.values()) / sum(scores.values()) if sum(scores.values()) > 0 else 0
        }

def main():
    router = MockRouter()
    
    print("=" * 80)
    print("Capibara6 Router Simulation")
    print("This demonstrates how queries would be routed to different models")
    print("=" * 80)
    
    test_queries = [
        "What is 2+2?",
        "How do I reverse a string in Python?",
        "Write a JavaScript function to sort an array",
        "Explain quantum computing",
        "What are the benefits of using React vs Angular?",
        "Can you help me debug this error: TypeError: Cannot read property 'map' of undefined?",
        "How does machine learning work?",
        "Tell me about deep learning architectures",
        "What's the weather like today?",
        "Compare different database systems"
    ]
    
    print(f"\nTesting {len(test_queries)} different types of queries:\n")
    
    for i, query in enumerate(test_queries, 1):
        result = router.route_and_respond(query)
        
        print(f"{i:2d}. Query: {query}")
        print(f"    → Routed to: {result['predicted_model']} ({result['model_info']['name']})")
        print(f"    → Scores: {result['model_scores']}")
        print(f"    → Confidence: {result['confidence']:.2f}")
        print(f"    → Response preview: {result['mock_response']}")
        print(f"    → Capabilities: {', '.join(result['model_info']['capabilities'])}")
        print()
    
    print("=" * 80)
    print("Interactive Mode - Type your own queries (or 'quit' to exit)")
    print("=" * 80)
    
    while True:
        user_query = input("\nEnter your query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        result = router.route_and_respond(user_query)
        
        print(f"\nQuery: {result['input_query']}")
        print(f"Routed to: {result['predicted_model']} ({result['model_info']['name']})")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Model capabilities: {', '.join(result['model_info']['capabilities'])}")
        print(f"Expected response: {result['mock_response']}")
        print(f"Score breakdown: {result['model_scores']}")

if __name__ == "__main__":
    main()