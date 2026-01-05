#!/usr/bin/env python3
"""
Capibara6 Router System - Demonstration
Shows how the routing system works conceptually
"""

import json
import time
from typing import Dict, List

class Capibara6RouterDemo:
    def __init__(self):
        # Load model configurations
        with open('/home/elect/capibara6/model_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.models = self.config['models']
        
        # Map internal keys to actual model names
        self.model_key_to_name = {key: details['name'] for key, details in self.models.items()}
        self.model_name_to_key = {details['name']: key for key, details in self.models.items()}
        
        # Define routing logic based on internal keys
        self.routing_rules = {
            'fast_response': {
                'keywords': ['simple', 'basic', 'what is', 'hello', 'hi', 'math', 'calculate', 'quick', 'fast', 'short', 'how are you', '2+2'],
                'use_cases': ['preguntas simples', 'respuestas rápidas', 'chistes', 'saludos', 'respuestas directas'],
                'description': 'Fast response model for simple queries'
            },
            'balanced': {
                'keywords': ['code', 'python', 'javascript', 'program', 'function', 'debug', 'error', 'technical', 'algorithm', 'programming', 'explain', 'how to'],
                'use_cases': ['explicaciones', 'análisis intermedio', 'redacción', 'resumen corto', 'programación', 'análisis técnico'],
                'description': 'Technical model for coding and intermediate analysis'
            },
            'complex': {
                'keywords': ['analyze', 'complex', 'deep', 'detailed', 'research', 'think step', 'comprehensive', 'long', 'compare', 'evaluate', 'strategy'],
                'use_cases': ['análisis profundo', 'razonamiento complejo', 'planificación', 'análisis técnico'],
                'description': 'Complex model for deep analysis and reasoning'
            }
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query and determine which model should handle it"""
        query_lower = query.lower()
        
        scores = {}
        for model_key, rules in self.routing_rules.items():
            score = 0
            
            # Score based on keywords
            for keyword in rules['keywords']:
                if keyword in query_lower:
                    score += 2
            
            # Additional scoring based on query length and complexity
            if len(query) < 30:
                if model_key == 'fast_response':
                    score += 1
            elif len(query) > 100:
                if model_key in ['balanced', 'complex']:
                    score += 2
            
            # Check for technical terms
            if any(tech_term in query_lower for tech_term in ['code', 'function', 'debug', 'algorithm', 'python', 'javascript']):
                if model_key == 'balanced':
                    score += 3
            elif any(tech_term in query_lower for tech_term in ['analyze', 'research', 'deep', 'compare', 'strategy']):
                if model_key == 'complex':
                    score += 3
            
            scores[model_key] = score
        
        # Determine best model
        best_model = max(scores, key=scores.get) if scores else list(self.models.keys())[0]
        
        return {
            'input_query': query,
            'scores': scores,
            'predicted_model': best_model,
            'model_details': self.models[best_model],
            'routing_rules_used': self.routing_rules[best_model]
        }
    
    def run_demonstration(self):
        """Run the demonstration showing how routing works"""
        print("=" * 90)
        print("Capibara6 Router System - DEMONSTRATION")
        print("=" * 90)
        print("This shows how queries are intelligently routed to the most appropriate model")
        print("based on content analysis, complexity, and the model's specialization.\n")
        
        print("Available Models:")
        for key, model in self.models.items():
            print(f"  • {key} ({model['name']}): {model['description']}")
            print(f"    Max tokens: {model['max_tokens']}, Timeout: {model['timeout']}ms")
            print(f"    Use cases: {model['use_case']}")
        print()
        
        print("DEMONSTRATION QUERIES:")
        print("-" * 50)
        
        # Test queries
        test_queries = [
            "What is 2+2?",
            "How do I reverse a string in Python?",
            "Explain quantum computing in simple terms",
            "Write a JavaScript function to sort an array",
            "Analyze the impact of AI on healthcare",
            "Debug this error: TypeError: Cannot read property 'map' of undefined",
            "Hello, how are you?",
            "Compare different deep learning frameworks"
        ]
        
        for i, query in enumerate(test_queries, 1):
            result = self.analyze_query(query)
            
            print(f"\n{i}. Query: \"{query}\"")
            print(f"   Routing Analysis:")
            for model, score in result['scores'].items():
                marker = " → " if model == result['predicted_model'] else "   "
                model_name = self.model_key_to_name[model]
                print(f"{marker}  {model} ({model_name}): {score} points")
            
            chosen_model_name = self.model_key_to_name[result['predicted_model']]
            print(f"   → Routed to: {result['predicted_model']} ({chosen_model_name}) - {result['routing_rules_used']['description']}")
            
            # Show why it was chosen
            reason = self._explain_routing(result['input_query'], result['predicted_model'])
            print(f"   → Reason: {reason}")
        
        print("\n" + "=" * 90)
        print("INTERACTIVE MODE - Try your own queries (type 'quit' to exit)")
        print("=" * 90)
        
        while True:
            user_query = input("\nEnter your query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q', '']:
                print("Thanks for testing the Capibara6 Router System!")
                break
            
            result = self.analyze_query(user_query)
            
            chosen_model_name = self.model_key_to_name[result['predicted_model']]
            print(f"\nQuery: \"{result['input_query']}\"")
            print("Model Scores:")
            for model, score in result['scores'].items():
                model_name = self.model_key_to_name[model]
                marker = " → " if model == result['predicted_model'] else "   "
                print(f"{marker}  {model} ({model_name}): {score} points")
            
            print(f"\nRouted to: {result['predicted_model']} ({chosen_model_name})")
            print(f"Description: {result['model_details']['description']}")
            print(f"Max tokens: {result['model_details']['max_tokens']}")
            print(f"Use cases: {result['model_details']['use_case']}")
            
            reason = self._explain_routing(result['input_query'], result['predicted_model'])
            print(f"Routing reason: {reason}")
    
    def _explain_routing(self, query: str, model: str) -> str:
        """Explain why a query was routed to a specific model"""
        query_lower = query.lower()
        rules = self.routing_rules[model]
        
        reasons = []
        
        # Check which keywords matched
        for keyword in rules['keywords']:
            if keyword in query_lower:
                reasons.append(f"'{keyword}' keyword matched")
        
        # Check query length
        if len(query) < 30 and model == 'fast_response':
            reasons.append("short query suitable for fast response")
        elif len(query) > 100 and model in ['balanced', 'complex']:
            reasons.append("long query needs detailed response")
        
        if not reasons:
            reasons.append("general match based on query analysis")
        
        return ", ".join(reasons)


def main():
    print("Initializing Capibara6 Router System Demo...")
    time.sleep(0.5)  # Simulate loading
    
    try:
        router = Capibara6RouterDemo()
        router.run_demonstration()
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        print("The system is properly configured with:")
        print("- Model routing rules")
        print("- Configuration at /home/elect/capibara6/model_config.json")
        print("- Available models: fast_response (phi4:mini), balanced (qwen2.5-coder-1.5b), complex (gpt-oss:20b)")


if __name__ == "__main__":
    main()