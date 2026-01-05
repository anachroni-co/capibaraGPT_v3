#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Guides - Sistema de guías de usuario para Capibara6.
"""

import logging
import os
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UserGuide:
    """Guía de usuario."""
    title: str
    description: str
    content: str
    category: str
    difficulty: str
    estimated_time: str

class UserGuideGenerator:
    """Generador de guías de usuario."""
    
    def __init__(self):
        self.guides: List[UserGuide] = []
        self._initialize_guides()
        logger.info("UserGuideGenerator inicializado")
    
    def _initialize_guides(self):
        """Inicializa las guías de usuario."""
        guides = [
            UserGuide(
                title="Getting Started",
                description="Guía de inicio rápido para nuevos usuarios",
                category="Getting Started",
                difficulty="Beginner",
                estimated_time="10 minutes",
                content=self._get_getting_started_content()
            ),
            UserGuide(
                title="API Integration",
                description="Cómo integrar la API de Capibara6",
                category="Development",
                difficulty="Intermediate",
                estimated_time="30 minutes",
                content=self._get_api_integration_content()
            ),
            UserGuide(
                title="Advanced Features",
                description="Características avanzadas del sistema",
                category="Advanced",
                difficulty="Advanced",
                estimated_time="45 minutes",
                content=self._get_advanced_features_content()
            )
        ]
        
        self.guides.extend(guides)
    
    def _get_getting_started_content(self) -> str:
        return """# Getting Started with Capibara6

## What is Capibara6?

Capibara6 is an advanced AI agent system with intelligent routing, ACE framework, E2B execution, and scalability features.

## Quick Start

### 1. Authentication

Get your API key from the dashboard and authenticate:

```bash
curl -X POST "https://api.capibara6.com/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{"username": "your_username", "password": "your_password"}'
```

### 2. Your First Query

```bash
curl -X POST "https://api.capibara6.com/api/v1/query" \\
  -H "Authorization: Bearer <your_token>" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Hello, how are you?"}'
```

### 3. Understanding the Response

The system returns:
- **routing_result**: Which model was selected
- **ace_result**: Enhanced context from ACE framework
- **e2b_result**: Code execution results (if applicable)
- **processing_time_ms**: How long it took to process

## Next Steps

- Explore the API documentation
- Try different query types
- Set up monitoring and alerts
- Configure your preferences

## Support

Need help? Contact us at support@capibara6.com
"""
    
    def _get_api_integration_content(self) -> str:
        return """# API Integration Guide

## Authentication Methods

### JWT Tokens
```python
import requests

headers = {
    'Authorization': 'Bearer <your_jwt_token>',
    'Content-Type': 'application/json'
}
```

### API Keys
```python
headers = {
    'X-API-Key': '<your_api_key>',
    'Content-Type': 'application/json'
}
```

## Python SDK Example

```python
from capibara6 import Capibara6Client

client = Capibara6Client(api_key='your_api_key')

# Simple query
response = client.query("How to create a Python function?")
print(response.result)

# Query with context
response = client.query(
    "Explain this code",
    context={"code": "def hello(): print('world')"},
    options={"max_tokens": 1000}
)
```

## JavaScript SDK Example

```javascript
const Capibara6 = require('capibara6');

const client = new Capibara6({
    apiKey: 'your_api_key'
});

// Simple query
client.query('How to create a JavaScript function?')
    .then(response => console.log(response.result))
    .catch(error => console.error(error));
```

## Error Handling

```python
try:
    response = client.query("Your query here")
except Capibara6Error as e:
    print(f"Error: {e.message}")
    print(f"Status Code: {e.status_code}")
```

## Rate Limiting

The API implements rate limiting:
- 100 requests/minute for API endpoints
- 200 requests/minute for GraphQL
- 10 batches/minute for batch processing

Handle rate limits gracefully:

```python
import time

try:
    response = client.query("Your query")
except RateLimitError as e:
    time.sleep(e.retry_after)
    response = client.query("Your query")
```
"""
    
    def _get_advanced_features_content(self) -> str:
        return """# Advanced Features Guide

## Intelligent Routing

The system automatically selects the best model for your query:

- **capibara6-20b**: For medium complexity tasks
- **capibara6-120b**: For complex reasoning and analysis

### Custom Routing

```python
response = client.query(
    "Complex analysis task",
    options={
        "force_model": "capibara6-120b",
        "complexity_threshold": 0.8
    }
)
```

## ACE Framework

The Adaptive Context Evolution framework enhances your queries:

### Context Awareness
```python
response = client.query(
    "Continue the conversation",
    context={
        "previous_messages": [...],
        "user_preferences": {...},
        "domain_knowledge": {...}
    }
)
```

### Playbook Integration
```python
response = client.query(
    "Debug this Python code",
    options={
        "playbook": "python_debugging",
        "awareness_level": "high"
    }
)
```

## E2B Code Execution

Execute code in secure sandboxes:

### Python Execution
```python
response = client.query(
    "Run this Python code: print('Hello World')",
    options={"execute_code": True}
)

print(response.e2b_result.output)
```

### Multi-language Support
- Python
- JavaScript
- SQL
- Bash

## Batch Processing

Process multiple queries efficiently:

```python
queries = [
    {"query": "Query 1", "priority": "high"},
    {"query": "Query 2", "priority": "medium"},
    {"query": "Query 3", "priority": "low"}
]

response = client.batch_process(queries)
```

## Monitoring and Alerts

Set up monitoring for your usage:

```python
# Get metrics
metrics = client.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average response time: {metrics.avg_response_time}ms")

# Set up alerts
client.set_alert(
    metric="response_time",
    threshold=2000,  # 2 seconds
    action="email"
)
```

## Cost Optimization

Monitor and optimize your costs:

```python
# Get cost breakdown
costs = client.get_cost_breakdown()
print(f"Daily cost: ${costs.daily_total}")

# Set budget alerts
client.set_budget_alert(
    daily_limit=100,  # $100/day
    alert_threshold=80  # Alert at 80%
)
```

## Best Practices

1. **Use appropriate context**: Provide relevant context for better results
2. **Batch similar queries**: Use batch processing for efficiency
3. **Monitor usage**: Set up alerts and monitor costs
4. **Handle errors gracefully**: Implement proper error handling
5. **Cache results**: Cache responses when appropriate
6. **Use streaming**: For long responses, use streaming endpoints

## Troubleshooting

### Common Issues

1. **Authentication errors**: Check your API key or JWT token
2. **Rate limiting**: Implement exponential backoff
3. **Timeout errors**: Increase timeout or use batch processing
4. **Memory errors**: Reduce query complexity or use smaller models

### Debug Mode

Enable debug mode for detailed logging:

```python
client = Capibara6Client(
    api_key='your_api_key',
    debug=True
)
```
"""
    
    def generate_guide_html(self, guide: UserGuide) -> str:
        """Genera HTML para una guía."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{guide.title} - Capibara6</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        .header {{
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .meta {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .meta span {{
            margin-right: 20px;
            font-weight: bold;
        }}
        pre {{
            background: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{guide.title}</h1>
        <p>{guide.description}</p>
    </div>
    
    <div class="meta">
        <span>Category: {guide.category}</span>
        <span>Difficulty: {guide.difficulty}</span>
        <span>Time: {guide.estimated_time}</span>
    </div>
    
    <div class="content">
        {self._markdown_to_html(guide.content)}
    </div>
</body>
</html>
        """
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Convierte Markdown básico a HTML."""
        html = markdown
        html = html.replace('```python', '<pre><code class="language-python">')
        html = html.replace('```javascript', '<pre><code class="language-javascript">')
        html = html.replace('```bash', '<pre><code class="language-bash">')
        html = html.replace('```', '</code></pre>')
        html = html.replace('`', '<code>')
        html = html.replace('\n# ', '\n<h1>')
        html = html.replace('\n## ', '\n<h2>')
        html = html.replace('\n### ', '\n<h3>')
        html = html.replace('\n#### ', '\n<h4>')
        html = html.replace('\n**', '\n<strong>')
        html = html.replace('**', '</strong>')
        html = html.replace('\n- ', '\n<li>')
        html = html.replace('\n', '<br>\n')
        return html
    
    def save_guides(self, output_dir: str = "docs/guides"):
        """Guarda todas las guías."""
        os.makedirs(output_dir, exist_ok=True)
        
        for guide in self.guides:
            # HTML
            html_content = self.generate_guide_html(guide)
            filename = guide.title.lower().replace(' ', '_').replace('&', 'and')
            with open(os.path.join(output_dir, f"{filename}.html"), 'w') as f:
                f.write(html_content)
            
            # Markdown
            with open(os.path.join(output_dir, f"{filename}.md"), 'w') as f:
                f.write(f"# {guide.title}\n\n{guide.content}")
        
        # Índice
        index_content = self._generate_index()
        with open(os.path.join(output_dir, "index.html"), 'w') as f:
            f.write(index_content)
        
        logger.info(f"Guías guardadas en: {output_dir}")
    
    def _generate_index(self) -> str:
        """Genera el índice de guías."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Capibara6 User Guides</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .guide { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .meta { color: #666; font-size: 0.9em; }
        a { color: #007acc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Capibara6 User Guides</h1>
    <p>Comprehensive guides to help you get the most out of Capibara6.</p>
"""
        
        for guide in self.guides:
            filename = guide.title.lower().replace(' ', '_').replace('&', 'and')
            html += f"""
    <div class="guide">
        <h2><a href="{filename}.html">{guide.title}</a></h2>
        <p>{guide.description}</p>
        <div class="meta">
            Category: {guide.category} | 
            Difficulty: {guide.difficulty} | 
            Time: {guide.estimated_time}
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html


# Instancia global
user_guide_generator = UserGuideGenerator()


def get_user_guide_generator() -> UserGuideGenerator:
    """Obtiene la instancia global del generador de guías."""
    return user_guide_generator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = UserGuideGenerator()
    generator.save_guides()
    
    print(f"Guías generadas: {len(generator.guides)}")
    print("Sistema de guías de usuario funcionando correctamente!")

