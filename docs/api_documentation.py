#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Documentation - Sistema de documentación automática de APIs.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

@dataclass
class APIEndpoint:
    """Documentación de un endpoint de API."""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    tags: List[str]
    deprecated: bool = False
    security: Optional[List[Dict[str, Any]]] = None

@dataclass
class APISchema:
    """Esquema de un modelo de datos."""
    name: str
    type: str
    properties: Dict[str, Any]
    required: List[str]
    description: Optional[str] = None
    example: Optional[Dict[str, Any]] = None

class APIDocumentationGenerator:
    """Generador de documentación de API."""
    
    def __init__(self):
        self.endpoints: List[APIEndpoint] = []
        self.schemas: List[APISchema] = []
        self._initialize_api_documentation()
        
        logger.info("APIDocumentationGenerator inicializado")
    
    def _initialize_api_documentation(self):
        """Inicializa la documentación de la API."""
        
        # Esquemas de datos
        self._initialize_schemas()
        
        # Endpoints de la API
        self._initialize_endpoints()
        
        logger.info(f"Documentación inicializada: {len(self.endpoints)} endpoints, {len(self.schemas)} schemas")
    
    def _initialize_schemas(self):
        """Inicializa los esquemas de datos."""
        schemas = [
            APISchema(
                name="QueryRequest",
                type="object",
                properties={
                    "query": {"type": "string", "description": "La consulta del usuario"},
                    "context": {"type": "object", "description": "Contexto adicional"},
                    "options": {"type": "object", "description": "Opciones de procesamiento"}
                },
                required=["query"],
                description="Request para procesar una consulta",
                example={
                    "query": "How to create a Python function?",
                    "context": {"user_id": "user_123"},
                    "options": {"max_tokens": 1000}
                }
            ),
            APISchema(
                name="QueryResponse",
                type="object",
                properties={
                    "query": {"type": "string"},
                    "routing_result": {"$ref": "#/components/schemas/RoutingResult"},
                    "ace_result": {"$ref": "#/components/schemas/ACEResult"},
                    "e2b_result": {"$ref": "#/components/schemas/E2BResult"},
                    "processing_time_ms": {"type": "number"},
                    "timestamp": {"type": "string", "format": "date-time"}
                },
                required=["query", "processing_time_ms", "timestamp"],
                description="Response de una consulta procesada"
            ),
            APISchema(
                name="RoutingResult",
                type="object",
                properties={
                    "model_20b_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "model_120b_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "selected_model": {"type": "string", "enum": ["capibara6-20b", "capibara6-120b"]},
                    "reasoning": {"type": "string"}
                },
                required=["selected_model", "reasoning"],
                description="Resultado del routing inteligente"
            ),
            APISchema(
                name="ACEResult",
                type="object",
                properties={
                    "enhanced_context": {"type": "string"},
                    "awareness_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "playbook_used": {"type": "string"}
                },
                required=["enhanced_context", "awareness_score"],
                description="Resultado del framework ACE"
            ),
            APISchema(
                name="E2BResult",
                type="object",
                properties={
                    "execution_success": {"type": "boolean"},
                    "output": {"type": "string"},
                    "error": {"type": "string"},
                    "execution_time_ms": {"type": "number"}
                },
                required=["execution_success", "execution_time_ms"],
                description="Resultado de ejecución E2B"
            ),
            APISchema(
                name="ModelInfo",
                type="object",
                properties={
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "max_tokens": {"type": "integer"},
                    "capabilities": {"type": "array", "items": {"type": "string"}}
                },
                required=["id", "name", "description", "max_tokens", "capabilities"],
                description="Información de un modelo"
            ),
            APISchema(
                name="SystemMetrics",
                type="object",
                properties={
                    "timestamp": {"type": "string", "format": "date-time"},
                    "system": {
                        "type": "object",
                        "properties": {
                            "uptime_seconds": {"type": "number"},
                            "environment": {"type": "string"},
                            "version": {"type": "string"}
                        }
                    },
                    "components": {"type": "object"}
                },
                required=["timestamp", "system"],
                description="Métricas del sistema"
            ),
            APISchema(
                name="BatchRequest",
                type="object",
                properties={
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                            },
                            "required": ["query"]
                        }
                    }
                },
                required=["queries"],
                description="Request para procesamiento en batch"
            ),
            APISchema(
                name="ErrorResponse",
                type="object",
                properties={
                    "error": {"type": "string"},
                    "status_code": {"type": "integer"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "details": {"type": "object"}
                },
                required=["error", "status_code", "timestamp"],
                description="Response de error"
            )
        ]
        
        self.schemas.extend(schemas)
    
    def _initialize_endpoints(self):
        """Inicializa los endpoints de la API."""
        endpoints = [
            APIEndpoint(
                path="/health",
                method="GET",
                summary="Health Check",
                description="Verifica el estado de salud del sistema",
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Sistema saludable",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                        "version": {"type": "string"},
                                        "environment": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Health"]
            ),
            APIEndpoint(
                path="/health/detailed",
                method="GET",
                summary="Detailed Health Check",
                description="Verifica el estado detallado de todos los componentes",
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Estado detallado del sistema",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "components": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Health"]
            ),
            APIEndpoint(
                path="/api/v1/query",
                method="POST",
                summary="Process Query",
                description="Procesa una consulta usando el sistema completo de Capibara6",
                parameters=[],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/QueryRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Query procesada exitosamente",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/QueryResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Request inválido",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    },
                    "500": {
                        "description": "Error interno del servidor",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                },
                tags=["Query Processing"],
                security=[{"BearerAuth": []}]
            ),
            APIEndpoint(
                path="/api/v1/models",
                method="GET",
                summary="Get Available Models",
                description="Obtiene información de los modelos disponibles",
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Lista de modelos disponibles",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "models": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/ModelInfo"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Models"],
                security=[{"BearerAuth": []}]
            ),
            APIEndpoint(
                path="/api/v1/metrics",
                method="GET",
                summary="Get System Metrics",
                description="Obtiene métricas del sistema",
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Métricas del sistema",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SystemMetrics"}
                            }
                        }
                    }
                },
                tags=["Metrics"],
                security=[{"BearerAuth": []}]
            ),
            APIEndpoint(
                path="/api/v1/batch",
                method="POST",
                summary="Process Batch Queries",
                description="Procesa múltiples consultas en batch",
                parameters=[],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/BatchRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Batch procesado exitosamente",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "batch_id": {"type": "string"},
                                        "results": {"type": "array"},
                                        "timestamp": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Batch Processing"],
                security=[{"BearerAuth": []}]
            ),
            APIEndpoint(
                path="/api/v1/cache/stats",
                method="GET",
                summary="Get Cache Statistics",
                description="Obtiene estadísticas del caché",
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Estadísticas del caché",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_entries": {"type": "integer"},
                                        "hit_rate": {"type": "number"},
                                        "l1_utilization": {"type": "number"},
                                        "l2_utilization": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Cache"],
                security=[{"BearerAuth": []}]
            ),
            APIEndpoint(
                path="/api/v1/cache",
                method="DELETE",
                summary="Clear Cache",
                description="Limpia el caché del sistema",
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Caché limpiado exitosamente",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Cache"],
                security=[{"BearerAuth": []}]
            )
        ]
        
        self.endpoints.extend(endpoints)
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Genera la especificación OpenAPI 3.0."""
        openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "Capibara6 API",
                "description": "Advanced AI Agent System with Intelligent Routing, ACE, E2B, and Scalability",
                "version": "1.0.0",
                "contact": {
                    "name": "Capibara6 Team",
                    "email": "support@capibara6.com",
                    "url": "https://capibara6.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "https://api.capibara6.com",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.capibara6.com",
                    "description": "Staging server"
                },
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                        "description": "JWT token for authentication"
                    },
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                        "description": "API key for authentication"
                    }
                }
            },
            "tags": [
                {"name": "Health", "description": "Health check endpoints"},
                {"name": "Query Processing", "description": "Query processing endpoints"},
                {"name": "Models", "description": "Model information endpoints"},
                {"name": "Metrics", "description": "System metrics endpoints"},
                {"name": "Batch Processing", "description": "Batch processing endpoints"},
                {"name": "Cache", "description": "Cache management endpoints"}
            ]
        }
        
        # Añadir paths
        for endpoint in self.endpoints:
            if endpoint.path not in openapi_spec["paths"]:
                openapi_spec["paths"][endpoint.path] = {}
            
            openapi_spec["paths"][endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated
            }
            
            if endpoint.parameters:
                openapi_spec["paths"][endpoint.path][endpoint.method.lower()]["parameters"] = endpoint.parameters
            
            if endpoint.request_body:
                openapi_spec["paths"][endpoint.path][endpoint.method.lower()]["requestBody"] = endpoint.request_body
            
            if endpoint.responses:
                openapi_spec["paths"][endpoint.path][endpoint.method.lower()]["responses"] = endpoint.responses
            
            if endpoint.security:
                openapi_spec["paths"][endpoint.path][endpoint.method.lower()]["security"] = endpoint.security
        
        # Añadir schemas
        for schema in self.schemas:
            openapi_spec["components"]["schemas"][schema.name] = {
                "type": schema.type,
                "properties": schema.properties,
                "required": schema.required
            }
            
            if schema.description:
                openapi_spec["components"]["schemas"][schema.name]["description"] = schema.description
            
            if schema.example:
                openapi_spec["components"]["schemas"][schema.name]["example"] = schema.example
        
        return openapi_spec
    
    def generate_swagger_ui_html(self) -> str:
        """Genera HTML para Swagger UI."""
        openapi_spec = self.generate_openapi_spec()
        spec_json = json.dumps(openapi_spec, indent=2)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Capibara6 API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin:0;
            background: #fafafa;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                spec: {spec_json},
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                requestInterceptor: function(request) {{
                    // Añadir headers de autenticación si están disponibles
                    const token = localStorage.getItem('capibara6_token');
                    if (token) {{
                        request.headers['Authorization'] = 'Bearer ' + token;
                    }}
                    return request;
                }}
            }});
            
            window.ui = ui;
        }};
    </script>
</body>
</html>
        """
        
        return html_template
    
    def generate_postman_collection(self) -> Dict[str, Any]:
        """Genera una colección de Postman."""
        collection = {
            "info": {
                "name": "Capibara6 API",
                "description": "Advanced AI Agent System API",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
                "version": "1.0.0"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{jwt_token}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "https://api.capibara6.com",
                    "type": "string"
                },
                {
                    "key": "jwt_token",
                    "value": "",
                    "type": "string"
                }
            ],
            "item": []
        }
        
        for endpoint in self.endpoints:
            item = {
                "name": endpoint.summary,
                "request": {
                    "method": endpoint.method,
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json",
                            "type": "text"
                        }
                    ],
                    "url": {
                        "raw": "{{base_url}}" + endpoint.path,
                        "host": ["{{base_url}}"],
                        "path": endpoint.path.strip("/").split("/")
                    },
                    "description": endpoint.description
                }
            }
            
            if endpoint.request_body:
                item["request"]["body"] = {
                    "mode": "raw",
                    "raw": json.dumps(endpoint.request_body.get("content", {}).get("application/json", {}).get("schema", {}), indent=2),
                    "options": {
                        "raw": {
                            "language": "json"
                        }
                    }
                }
            
            collection["item"].append(item)
        
        return collection
    
    def save_documentation(self, output_dir: str = "docs/api"):
        """Guarda toda la documentación en archivos."""
        os.makedirs(output_dir, exist_ok=True)
        
        # OpenAPI spec
        openapi_spec = self.generate_openapi_spec()
        with open(os.path.join(output_dir, "openapi.json"), 'w') as f:
            json.dump(openapi_spec, f, indent=2)
        
        with open(os.path.join(output_dir, "openapi.yaml"), 'w') as f:
            yaml.dump(openapi_spec, f, default_flow_style=False)
        
        # Swagger UI HTML
        swagger_html = self.generate_swagger_ui_html()
        with open(os.path.join(output_dir, "swagger-ui.html"), 'w') as f:
            f.write(swagger_html)
        
        # Postman collection
        postman_collection = self.generate_postman_collection()
        with open(os.path.join(output_dir, "postman_collection.json"), 'w') as f:
            json.dump(postman_collection, f, indent=2)
        
        # Documentación en Markdown
        markdown_doc = self.generate_markdown_documentation()
        with open(os.path.join(output_dir, "README.md"), 'w') as f:
            f.write(markdown_doc)
        
        logger.info(f"Documentación de API guardada en: {output_dir}")
    
    def generate_markdown_documentation(self) -> str:
        """Genera documentación en Markdown."""
        markdown = f"""# Capibara6 API Documentation

## Overview

Capibara6 es un sistema avanzado de agentes AI con routing inteligente, framework ACE, ejecución E2B y escalabilidad.

## Base URL

- **Production**: `https://api.capibara6.com`
- **Staging**: `https://staging-api.capibara6.com`
- **Development**: `http://localhost:8000`

## Authentication

La API utiliza autenticación JWT. Incluye el token en el header:

```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

"""
        
        # Agrupar endpoints por tag
        endpoints_by_tag = {}
        for endpoint in self.endpoints:
            for tag in endpoint.tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)
        
        for tag, endpoints in endpoints_by_tag.items():
            markdown += f"### {tag}\n\n"
            
            for endpoint in endpoints:
                markdown += f"#### {endpoint.method} {endpoint.path}\n\n"
                markdown += f"**{endpoint.summary}**\n\n"
                markdown += f"{endpoint.description}\n\n"
                
                if endpoint.parameters:
                    markdown += "**Parameters:**\n\n"
                    for param in endpoint.parameters:
                        markdown += f"- `{param.get('name', 'N/A')}` ({param.get('in', 'N/A')}): {param.get('description', 'N/A')}\n"
                    markdown += "\n"
                
                if endpoint.request_body:
                    markdown += "**Request Body:**\n\n"
                    markdown += "```json\n"
                    markdown += json.dumps(endpoint.request_body, indent=2)
                    markdown += "\n```\n\n"
                
                markdown += "**Responses:**\n\n"
                for status_code, response in endpoint.responses.items():
                    markdown += f"- `{status_code}`: {response.get('description', 'N/A')}\n"
                
                markdown += "\n---\n\n"
        
        # Añadir ejemplos
        markdown += """## Examples

### Process Query

```bash
curl -X POST "https://api.capibara6.com/api/v1/query" \\
  -H "Authorization: Bearer <your_token>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "How to create a Python function?",
    "context": {"user_id": "user_123"},
    "options": {"max_tokens": 1000}
  }'
```

### Get Models

```bash
curl -X GET "https://api.capibara6.com/api/v1/models" \\
  -H "Authorization: Bearer <your_token>"
```

### Get Metrics

```bash
curl -X GET "https://api.capibara6.com/api/v1/metrics" \\
  -H "Authorization: Bearer <your_token>"
```

## Error Handling

La API utiliza códigos de estado HTTP estándar:

- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

Los errores incluyen un objeto JSON con detalles:

```json
{
  "error": "Error message",
  "status_code": 400,
  "timestamp": "2023-11-20T12:00:00Z",
  "details": {}
}
```

## Rate Limiting

La API implementa rate limiting:

- **API**: 100 requests por minuto por IP
- **GraphQL**: 200 requests por minuto por IP
- **Batch**: 10 batches por minuto por IP

Los headers de rate limiting incluyen:

- `X-RateLimit-Limit`: Límite de requests
- `X-RateLimit-Remaining`: Requests restantes
- `X-RateLimit-Reset`: Timestamp de reset

## Support

Para soporte técnico:

- **Email**: support@capibara6.com
- **Documentation**: https://docs.capibara6.com
- **GitHub**: https://github.com/capibara6/capibara6
"""
        
        return markdown


# Instancia global
api_doc_generator = APIDocumentationGenerator()


def get_api_documentation_generator() -> APIDocumentationGenerator:
    """Obtiene la instancia global del generador de documentación."""
    return api_doc_generator


if __name__ == "__main__":
    # Test del generador de documentación
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    generator = APIDocumentationGenerator()
    
    # Generar documentación
    generator.save_documentation()
    
    # Mostrar información
    print(f"Endpoints documentados: {len(generator.endpoints)}")
    print(f"Schemas documentados: {len(generator.schemas)}")
    
    # Generar especificación OpenAPI
    openapi_spec = generator.generate_openapi_spec()
    print(f"OpenAPI spec generada: {len(json.dumps(openapi_spec))} caracteres")
    
    print("Generador de documentación de API funcionando correctamente!")
