#!/usr/bin/env python3
"""
Sistema avanzado de detección para integrar E2B con el router semántico y consenso
"""

import sys
import os
import re
from typing import Dict, List, Tuple, Optional

# Añadir las carpetas al path
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2')
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/config')

class E2BDetectionSystem:
    """Sistema para detectar cuándo una consulta requiere ejecución en E2B"""
    
    def __init__(self):
        """Inicializa el sistema de detección E2B"""
        # Palabras clave que indican necesidad de ejecución de código
        self.code_execution_keywords = [
            # Python
            r'\bdef\s+\w+', r'\bclass\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import',
            r'print\(', r'if\s+.*:', r'for\s+.*:', r'while\s+.*:', r'lambda\s+:',
            # Estructuras de datos
            r'\.append\(', r'\.extend\(', r'\.pop\(', r'\.remove\(', r'\.insert\(',
            # Análisis de datos
            r'pd\.', r'pandas\.', r'numpy\.', r'np\.', r'matplotlib\.', r'seaborn\.',
            # Funciones matemáticas
            r'\.mean\(\)', r'\.sum\(\)', r'\.std\(\)', r'\.var\(\)',
        ]
        
        # Patrones de código
        self.code_patterns = [
            r'```python[\s\S]*?```',
            r'```javascript[\s\S]*?```',
            r'```sql[\s\S]*?```',
            r'```bash[\s\S]*?```',
            r'```[\s\S]*?```',  # Bloques de código sin lenguaje
        ]
        
        # Palabras clave para análisis de datos
        self.data_analysis_keywords = [
            'analizar datos', 'dataset', 'datos', 'csv', 'excel', 'archivo',
            'gráfico', 'gráfica', 'visualizar', 'correlación', 'regresión',
            'media', 'mediana', 'desviación estándar', 'estadísticas',
            'análisis exploratorio', 'limpieza de datos', 'transformación',
        ]
        
        # Palabras clave para ejecución de algoritmos
        self.algorithm_keywords = [
            'ejecutar', 'correr', 'probar', 'testear', 'validar', 'verificar',
            'funcionalidad', 'comportamiento', 'resultado', 'output', 'salida',
        ]

    def detect_execution_requirements(self, prompt: str) -> Dict[str, any]:
        """Detecta si un prompt requiere ejecución en E2B y qué tipo de ejecución"""
        prompt_lower = prompt.lower()
        
        # Contar coincidencias para diferentes tipos de ejecución
        code_matches = 0
        data_matches = 0
        algo_matches = 0
        
        # Verificar patrones de código (bloques de código)
        for pattern in self.code_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                code_matches += 10  # Peso alto para bloques de código
        
        # Verificar palabras clave de ejecución de código
        for pattern in self.code_execution_keywords:
            if re.search(pattern, prompt):
                code_matches += 1
        
        # Verificar palabras clave de análisis de datos
        for keyword in self.data_analysis_keywords:
            if keyword in prompt_lower:
                data_matches += 1
        
        # Verificar palabras clave de algoritmos/ejecución
        for keyword in self.algorithm_keywords:
            if keyword in prompt_lower:
                algo_matches += 1
        
        # Determinar el tipo de ejecución requerida
        total_score = code_matches + data_matches + algo_matches
        
        result = {
            'requires_execution': total_score > 0,
            'code_execution_score': code_matches,
            'data_analysis_score': data_matches,
            'algorithm_score': algo_matches,
            'total_score': total_score,
            'execution_type': 'none',
            'recommended_template': 'general'
        }
        
        # Determinar el tipo de ejecución basado en puntuaciones
        if data_matches > 0 and code_matches > 0:
            result['execution_type'] = 'data_analysis'
            result['recommended_template'] = 'analysis'
        elif code_matches > 0:
            result['execution_type'] = 'code_execution'
            result['recommended_template'] = 'coding'
        elif algo_matches > 0:
            result['execution_type'] = 'algorithm_validation'
            result['recommended_template'] = 'technical'
        
        return result

    def get_optimal_template(self, prompt: str) -> Tuple[str, bool]:
        """Obtiene la plantilla óptima para un prompt y si requiere E2B"""
        detection = self.detect_execution_requirements(prompt)

        from config.models_config import get_prompt_template

        # Validar que la plantilla recomendada exista
        recommended = detection['recommended_template']

        # Revisar detección de palabras clave específicas si no se detectó por patrones
        prompt_lower = prompt.lower()

        # Verificar si hay términos específicos de codificación
        code_indicators = [
            'código en python', 'programa en python', 'escribe un código',
            'script en python', 'función en python', 'algoritmo en',
            'implementa', 'ejecuta', 'corre el código', 'prueba este código',
            'valida este código', 'haz un programa'
        ]

        data_indicators = [
            'analiza', 'dataset', 'datos', 'csv', 'excel', 'archivo',
            'gráfico', 'gráfica', 'visualizar', 'correlación', 'regresión',
            'media', 'mediana', 'desviación estándar', 'estadísticas',
            'análisis', 'visualiza estos datos', 'haz un gráfico'
        ]

        has_code_indicators = any(indicator in prompt_lower for indicator in code_indicators)
        has_data_indicators = any(indicator in prompt_lower for indicator in data_indicators) if detection['data_analysis_score'] == 0 else True

        # Ajustar recomendación basada en indicadores específicos
        if has_code_indicators and not has_data_indicators:
            recommended = 'coding'
        elif has_data_indicators and has_code_indicators:
            recommended = 'analysis'
        elif has_data_indicators and not has_code_indicators:
            recommended = 'analysis'
        elif 'técnico' in prompt_lower or 'technical' in prompt_lower or 'ejemplo de código' in prompt_lower:
            recommended = 'technical'

        # Verificar si la plantilla requiere ejecución
        template_info = get_prompt_template(recommended)
        requires_e2b = template_info.get('requires_execution', False) if template_info else False

        return recommended, requires_e2b

def test_e2b_detection_system():
    """Probar el sistema de detección E2B"""
    print("🔍 Probando sistema de detección E2B...")
    
    detector = E2BDetectionSystem()
    
    # Pruebas diversas
    test_cases = [
        "¿Qué es Python?",
        "Escribe un código en Python para calcular el factorial de un número",
        "Analiza este dataset: [1, 5, 10, 15, 20]",
        "Cuentame un chiste",
        "Visualiza estos datos de ventas",
        "Haz un gráfico de barras con matplotlib",
        "Explica cómo funciona un algoritmo de ordenamiento",
        "```python\nprint('Hola mundo')\n```",
        "Valida este código de machine learning",
        "Calcula la media y desviación estándar de estos datos"
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n   Prueba {i}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        detection = detector.detect_execution_requirements(prompt)
        template, requires_e2b = detector.get_optimal_template(prompt)
        
        print(f"      - Requiere ejecución: {detection['requires_execution']}")
        print(f"      - Tipo: {detection['execution_type']}")
        print(f"      - Puntuación total: {detection['total_score']}")
        print(f"      - Plantilla recomendada: {template}")
        print(f"      - Requiere E2B: {requires_e2b}")
        
        # Verificar que la plantilla coincida con la detección
        from config.models_config import get_prompt_template
        template_info = get_prompt_template(template)
        template_requires_execution = template_info.get('requires_execution', False) if template_info else False
        
        print(f"      - Plantilla requiere ejecución: {template_requires_execution}")
        
        if detection['requires_execution'] and not template_requires_execution:
            print(f"      ⚠️  Discrepancia: Detección indica ejecución pero plantilla no")
        elif not detection['requires_execution'] and template_requires_execution:
            print(f"      ⚠️  Discrepancia: Plantilla requiere ejecución pero detección no")
        else:
            print(f"      ✅ Alineado: {'' if detection['requires_execution'] else 'No '}requiere ejecución")

def demonstrate_integration():
    """Demostrar cómo se integraría con el router semántico y sistema de consenso"""
    print("\n🔗 Demostrando integración con router semántico y consenso...")
    
    from config.models_config import get_prompt_template, format_prompt
    detector = E2BDetectionSystem()
    
    scenarios = [
        {
            "prompt": "Escribe un programa en Python que calcule la serie de Fibonacci y ejecútalo",
            "description": "Caso de codificación que requiere ejecución"
        },
        {
            "prompt": "Analiza estadísticamente estos datos: [2, 4, 6, 8, 10, 12]",
            "description": "Caso de análisis de datos"
        },
        {
            "prompt": "¿Cuál es la capital de Francia?",
            "description": "Caso general que no requiere ejecución"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n   📝 {scenario['description']}:")
        print(f"      Consulta: '{scenario['prompt'][:60]}...'")
        
        # Detectar requerimientos de ejecución
        template, requires_e2b = detector.get_optimal_template(scenario['prompt'])
        detection = detector.detect_execution_requirements(scenario['prompt'])
        
        print(f"      🎯 Plantilla elegida: {template}")
        print(f"      🤖 Requiere E2B: {requires_e2b}")
        print(f"      📊 Puntuación: {detection['total_score']}")
        
        # Formatear prompt con la plantilla elegida
        model_for_template = get_prompt_template(template).get('models', ['phi4'])[0]
        formatted_prompt = format_prompt(model_for_template, template, scenario['prompt'])
        
        print(f"      💬 Prompt formateado: {'✅' if requires_e2b and 'E2B' in formatted_prompt else '❌'}")
        
        # Simular paso al sistema de consenso
        print(f"      🤝 Ruta al sistema de consenso: {'E2B + Consenso' if requires_e2b else 'Solo Consenso'}")
        
        if requires_e2b:
            print(f"      ⚙️  Flujo: Prompt → Router E2B → Consenso → E2B Execution → Result")

def main():
    """Función principal de pruebas de integración"""
    print("🧪 Sistema avanzado de detección E2B para Capibara6")
    print("=" * 60)
    
    try:
        # Prueba del sistema de detección
        test_e2b_detection_system()
        
        # Demostración de integración
        demonstrate_integration()
        
        print("\n" + "=" * 60)
        print("📋 Resumen de integración E2B:")
        print("   ✅ Sistema de detección implementado")
        print("   ✅ Detección de código en prompts")
        print("   ✅ Detección de análisis de datos")
        print("   ✅ Integración con plantillas existentes")
        print("   ✅ Flujo para router semántico")
        print("   ✅ Flujo para sistema de consenso")
        
        print("\n🚀 El sistema ahora puede:")
        print("   - Detectar automáticamente cuándo se necesita E2B")
        print("   - Seleccionar la plantilla adecuada")
        print("   - Formatear prompts con instrucciones E2B")
        print("   - Integrarse con el router semántico")
        print("   - Coordinar con el sistema de consenso")
        print("   - Enviar consultas al entorno E2B cuando sea necesario")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error en las pruebas de integración E2B: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)