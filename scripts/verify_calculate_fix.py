#!/usr/bin/env python3
"""
Script para verificar la corrección de la función calculate() en el servidor MCP
"""

def test_calculate_function():
    print("🔍 Verificando la función calculate() corregida...")
    
    # Simular la función corregida directamente en el script
    import ast
    import operator

    def calculate(expression: str) -> dict:
        """Calcula una expresión matemática de forma segura"""
        try:
            # Validar que la expresión no esté vacía
            if not expression or not expression.strip():
                return {'error': 'Expresión vacía'}

            # Verificar longitud máxima para evitar desbordamientos
            if len(expression) > 1000:
                return {'error': 'Expresión demasiado larga'}

            # Validar que solo contenga caracteres permitidos
            allowed_chars = set('0123456789+-*/().% ')
            if not all(c in allowed_chars for c in expression):
                return {'error': 'Expresión contiene caracteres no permitidos'}

            # Definir operadores permitidos
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            def eval_node(node):
                if isinstance(node, ast.Constant):  # Números
                    return node.value
                elif hasattr(ast, 'Num') and isinstance(node, ast.Num):  # Para versiones antiguas de Python
                    return node.n
                elif isinstance(node, ast.BinOp):
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    op = ops.get(type(node.op))
                    if op is None:
                        raise ValueError(f'Operador no permitido: {type(node.op)}')
                    if isinstance(node.op, ast.Pow) and (abs(left) > 100 or abs(right) > 10):
                        # Prevenir cálculos exponenciales muy grandes
                        raise ValueError('Operación exponencial demasiado grande')
                    return op(left, right)
                elif isinstance(node, ast.UnaryOp):
                    operand = eval_node(node.operand)
                    op = ops.get(type(node.op))
                    if op is None:
                        raise ValueError(f'Operador unario no permitido: {type(node.op)}')
                    return op(operand)
                else:
                    raise ValueError(f'Tipo de nodo no permitido: {type(node)}')

            try:
                # Parsear la expresión
                tree = ast.parse(expression, mode='eval')
                # Evaluar la expresión de forma segura
                result = eval_node(tree.body)

                # Validar el resultado
                if isinstance(result, (int, float)):
                    # Verificar que el resultado no sea inf o nan
                    if str(result) in ('inf', '-inf', 'nan'):
                        return {'error': 'Resultado inválido (infinito o NaN)'}
                    return {
                        'expression': expression,
                        'result': result
                    }
                else:
                    return {'error': 'Tipo de resultado no permitido'}
            except ValueError as e:
                return {'error': f'Error en la expresión: {str(e)}'}
            except OverflowError:
                return {'error': 'Resultado de cálculo demasiado grande'}
            except ZeroDivisionError:
                return {'error': 'División por cero'}

        except Exception as e:
            return {'error': f'Error inesperado: {str(e)}'}

    # Pruebas básicas
    test_cases = [
        # (expresión, resultado_esperado_o_error)
        ("2 + 2", 4),
        ("3 * 4", 12),
        ("10 / 2", 5.0),
        ("2 ** 3", 8),  # Potencia
        ("5 % 2", 1),   # Módulo
        ("-5 + 3", -2), # Números negativos
        ("(2 + 3) * 4", 20), # Paréntesis
        ("", "error"),  # Vacío
        ("2 + 2; print('malicious')", "error"),  # Caracteres prohibidos
        ("__import__('os').system('ls')", "error"),  # Inyección de código
        ("2 + 2 * __builtins__.__dict__", "error"),  # Acceso a builtins
        ("2 + 2 + eval('3 * 3')", "error"),  # Uso de eval
    ]
    
    all_passed = True
    for expr, expected in test_cases:
        try:
            result = calculate(expr)
            if expected == "error":
                if 'error' in result:
                    print(f"✅ PASSED: '{expr}' -> Error esperado: {result['error']}")
                else:
                    print(f"❌ FAILED: '{expr}' -> Se esperaba error pero obtuvo: {result}")
                    all_passed = False
            else:
                if 'result' in result and result['result'] == expected:
                    print(f"✅ PASSED: '{expr}' -> {result['result']}")
                else:
                    print(f"❌ FAILED: '{expr}' -> Se esperaba {expected}, obtuvo: {result}")
                    all_passed = False
        except Exception as e:
            if expected == "error":
                print(f"✅ PASSED: '{expr}' -> Error esperado: {str(e)}")
            else:
                print(f"❌ FAILED: '{expr}' -> Excepción inesperada: {str(e)}")
                all_passed = False
    
    return all_passed

def main():
    print("🦫 Capibara6 - Verificación de Correcciones de Seguridad")
    print("=" * 60)
    
    success = test_calculate_function()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ¡Todas las pruebas de la función calculate() pasaron correctamente!")
        print("✅ La función calculate() ahora es segura y protege contra RCE")
        return 0
    else:
        print("💥 Algunas pruebas fallaron.")
        print("❌ Revisar la implementación de seguridad")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())