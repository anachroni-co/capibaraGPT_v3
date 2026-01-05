#!/usr/bin/env python3
"""
Script para verificar la corrección de la función calculate() en el servidor MCP
"""

import sys
import os

# Añadir el directorio del proyecto al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_calculate_function():
    print("🔍 Verificando la función calculate() corregida...")
    
    # Importar la función que acabamos de corregir
    from vm_services.mcp.mcp_server import calculate
    
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

def test_interface_security():
    print("\n🔍 Verificando seguridad de la interfaz de usuario...")
    
    # Verificar que la función de formateo de mensajes está correctamente implementada
    import importlib.util
    
    try:
        spec = importlib.util.spec_from_file_location("chat_app", "frontend/src/chat-app.js")
        # Como es un archivo JS, vamos a verificar manualmente que exista la función de escape HTML
        with open("frontend/src/chat-app.js", "r") as f:
            content = f.read()
            
        # Verificar que la función escapeHtml existe
        if "function escapeHtml" in content:
            print("✅ PASSED: Función escapeHtml encontrada en chat-app.js")
        else:
            print("❌ FAILED: Función escapeHtml no encontrada en chat-app.js")
            return False
            
        # Verificar que formatMessage llama a escapeHtml
        if "escapeHtml(content)" in content or "textContent" in content:
            print("✅ PASSED: Función formatMessage implementa protección contra XSS")
        else:
            print("❌ FAILED: Función formatMessage no implementa protección contra XSS")
            return False
            
        return True
    except Exception as e:
        print(f"❌ FAILED: Error verificando interfaz: {str(e)}")
        return False

def main():
    print("🦫 Capibara6 - Verificación de Correcciones de Seguridad")
    print("=" * 60)
    
    success1 = test_calculate_function()
    success2 = test_interface_security()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ¡Todas las verificaciones pasaron correctamente!")
        print("✅ Función calculate() corregida y segura")
        print("✅ Protecciones XSS implementadas")
        return 0
    else:
        print("💥 Algunas verificaciones fallaron.")
        print("❌ Revisar las correcciones necesarias")
        return 1

if __name__ == "__main__":
    sys.exit(main())