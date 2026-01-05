#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de inicio para capibara6 con integración MCP
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path

def start_flask_server():
    """Iniciar servidor Flask con MCP integrado"""
    print("🚀 Iniciando servidor Flask con MCP...")
    
    # Cambiar al directorio del backend
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Puerto para Railway (usa variable de entorno PORT)
    port = int(os.getenv('PORT', 5000))
    
    # Iniciar servidor Flask con MCP
    try:
        from mcp_server import app
        print(f"✅ Servidor Flask con MCP iniciado en puerto {port}")
        print(f"🌐 URL: http://localhost:{port}")
        print(f"📚 Documentación MCP: http://localhost:{port}/mcp")
        print(f"🔧 API MCP: http://localhost:{port}/api/mcp/")
        
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        sys.exit(1)

def start_mcp_standalone():
    """Iniciar solo el conector MCP (para testing)"""
    print("🧪 Iniciando conector MCP standalone...")
    
    try:
        import asyncio
        from mcp_connector import Capibara6MCPConnector
        
        async def test_connector():
            connector = Capibara6MCPConnector()
            
            # Test básico
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            }
            
            response = await connector.handle_request(init_request)
            print("✅ Conector MCP inicializado correctamente")
            print(f"📊 Capacidades: {list(response.get('result', {}).get('capabilities', {}).keys())}")
        
        asyncio.run(test_connector())
        
    except Exception as e:
        print(f"❌ Error en conector MCP: {e}")
        sys.exit(1)

def run_tests():
    """Ejecutar tests del conector MCP"""
    print("🧪 Ejecutando tests del conector MCP...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "test_mcp.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Tests completados exitosamente")
            print(result.stdout)
        else:
            print("❌ Tests fallaron")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error ejecutando tests: {e}")

def show_help():
    """Mostrar ayuda"""
    print("""
🦫 capibara6 MCP Connector - Script de Inicio

Uso: python start_mcp.py [comando]

Comandos disponibles:
  server     - Iniciar servidor Flask con MCP integrado (por defecto)
  standalone - Iniciar solo el conector MCP para testing
  test       - Ejecutar tests del conector MCP
  help       - Mostrar esta ayuda

Ejemplos:
  python start_mcp.py server
  python start_mcp.py standalone
  python start_mcp.py test

Variables de entorno:
  PORT       - Puerto del servidor (por defecto: 5000)
  DEBUG      - Modo debug (por defecto: False)

Para más información:
  📚 Documentación: https://modelcontextprotocol.io
  🌐 Web: https://capibara6.com
  📧 Email: info@anachroni.co
""")

def main():
    """Función principal"""
    command = sys.argv[1] if len(sys.argv) > 1 else "server"
    
    print("🦫 capibara6 MCP Connector")
    print("=" * 40)
    
    if command == "server":
        start_flask_server()
    elif command == "standalone":
        start_mcp_standalone()
    elif command == "test":
        run_tests()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Comando desconocido: {command}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()