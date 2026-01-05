#!/usr/bin/env python3
"""
Script de inicio para el Servidor Integrado Capibara6
Incluye: GPT-OSS-20B Proxy + Smart MCP + Coqui TTS
"""

import subprocess
import os
import sys

def main():
    print("🚀 Iniciando Servidor Integrado Capibara6...")
    print("📦 Componentes incluidos:")
    print("   • Proxy CORS para GPT-OSS-20B")
    print("   • Smart MCP (Contexto Inteligente)")
    print("   • Coqui TTS (Síntesis de Voz)")
    print()
    
    # Asegúrate de que el directorio actual sea el de 'backend'
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    
    # Ejecutar el servidor integrado
    print("🌐 Iniciando servidor en puerto 5000...")
    print("🔗 URLs disponibles:")
    print("   • Chat: http://localhost:5000/api/chat")
    print("   • Health: http://localhost:5000/health")
    print("   • MCP: http://localhost:5000/api/mcp/analyze")
    print("   • TTS: http://localhost:5000/api/tts/speak")
    print()
    
    try:
        subprocess.run([sys.executable, "capibara6_integrated_server.py"])
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error al iniciar el servidor: {e}")

if __name__ == '__main__':
    main()
