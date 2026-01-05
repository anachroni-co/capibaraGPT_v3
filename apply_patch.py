#!/usr/bin/env python3
"""
Script para aplicar el parche de triton_key de forma precisa
"""
import re

# Ruta del archivo
file_path = '/home/elect/venv/lib/python3.11/site-packages/torch/_inductor/codecache.py'

# Leer el archivo original
with open(file_path, 'r') as f:
    content = f.read()

# Definir el patrón para encontrar la sección específica
pattern = r'(\s+)try:\s*\n\s+from triton.compiler.compiler import triton_key\s*\n(\s+)# Use triton_key instead of triton.__version__ as the version\s*\n\s+# is not updated with each code change\s*\n\s+triton_version = triton_key\(\)\s*\n(\s+)except ModuleNotFoundError:\s*\n(\s+)triton_version = None'

# Reemplazo correcto
replacement = r'\1triton_version = None\n\1try:\n\2from triton.compiler.compiler import triton_key\n\3# Use triton_key instead of triton.__version__ as the version\n\4# is not updated with each code change\n\4triton_version = triton_key()\n\1except (ModuleNotFoundError, ImportError):\n\5# Para versiones recientes de Triton donde triton_key no existe\n\5try:\n\5    import triton\n\5    triton_version = f"triton_{triton.__version__}"\n\5except ImportError:\n\5    triton_version = None'

# Aplicar el parche
new_content = re.sub(pattern, replacement, content)

# Escribir el archivo modificado
with open(file_path, 'w') as f:
    f.write(new_content)

print("✅ Parche aplicado correctamente al archivo codecache.py")