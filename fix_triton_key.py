#!/usr/bin/env python3
"""
Aplicar parche preciso al archivo codecache.py
"""

import re

# Leer el archivo
with open('/home/elect/venv/lib/python3.11/site-packages/torch/_inductor/codecache.py', 'r') as f:
    content = f.read()

# Buscar el bloque original y reemplazarlo
# Buscar: try: followed by import triton_key, comments, function call, except
start_marker = "from triton.compiler.compiler import triton_key"
end_marker = "except ModuleNotFoundError:"

# Dividir el contenido para localizar la sección específica
lines = content.split('\n')

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Ver si encontramos la línea que comienza el bloque a reemplazar
    if 'from triton.compiler.compiler import triton_key' in line and 'try:' in lines[i-1]:
        # Encontramos el inicio del bloque a reemplazar
        # Encontrar el final del bloque (hasta el except ModuleNotFoundError)
        indent = len(line) - len(line.lstrip())
        spaces = ' ' * indent
        
        # Construir el nuevo bloque
        new_logic = [
            f'{spaces}triton_version = None',
            f'{spaces}try:',
            f'{spaces}    from triton.compiler.compiler import triton_key',
            '',
            f'{spaces}    # Use triton_key instead of triton.__version__ as the version',
            f'{spaces}    # is not updated with each code change',
            f'{spaces}    triton_version = triton_key()',
            f'{spaces}except (ModuleNotFoundError, ImportError):',
            f'{spaces}    # Para versiones recientes de Triton donde triton_key no existe',
            f'{spaces}    try:',
            f'{spaces}        import triton',
            f'{spaces}        triton_version = f"triton_{{triton.__version__}}"',
            f'{spaces}    except ImportError:',
            f'{spaces}        triton_version = None'
        ]
        
        # Agregar el nuevo bloque
        new_lines.extend(new_logic)
        
        # Saltar las líneas originales del bloque
        j = i
        while j < len(lines) and not lines[j].strip().startswith('except ModuleNotFoundError'):
            j += 1
        # Saltar también la línea del except ModuleNotFoundError
        if j < len(lines) and lines[j].strip().startswith('except ModuleNotFoundError'):
            j += 1
            # Saltar la línea "triton_version = None" que le sigue
            if j < len(lines) and 'triton_version = None' in lines[j] and lines[j].startswith(' '):
                j += 1
        
        # Actualizar el índice
        i = j
    else:
        new_lines.append(line)
        i += 1

# Escribir el contenido modificado
with open('/home/elect/venv/lib/python3.11/site-packages/torch/_inductor/codecache.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("✅ Parche aplicado al archivo codecache.py")