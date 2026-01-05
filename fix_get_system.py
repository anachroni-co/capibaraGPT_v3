#!/usr/bin/env python3
"""
Corrección precisa del archivo codecache.py
"""

# Leer el archivo
with open('/home/elect/venv/lib/python3.11/site-packages/torch/_inductor/codecache.py', 'r') as f:
    lines = f.readlines()

# Encontrar la línea exacta donde empieza el problema (línea 205 en el original)
# La estructura debería ser:
# def get_system():
#     triton_version = None
#     try:
#         from triton.compiler.compiler import triton_key
#         # ...
#     except (ModuleNotFoundError, ImportError):
#         # ...
#     try:
#         system = {...}

start_line_idx = -1
for i, line in enumerate(lines):
    if 'def get_system() -> dict[str, Any]:' in line:
        start_line_idx = i
        break

if start_line_idx != -1:
    # Buscar la línea con el import original
    import_line_idx = -1
    for i in range(start_line_idx, min(start_line_idx + 20, len(lines))):
        if 'from triton.compiler.compiler import triton_key' in lines[i]:
            import_line_idx = i
            break

    # Reemplazar la sección con la estructura correcta
    if import_line_idx != -1:
        # Encontrar el nivel de indentación
        original_indent = len(lines[import_line_idx]) - len(lines[import_line_idx].lstrip())
        spaces = ' ' * original_indent
        
        # Crear el nuevo bloque
        new_section = [
            f'{spaces}triton_version = None\n',
            f'{spaces}try:\n',
            f'{spaces}    from triton.compiler.compiler import triton_key\n',
            f'{spaces}\n',
            f'{spaces}    # Use triton_key instead of triton.__version__ as the version\n',
            f'{spaces}    # is not updated with each code change\n',
            f'{spaces}    triton_version = triton_key()\n',
            f'{spaces}except (ModuleNotFoundError, ImportError):\n',
            f'{spaces}    # Para versiones recientes de Triton donde triton_key no existe\n',
            f'{spaces}    try:\n',
            f'{spaces}        import triton\n',
            f'{spaces}        triton_version = f"triton_{{triton.__version__}}"\n',
            f'{spaces}    except ImportError:\n',
            f'{spaces}        triton_version = None\n',
        ]
        
        # Encontrar hasta dónde llega el bloque original que queremos reemplazar
        # Buscamos la siguiente línea que no esté indentada con el mismo nivel
        end_idx = import_line_idx
        for i in range(import_line_idx, len(lines)):
            current_line = lines[i]
            if current_line.strip() and not current_line.startswith(' ') and not current_line.startswith('\t'):
                # Esta línea está en el mismo nivel que el exterior
                break
            elif current_line.strip() and len(current_line) - len(current_line.lstrip()) < original_indent and not current_line.startswith('#'):
                # Esta línea está en un nivel superior al bloque que queremos reemplazar
                end_idx = i
                break
            elif 'except ModuleNotFoundError:' in current_line and len(current_line) - len(current_line.lstrip()) == original_indent:
                # Hasta aquí llega el bloque original
                end_idx = i
                break
            end_idx = i
        
        # Reemplazar el bloque
        lines = lines[:import_line_idx] + new_section + lines[end_idx+1:]
        
        # Escribir el archivo modificado
        with open('/home/elect/venv/lib/python3.11/site-packages/torch/_inductor/codecache.py', 'w') as f:
            f.writelines(lines)
        
        print("✅ Corrección aplicada al archivo codecache.py")
    else:
        print("⚠️ No se encontró la línea import.")
else:
    print("⚠️ No se encontró la función get_system.")