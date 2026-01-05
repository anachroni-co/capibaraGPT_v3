#!/bin/bash
# Fix Windows line endings in all shell scripts

echo "Corrigiendo fin de línea en scripts..."

for file in *.sh; do
    if [ -f "$file" ]; then
        sed -i 's/\r$//' "$file"
        echo "  ✓ $file"
    fi
done

echo ""
echo "✓ Todos los scripts corregidos"
echo ""
echo "Ahora puedes ejecutar:"
echo "  ./start-capibara6.sh"
