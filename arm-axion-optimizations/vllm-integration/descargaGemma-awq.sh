cd /home/elect/models

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Descargando Gemma 3 27B AWQ INT4 (quantizado)                     ║"
echo "║  Mejoras esperadas:                                                ║"
echo "║  - Velocidad: 0.7 → 3-5 tokens/s (4-7x más rápido)               ║"
echo "║  - Memoria: 51GB → 13GB (4x menos)                                ║"
echo "║  - Tamaño: ~13-14GB descarga                                      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Descargar modelo AWQ
hf download gaunernst/gemma-3-27b-it-int4-awq \
  --local-dir gemma-3-27b-it-awq \
 

echo ""
echo "✅ Descarga completada"
du -sh gemma-3-27b-it-awq/
