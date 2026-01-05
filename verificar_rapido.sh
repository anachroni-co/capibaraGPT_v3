#!/bin/bash
# Script rápido para verificar solo lo esencial

PROJECT="mamba-001"

echo "🔍 Verificación Rápida de Servicios"
echo "===================================="
echo ""

# Verificar bounty2
echo "📡 bounty2..."
gcloud compute ssh bounty2 --zone=europe-west4-a --project="$PROJECT" \
    --command="ps aux | grep -E '(python|ollama)' | grep -v grep | wc -l && sudo ss -tuln 2>/dev/null | grep -E ':(5001|11434)' | wc -l" \
    2>&1 | tail -3

echo ""
echo "📡 rag3..."
gcloud compute ssh rag3 --zone=europe-west2-c --project="$PROJECT" \
    --command="ps aux | grep python | grep -v grep | wc -l && sudo ss -tuln 2>/dev/null | grep 8000 | wc -l" \
    2>&1 | tail -3

echo ""
echo "📡 gpt-oss-20b..."
gcloud compute ssh gpt-oss-20b --zone=europe-southwest1-b --project="$PROJECT" \
    --command="ps aux | grep python | grep -v grep | wc -l && sudo ss -tuln 2>/dev/null | grep -E ':(500[0-9]|5010|5678)' | wc -l" \
    2>&1 | tail -3

echo ""
echo "✅ Verificación rápida completada"

