#!/usr/bin/env python3
"""
Script simple para probar la conexión básica con cada modelo
en el sistema ARM-Axion multimodelo
"""

import requests
import json
import time
from typing import List

# Configuración
BASE_URL = "http://localhost:8080"
TEST_QUERY = "Hola, ¿cómo estás?"

def get_available_experts():
    """Obtiene la lista de expertos disponibles"""
    try:
        response = requests.get(f"{BASE_URL}/experts", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [expert["expert_id"] for expert in data.get("experts", [])]
        else:
            print(f"❌ Error al obtener expertos: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return []

def test_single_expert(expert_id: str):
    """Prueba un experto específico"""
    print(f"   Probando {expert_id}...")

    try:
        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/api/generate",  # Usar el endpoint correcto
            json={
                "model": expert_id,
                "prompt": TEST_QUERY,
                "temperature": 0.1,
                "max_tokens": 20
            },
            headers={"Content-Type": "application/json"},
            timeout=60  # Aumentar timeout para permitir carga de modelo si es necesario
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                content = result["response"]
                tokens = len(content.split())

                print(f"   ✅ {expert_id} - OK | Tiempo: {elapsed_time:.3f}s | Tokens: {tokens}")
                return True
            else:
                print(f"   ❌ {expert_id} - No response in result")
                print(f"      {result}")
                return False
        else:
            print(f"   ❌ {expert_id} - HTTP {response.status_code}: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"   ⏳ {expert_id} - Timeout (posiblemente cargando el modelo)")
        return False
    except Exception as e:
        print(f"   ❌ {expert_id} - Error: {e}")
        return False

def main():
    print("🔍 Conectándose al servidor multimodelo...")
    print(f"   URL: {BASE_URL}")
    print()

    # Obtener expertos disponibles
    experts = get_available_experts()

    if not experts:
        print("❌ No se pudieron obtener expertos. Asegúrate que el servidor esté corriendo.")
        return

    print(f"✅ {len(experts)} expertos encontrados:")
    for expert in experts:
        print(f"   - {expert}")
    print()

    # Probar cada experto
    print("🧪 Probando cada experto individualmente...")
    print()

    successful = 0
    for expert in experts:
        if test_single_expert(expert):
            successful += 1
        print()  # Línea en blanco entre expertos

    print("="*60)
    print(f"✅ {successful}/{len(experts)} expertos respondieron correctamente")

    if successful != len(experts):
        failed = len(experts) - successful
        print(f"❌ {failed} expertos fallaron")

    print("="*60)

if __name__ == "__main__":
    main()