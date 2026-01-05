#!/usr/bin/env python3
"""
Script para verificar la conectividad entre VMs y probar el sistema de Programming-Only RAG
"""

import socket
import requests
import time
from programming_rag_detector import is_programming_query


def test_connectivity_to_internal_ip():
    """Test if we can connect to the model server using internal IP"""
    internal_ip = "10.204.0.9"  # This VM's internal IP
    port = 8082
    
    print(f"🔍 Testing connectivity to internal IP: {internal_ip}:{port}")
    
    # Test if port is open
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((internal_ip, port))
    sock.close()
    
    if result == 0:
        print(f"✅ Port {port} is OPEN on {internal_ip}")
    else:
        print(f"❌ Port {port} is CLOSED on {internal_ip}")
    
    # Test health endpoint
    try:
        response = requests.get(f"http://{internal_ip}:{port}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    return result == 0


def test_connectivity_to_external_ip():
    """Test if we can connect to the model server using external IP"""
    external_ip = "34.175.104.75"  # This VM's external IP
    port = 8082
    
    print(f"🔍 Testing connectivity to external IP: {external_ip}:{port}")
    
    # Test if port is open
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((external_ip, port))
    sock.close()
    
    if result == 0:
        print(f"✅ Port {port} is OPEN on {external_ip}")
    else:
        print(f"❌ Port {port} is CLOSED on {external_ip}")
    
    # Test health endpoint
    try:
        response = requests.get(f"http://{external_ip}:{port}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    return result == 0


def test_connectivity_to_localhost():
    """Test if we can connect to the model server using localhost"""
    localhost = "localhost"
    port = 8082
    
    print(f"🔍 Testing connectivity to localhost: {localhost}:{port}")
    
    try:
        response = requests.get(f"http://{localhost}:{port}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_model_functionality():
    """Test the actual model functionality"""
    print(f"\n🔬 Testing model functionality...")
    
    url = "http://localhost:8082/v1/chat/completions"
    
    payload = {
        "model": "aya_expanse_multilingual",
        "messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model test successful!")
            print(f"   Response: {result['choices'][0]['message']['content'][:100]}...")
            return True
        else:
            print(f"❌ Model test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False


def test_programming_detection():
    """Test the programming-only RAG detection system"""
    print(f"\n🎯 Testing Programming-Only RAG Detection...")
    
    test_queries = [
        ("How do I sort an array in Python?", True, "programming"),
        ("What is the weather like today?", False, "general"),
        ("Debug this JavaScript code for me", True, "programming"),
        ("Tell me about history", False, "general"),
        ("Show me a Python example for database connection", True, "programming"),
        ("How to cook pasta?", False, "general")
    ]
    
    correctly_classified = 0
    for query, expected, category in test_queries:
        is_programming = is_programming_query(query)
        status = "✅" if is_programming == expected else "❌"
        print(f"   {status} {category.upper()}: '{query[:35]}...' → {is_programming} (expected: {expected})")
        if is_programming == expected:
            correctly_classified += 1
    
    print(f"\n✅ Programming detection: {correctly_classified}/{len(test_queries)} correct")
    return correctly_classified == len(test_queries)


def main():
    print("🦫 CAPIBARA6 - VM Connectivity & Programming RAG Test")
    print("=" * 80)
    print("Testing connectivity between VMs and Programming-Only RAG system")
    print("This will help identify why VM services can't connect to this VM models-europe")
    print("=" * 80)
    
    print(f"\n📡 VM models-europe Info:")
    print(f"   Internal IP: 10.204.0.9")
    print(f"   External IP: 34.175.104.75")
    print(f"   Model server port: 8082")
    print(f"   Programming-Only RAG: ACTIVE")
    
    # Test connectivity
    print(f"\n🌐 Connectivity Tests:")
    localhost_ok = test_connectivity_to_localhost()
    internal_ok = test_connectivity_to_internal_ip()
    external_ok = test_connectivity_to_external_ip()
    
    # Test functionality
    print(f"\n⚙️  Functionality Tests:")
    model_ok = test_model_functionality()
    
    # Test programming detection
    print(f"\n🎯 RAG Detection Test:")
    detection_ok = test_programming_detection()
    
    print(f"\n📋 RESULTS SUMMARY:")
    print(f"   - Localhost connectivity: {'✅ OK' if localhost_ok else '❌ FAIL'}")
    print(f"   - Internal IP connectivity: {'✅ OK' if internal_ok else '❌ FAIL'}")
    print(f"   - External IP connectivity: {'✅ OK' if external_ok else '❌ FAIL'}")
    print(f"   - Model functionality: {'✅ OK' if model_ok else '❌ FAIL'}")
    print(f"   - Programming detection: {'✅ OK' if detection_ok else '❌ FAIL'}")
    
    if model_ok and detection_ok and localhost_ok:
        print(f"\n🎯 STATUS: ✅ VM models-europe is ready and functional")
        print(f"   - The model server is running and responding correctly")
        print(f"   - The Programming-Only RAG system is working")
        print(f"   - Issue is likely with VM services connecting using wrong IP")
    else:
        print(f"\n⚠️  STATUS: Issues detected - VM may need configuration fix")
    
    print(f"\n💡 SOLUTION FOR VM SERVICES CONNECTION:")
    print(f"   VM services should connect to one of these (in order of preference):")
    print(f"   1. Internal IP (if in same subnet): http://10.204.0.9:8082/v1/chat/completions")
    print(f"   2. External IP: http://34.175.104.75:8082/v1/chat/completions")
    print(f"   3. The IP being used now (incorrect): http://34.175.48.2:8082/v1/chat/completions")


if __name__ == "__main__":
    main()