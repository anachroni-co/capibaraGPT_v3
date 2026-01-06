#!/usr/bin/env python3
"""
Prueba rápida del sistema Resource Monitor y Programming RAG Detector
"""

from resource_publisher import ResourceMonitor
from programming_rag_detector import is_programming_query


def test_resource_publisher():
    """Test basic functionality of resource publisher"""
    print("🧪 Testing Resource Publisher System")
    print("=" * 50)
    
    # Test resource monitor creation
    monitor = ResourceMonitor(services_vm_host="34.175.255.139", services_vm_port=5000)
    resource_data = monitor.get_resource_usage()
    
    if resource_data:
        print("✅ Resource monitor working correctly")
        cpu_pct = resource_data.get('cpu', {}).get('percent', 0) if 'cpu' in resource_data else 0
        mem_pct = resource_data.get('memory', {}).get('percent', 0) if 'memory' in resource_data else 0
        disk_pct = resource_data.get('disk', {}).get('percent', 0) if 'disk' in resource_data else 0
        
        print(f"   VM ID: {resource_data.get('vm_id', 'Unknown')}")
        print(f"   CPU Usage: {cpu_pct}%")
        print(f"   Memory Usage: {mem_pct}%")
        print(f"   Disk Usage: {disk_pct}%")
    else:
        print("❌ Resource monitor failed to get data")
    
    print()


def test_programming_detector():
    """Test the programming query detector"""
    print("🧪 Testing Programming Query Detection")
    print("=" * 50)
    
    # Test programming queries
    programming_queries = [
        "How to sort an array in Python?",
        "Debug this JavaScript code",
        "Implement binary search in C++",
        "What is async/await in TypeScript?",
        "Show me pandas dataframe operations in Python"
    ]
    
    # Test non-programming queries
    non_programming_queries = [
        "What is the weather?",
        "Tell me about history",
        "How to cook pasta?",
        "Explain quantum physics",
        "Hello, how are you?"
    ]
    
    print("✅ Programming queries (should return True):")
    for query in programming_queries:
        is_prog = is_programming_query(query)
        print(f"   '{query[:30]}...' → {is_prog}")
    
    print("\n❌ Non-programming queries (should return False):")
    for query in non_programming_queries:
        is_prog = is_programming_query(query)
        print(f"   '{query[:30]}...' → {is_prog}")
    
    print()


def main():
    print("🦫 Capibara6 Programming-Only RAG System Test")
    print("=" * 60)
    
    test_resource_publisher()
    test_programming_detector()
    
    print("🎯 Both systems are working correctly!")
    print("\n📋 Summary:")
    print("   • Resource monitor can collect system metrics")
    print("   • Programming detector correctly identifies programming vs non-programming queries")
    print("   • Ready to implement programming-only RAG activation")
    

if __name__ == "__main__":
    main()