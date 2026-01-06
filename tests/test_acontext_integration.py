#!/usr/bin/env python3
"""
Test script to verify Acontext integration functionality
"""
import asyncio
import os
import sys
from datetime import datetime

def test_acontext_integration():
    """Test the Acontext integration without external dependencies"""
    print("🧪 Testing Acontext integration...")

    # Add backend to path to import the module
    backend_path = os.path.join(os.getcwd(), 'backend')
    sys.path.insert(0, backend_path)

    try:
        from acontext_integration import AcontextIntegration, AcontextSession

        # Test initialization
        print("1. Testing AcontextIntegration initialization...")
        os.environ["ACONTEXT_BASE_URL"] = "http://test:8029/api/v1"
        os.environ["ACONTEXT_API_KEY"] = "sk-test-key"

        integration = AcontextIntegration()

        if integration.enabled:
            print("✅ Acontext integration initialized and enabled")
        else:
            print("❌ Acontext integration is disabled")
            return False

        print("✅ Acontext integration test completed")
        return True
    except ImportError as e:
        print(f"❌ Error importing AcontextIntegration: {e}")
        return False

async def test_gateway_server_import():
    """Test that the gateway server imports our Acontext integration without errors"""
    print("\n2. Testing gateway server import with Acontext integration...")
    try:
        # Try to import the gateway server
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

        # Import the gateway server to ensure it can import acontext_integration
        import backend.gateway_server

        print("✅ Gateway server imports Acontext integration successfully")
        return True
    except ImportError as e:
        print(f"❌ Error importing gateway server: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_acontext_session_creation():
    """Test Acontext session creation"""
    print("\n3. Testing Acontext session creation...")
    try:
        backend_path = os.path.join(os.getcwd(), 'backend')
        sys.path.insert(0, backend_path)

        from acontext_integration import AcontextSession

        session = AcontextSession(
            id="test-session-id",
            project_id="test-project-id",
            space_id="test-space-id",
            created_at=datetime.now()
        )

        print(f"✅ Session created: {session.id}")
        print(f"✅ Project ID: {session.project_id}")
        print(f"✅ Space ID: {session.space_id}")
        return True
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Acontext integration tests...\n")

    # Test the integration
    test1 = test_acontext_integration()

    # Test imports
    test2 = await test_gateway_server_import()

    # Test session creation
    test3 = await test_acontext_session_creation()

    print(f"\n✅ All Acontext integration tests completed")
    print(f"   - Integration initialization: {'PASSED' if test1 else 'FAILED'}")
    print(f"   - Gateway server import: {'PASSED' if test2 else 'FAILED'}")
    print(f"   - Session creation: {'PASSED' if test3 else 'FAILED'}")

    all_passed = test1 and test2 and test3
    print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    return all_passed

if __name__ == "__main__":
    asyncio.run(main())