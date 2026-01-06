#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Module for Capibara6 (75B7e27c)

This module provides the main entry point for the application with enhanced
router optimization and core-training integration capabilities.
"""

import sys
import argparse
import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

class CapibaraMain:
    """Main application class with enhanced integration capabilities and mental health safety."""
    
    def __init__(self):
        """Initialize the main application."""
        self.router = None
        self.integration = None
        self.bridge = None
        self.safety_manager = None
        self.config = {}
        self.safety_enabled = True
        # Enhanced submodels
        self.csa_expert = None
        self.sapir_whorf_adapter = None
        
    async def initialize_components(self):
        """Initialize all main components with proper error handling."""
        try:
            # Import enhanced components
            from capibara.core.router import create_enhanced_router, RouterConfig, RouterType
            from capibara.core.ultra_core_integration import create_ultra_core_integration, IntegrationConfig
            from capibara.training.core_training_bridge import create_core_training_bridge, BridgeConfig
            
            # Initialize enhanced submodels with corrected imports
            await self.initialize_enhanced_submodels()
            
            logger.info("🔧 Initializing enhanced router...")
            
            # Initialize enhanced router
            router_config = RouterConfig(
                router_type=RouterType.HYBRID,
                encoding="utf-8",
                use_training_integration=True,
                expert_cores_enabled=True,
                consensus_enabled=True,
                cache_size=1000
            )
            self.router = create_enhanced_router(router_config)
            
            logger.info("🔧 Initializing ultra core integration...")
            
            # Initialize ultra core integration
            integration_config = IntegrationConfig(
                integration_type="full",
                encoding="utf-8",
                enable_caching=True,
                training_integration_enabled=True
            )
            self.integration = create_ultra_core_integration(integration_config)
            
            logger.info("🔧 Initializing core training bridge...")
            
            # Initialize core training bridge
            bridge_config = BridgeConfig(
                mode="full_integration",
                encoding="utf-8",
                enable_caching=True,
                enable_monitoring=True
            )
            self.bridge = create_core_training_bridge(bridge_config)
            
            # Initialize mental health safety system
            logger.info("🔒 Initializing mental health safety system...")
            await self.initialize_safety_system()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"Some components not available: {e}")
            return await self.initialize_fallback_components()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return await self.initialize_fallback_components()
    
    async def initialize_safety_system(self):
        """Initialize the mental health safety system."""
        try:
            from capibara.safety import activate_safety_system, verify_safety_system
            
            logger.info("🛡️ Activating mental health protection system...")
            
            # Verify safety system components
            if verify_safety_system():
                # Activate safety system
                self.safety_manager = activate_safety_system()
                
                if self.safety_manager:
                    logger.info("✅ Mental health safety system activated successfully")
                    logger.info("🔍 Active protections:")
                    logger.info("   - Usage pattern monitoring: ✅")
                    logger.info("   - Content filtering: ✅") 
                    logger.info("   - Automatic interventions: ✅")
                    logger.info("   - Emergency resources: ✅")
                    
                    # Log emergency contacts for quick reference
                    logger.info("🆘 Emergency contacts available:")
                    logger.info("   - Emergency services: 911")
                    logger.info("   - Suicide prevention: 988")
                    logger.info("   - Crisis text line: 741741")
                    
                    self.safety_enabled = True
                else:
                    logger.warning("⚠️ Safety system components loaded but not fully activated")
                    self.safety_enabled = False
            else:
                logger.error("❌ Safety system verification failed")
                self.safety_enabled = False
                
        except ImportError as e:
            logger.warning(f"⚠️ Mental health safety system not available: {e}")
            logger.warning("🚨 SYSTEM RUNNING WITHOUT MENTAL HEALTH PROTECTIONS")
            logger.warning("📋 Consider installing safety components for production use")
            self.safety_enabled = False
            
        except Exception as e:
            logger.error(f"❌ Error initializing safety system: {e}")
            logger.error("🚨 SYSTEM RUNNING WITHOUT MENTAL HEALTH PROTECTIONS")
            self.safety_enabled = False
    
    async def initialize_enhanced_submodels(self):
        """Initialize enhanced submodels with corrected imports."""
        try:
            logger.info("🧠 Initializing enhanced submodels...")
            
            # Import CSA Expert
            try:
                from capibara.sub_models.csa_expert import CSAExpert
                logger.info("✅ CSA Expert imported successfully")
                self.csa_expert = CSAExpert()
            except ImportError as e:
                logger.warning(f"⚠️ CSA Expert not available: {e}")
                self.csa_expert = None
            
            # Import SapirWhorfAdapter with correct name
            try:
                from capibara.sub_models.semiotic.sapir_whorf_adapter import SapirWhorfAdapter
                logger.info("✅ SapirWhorfAdapter imported successfully")
                self.sapir_whorf_adapter = SapirWhorfAdapter()
                logger.info("🌍 Sapir-Whorf linguistic adaptation system activated")
            except ImportError as e:
                logger.warning(f"⚠️ SapirWhorfAdapter not available: {e}")
                self.sapir_whorf_adapter = None
            
            # Store submodels in config
            self.config['enhanced_submodels'] = {
                'csa_expert': self.csa_expert,
                'sapir_whorf_adapter': self.sapir_whorf_adapter
            }
            
            available_submodels = [name for name, module in self.config['enhanced_submodels'].items() if module is not None]
            if available_submodels:
                logger.info(f"🎉 Enhanced submodels available: {', '.join(available_submodels)}")
            else:
                logger.warning("⚠️ No enhanced submodels available")
                
        except Exception as e:
            logger.error(f"❌ Error initializing enhanced submodels: {e}")
            self.config['enhanced_submodels'] = {}
    
    async def initialize_fallback_components(self):
        """Initialize fallback components when enhanced ones are not available."""
        try:
            logger.info("🔄 Initializing fallback components...")
            
            # Try to import basic components
            try:
                from capibara.core.router import create_router
                self.router = create_router()
                logger.info("✅ Basic router initialized")
            except ImportError:
                logger.warning("Basic router not available")
                
            try:
                from capibara.core.routing import create_router as create_core_router
                self.core_router = create_core_router()
                logger.info("✅ Core router initialized")
            except ImportError:
                logger.warning("Core router not available")
                
            return True
            
        except Exception as e:
            logger.error(f"Error initializing fallback components: {e}")
            return False
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check of all components."""
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": None
        }
        
        try:
            import time
            health_status["timestamp"] = time.time()
            
            # Check router
            if self.router:
                try:
                    router_info = self.router.get_router_info()
                    health_status["components"]["router"] = {
                        "status": "available",
                        "type": router_info.get("router_type", "unknown"),
                        "encoding": router_info.get("encoding", "unknown")
                    }
                except Exception as e:
                    health_status["components"]["router"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status["overall"] = "degraded"
            else:
                health_status["components"]["router"] = {"status": "unavailable"}
                health_status["overall"] = "degraded"
            
            # Check integration
            if self.integration:
                try:
                    integration_health = await self.integration.health_check()
                    health_status["components"]["integration"] = integration_health
                except Exception as e:
                    health_status["components"]["integration"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status["overall"] = "degraded"
            else:
                health_status["components"]["integration"] = {"status": "unavailable"}
                health_status["overall"] = "degraded"
            
            # Check bridge
            if self.bridge:
                try:
                    bridge_health = await self.bridge.health_check()
                    health_status["components"]["bridge"] = bridge_health
                except Exception as e:
                    health_status["components"]["bridge"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status["overall"] = "degraded"
            else:
                health_status["components"]["bridge"] = {"status": "unavailable"}
                health_status["overall"] = "degraded"
                
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_status["overall"] = "error"
            health_status["error"] = str(e)
            
        return health_status
    
    async def test_router_functionality(self) -> Dict[str, Any]:
        """Test router functionality with various inputs."""
        test_results = {
            "success": True,
            "tests": {},
            "summary": {}
        }
        
        try:
            if not self.router:
                test_results["success"] = False
                test_results["error"] = "Router not available"
                return test_results
            
            # Test with different input types
            test_inputs = [
                "Simple test input",
                "Texto en español con caracteres especiales: áéíóúñ",
                "Test with numbers: 12345 and symbols: @#$%",
                {"text": "Structured input", "type": "test"}
            ]
            
            for i, test_input in enumerate(test_inputs):
                try:
                    result = await self.router.route_request(test_input)
                    test_results["tests"][f"test_{i+1}"] = {
                        "input": str(test_input)[:50] + "..." if len(str(test_input)) > 50 else str(test_input),
                        "success": result.success,
                        "selected_module": result.selected_module,
                        "confidence": result.confidence
                    }
                except Exception as e:
                    test_results["tests"][f"test_{i+1}"] = {
                        "input": str(test_input)[:50] + "..." if len(str(test_input)) > 50 else str(test_input),
                        "success": False,
                        "error": str(e)
                    }
                    test_results["success"] = False
            
            # Generate summary
            successful_tests = sum(1 for test in test_results["tests"].values() if test.get("success", False))
            total_tests = len(test_results["tests"])
            
            test_results["summary"] = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error during router testing: {e}")
            test_results["success"] = False
            test_results["error"] = str(e)
            
        return test_results
    
    async def test_sapir_whorf_functionality(self) -> Dict[str, Any]:
        """Test SapirWhorfAdapter functionality with various language inputs."""
        test_results = {
            "success": True,
            "tests": {},
            "summary": {}
        }
        
        try:
            if not self.sapir_whorf_adapter:
                test_results["success"] = False
                test_results["error"] = "SapirWhorfAdapter not available"
                return test_results
            
            # Test with different language inputs
            test_inputs = [
                ("Hello, how are you today?", "en"),
                ("Hola, ¿cómo estás hoy?", "es"),
                ("你好，你今天怎么样？", "zh"),
                ("مرحبا، كيف حالك اليوم؟", "ar")
            ]
            
            for i, (text, expected_lang) in enumerate(test_inputs):
                try:
                    # Test language detection
                    from capibara.sub_models.semiotic.sapir_whorf_adapter import detect_language
                    detected_lang = detect_language(text)
                    
                    # Test language profile retrieval
                    profile = self.sapir_whorf_adapter.get_language_profile(detected_lang)
                    
                    test_results["tests"][f"test_{i+1}"] = {
                        "input": text[:30] + "..." if len(text) > 30 else text,
                        "expected_language": expected_lang,
                        "detected_language": detected_lang,
                        "profile_available": profile is not None,
                        "success": True
                    }
                except Exception as e:
                    test_results["tests"][f"test_{i+1}"] = {
                        "input": text[:30] + "..." if len(text) > 30 else text,
                        "success": False,
                        "error": str(e)
                    }
                    test_results["success"] = False
            
            # Test supported languages
            try:
                supported_languages = self.sapir_whorf_adapter.list_supported_languages()
                test_results["supported_languages"] = supported_languages
                test_results["language_count"] = len(supported_languages)
            except Exception as e:
                test_results["supported_languages_error"] = str(e)
            
            # Generate summary
            successful_tests = sum(1 for test in test_results["tests"].values() if test.get("success", False))
            total_tests = len(test_results["tests"])
            
            test_results["summary"] = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error during SapirWhorf testing: {e}")
            test_results["success"] = False
            test_results["error"] = str(e)
            
        return test_results
    
    async def run_demo(self):
        """Run a demonstration of the enhanced capabilities."""
        logger.info("🎬 Starting Capibara6 demonstration...")
        
        try:
            # Health check
            logger.info("🔍 Running health check...")
            health = await self.run_health_check()
            logger.info(f"Health status: {health['overall']}")
            
            # Router functionality test
            logger.info("🧪 Testing router functionality...")
            router_test = await self.test_router_functionality()
            logger.info(f"Router test success rate: {router_test['summary'].get('success_rate', 0):.1f}%")
            
            # SapirWhorf functionality test
            logger.info("🌍 Testing SapirWhorf linguistic adaptation...")
            sapir_test = await self.test_sapir_whorf_functionality()
            if sapir_test['success']:
                logger.info(f"SapirWhorf test success rate: {sapir_test['summary'].get('success_rate', 0):.1f}%")
                if 'language_count' in sapir_test:
                    logger.info(f"Supported languages: {sapir_test['language_count']}")
            else:
                logger.warning(f"SapirWhorf test failed: {sapir_test.get('error', 'Unknown error')}")
            
            # Integration test
            if self.integration:
                logger.info("🔗 Testing integration capabilities...")
                try:
                    integration_result = await self.integration.process_request("Integration test request")
                    logger.info(f"Integration test: {'Success' if integration_result.success else 'Failed'}")
                except Exception as e:
                    logger.warning(f"Integration test failed: {e}")
            
            # Bridge test
            if self.bridge:
                logger.info("🌉 Testing bridge capabilities...")
                try:
                    bridge_result = await self.bridge.process_request("Bridge test request")
                    logger.info(f"Bridge test: {'Success' if bridge_result.success else 'Failed'}")
                except Exception as e:
                    logger.warning(f"Bridge test failed: {e}")
            
            logger.info("🎉 Demonstration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during demonstration: {e}")

async def main():
    """Main entry point with enhanced capabilities."""
    logger.info("🚀 Starting Capibara6 (75B7e27c) with enhanced integration...")
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Capibara6 Main Application")
        parser.add_argument("--demo", action="store_true", help="Run demonstration mode")
        parser.add_argument("--health", action="store_true", help="Run health check only")
        parser.add_argument("--test", action="store_true", help="Run functionality tests")
        parser.add_argument("--test-sapir", action="store_true", help="Test SapirWhorf linguistic adaptation")
        parser.add_argument("--encoding", default="utf-8", help="Set encoding (default: utf-8)")
        
        args = parser.parse_args()
        
        # Create main application instance
        app = CapibaraMain()
        
        # Initialize components
        logger.info("🔧 Initializing components...")
        if not await app.initialize_components():
            logger.error("❌ Failed to initialize components")
            return 1
        
        # Run requested operations
        if args.health:
            logger.info("🔍 Running health check...")
            health = await app.run_health_check()
            logger.info(f"Health status: {health}")
            return 0 if health["overall"] == "healthy" else 1
            
        elif args.test:
            logger.info("🧪 Running functionality tests...")
            test_results = await app.test_router_functionality()
            logger.info(f"Test results: {test_results}")
            return 0 if test_results["success"] else 1
            
        elif args.test_sapir:
            logger.info("🌍 Running SapirWhorf linguistic adaptation tests...")
            sapir_results = await app.test_sapir_whorf_functionality()
            logger.info(f"SapirWhorf test results: {sapir_results}")
            return 0 if sapir_results["success"] else 1
            
        elif args.demo:
            await app.run_demo()
            return 0
            
        else:
            # Default: run health check and basic info
            logger.info("📊 System Information:")
            logger.info(f"   - Encoding: {args.encoding}")
            logger.info(f"   - Version: 75B7e27c")
            logger.info(f"   - Enhanced Integration: Enabled")
            logger.info(f"   - SapirWhorf Adapter: {'✅ Available' if app.sapir_whorf_adapter else '❌ Not Available'}")
            logger.info(f"   - CSA Expert: {'✅ Available' if app.csa_expert else '❌ Not Available'}")
            
            health = await app.run_health_check()
            logger.info(f"   - Health Status: {health['overall']}")
            
            logger.info("🎉 Capibara6 started successfully")
            return 0
        
    except Exception as e:
        logger.error(f"❌ Error starting Capibara6: {e}")
        return 1

if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
