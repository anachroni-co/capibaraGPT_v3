"""
tools validate_robotics_integration module.

# This module provides functionality for validate_robotics_integration.
"""

import os
import sys
from pathlib import Path

def validate_robotics_structure():
    """Validates robotics directory structure"""
    print("🏗️  Validando estructura directorios robótica...")
    
    # directory principal
    robotics_dir = Path("capibara/data/datasets/robotics")
    if not robotics_dir.exists():
        print("❌ ERROR: Directorio robotics/ no existe")
        return False
    
    # Archivos requeridos
    required_files = [
        "__init__.py",
        "robotics_premium_datasets.py"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = robotics_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"   ✅ {file} - Existe")
    
    if missing_files:
        print(f"❌ ERROR: Archivos faltantes: {missing_files}")
        return False
        
    print("✅ Estructura directorios robótica: VÁLIDA")
    return True

def validate_robotics_imports():
    """Validates robotics module imports"""
    print("\n📦 Validando imports robótica...")
    
    try:
        # add path if es necessary
        current_dir = Path.cwd()
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))
        
        # Test import principal
        from capibara.data.datasets.robotics import (
            RoboticsPremiumDatasetManager,
            RoboTurkConfig,
            CalvinConfig,
            OpenXEmbodimentConfig
        )
        print("   ✅ Import classs principales - OK")
        
        # Test factory functions
        from capibara.data.datasets.robotics import (
            create_robotics_datasets_manager,
            get_robotics_datasets_summary,
            get_recommended_robotics_datasets_by_task
        )
        print("   ✅ Import factory functions - OK")
        
    except ImportError as e:
        print(f"❌ ERROR Import: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR Inesperado: {e}")
        return False
        
    print("✅ Imports robótica: VÁLIDOS")
    return True

def validate_robotics_configs():
    """Validates robotics dataset configurations"""
    print("\n⚙️  Validando configuraciones datasets...")
    
    try:
        from capibara.data.datasets.robotics import (
            RoboTurkConfig, CalvinConfig, OpenXEmbodimentConfig
        )
        
        # Test RoboTurk Config
        roboturk = RoboTurkConfig()
        assert roboturk.quality_score == 9.8
        assert roboturk.total_demonstrations == 111000
        assert "imitation_learning" in roboturk.use_cases
        print("   ✅ RoboTurk Config - Válida")
        
        # Test CALVIN Config  
        calvin = CalvinConfig()
        assert calvin.quality_score == 9.6
        assert calvin.total_episodes == 25000
        assert "language_conditioned_robotics" in calvin.use_cases
        print("   ✅ CALVIN Config - Válida")
        
        # Test Open X-Embodiment Config
        open_x = OpenXEmbodimentConfig()
        assert open_x.quality_score == 9.9
        assert open_x.total_robot_types == 22
        assert "cross_embodiment_learning" in open_x.use_cases
        print("   ✅ Open X-Embodiment Config - Válida")
        
    except Exception as e:
        print(f"❌ ERROR Configs: {e}")
        return False
        
    print("✅ Configuraciones datasets: VÁLIDAS")
    return True

def validate_robotics_manager():
    """Validates dataset manager functionality"""
    print("\n🎯 Validando RoboticsPremiumDatasetManager...")
    
    try:
        from capibara.data.datasets.robotics import create_robotics_datasets_manager
        
        # create test manager
        manager = create_robotics_datasets_manager("test_robotics")
        
        # Test metadatos
        assert manager.metadata["total_datasets"] == 3
        assert manager.metadata["average_quality_score"] > 9.5
        print("   ✅ Manager metadata - Válidos")
        
        # Test information datasets
        roboturk_info = manager.get_roboturk_info()
        assert "manipulation_tasks" in roboturk_info["capabilities"]
        print("   ✅ RoboTurk info - Válida")
        
        calvin_info = manager.get_calvin_info()  
        assert "language_grounding" in calvin_info["capabilities"]
        print("   ✅ CALVIN info - Válida")
        
        open_x_info = manager.get_open_x_info()
        assert "cross_embodiment" in open_x_info["capabilities"]
        print("   ✅ Open X-Embodiment info - Válida")
        
        # Test resumen integration
        summary = manager.get_integration_summary()
        assert summary["integration_overview"]["total_datasets"] == 3
        assert "Google DeepMind Robotics" in summary["integration_overview"]["authoritative_sources"]
        print("   ✅ Integration summary - Válido")
        
    except Exception as e:
        print(f"❌ ERROR Manager: {e}")
        return False
        
    print("✅ RoboticsPremiumDatasetManager: FUNCIONAL")
    return True

def validate_robotics_functions():
    """Validates robotics utility functions"""
    print("\n🔧 Validando funciones utilitarias...")
    
    try:
        from capibara.data.datasets.robotics import (
            get_robotics_datasets_summary,
            get_recommended_robotics_datasets_by_task
        )
        
        # Test summary function
        summary = get_robotics_datasets_summary()
        assert summary["integration_status"] == "COMPLETED - 3/3 datasets premium"
        assert "1.1M+ episodes" in summary["total_coverage"]["demonstrations"]
        print("   ✅ get_robotics_datasets_summary - Funcional")
        
        # Test recommendations
        imitation_rec = get_recommended_robotics_datasets_by_task("imitation_learning")
        assert imitation_rec["recommendation"]["primary"] == "RoboTurk Dataset"
        print("   ✅ get_recommended_robotics_datasets_by_task - Funcional")
        
        language_rec = get_recommended_robotics_datasets_by_task("language_conditioned")
        assert language_rec["recommendation"]["primary"] == "CALVIN Dataset"
        print("   ✅ Recomendaciones por tarea - Funcionales")
        
    except Exception as e:
        print(f"❌ ERROR Funciones: {e}")
        return False
        
    print("✅ Funciones utilitarias: FUNCIONALES")
    return True

def validate_integration_in_main_datasets():
    """Validates integration in main datasets module"""
    print("\n🔗 Validando integración en datasets principal...")
    
    try:
        from capibara.data.datasets import get_available_categories, get_robotics_summary
        
        # Test categorías disponibles
        categories = get_available_categories()
        assert "robotics" in categories
        print("   ✅ Categoría 'robotics' incluida - OK")
        
        # Test resumen robótica
        robotics_summary = get_robotics_summary()
        assert robotics_summary["status"] == "NUEVA DIMENSIÓN INTEGRADA"
        assert "RoboTurk (Berkeley)" in robotics_summary["datasets"]
        print("   ✅ Resumen robótica disponible - OK")
        
    except Exception as e:
        print(f"❌ ERROR Integración principal: {e}")
        return False
        
    print("✅ Integración en datasets principal: COMPLETA")
    return True

def main():
    # Main function for this module.
    logger.info("Module validate_robotics_integration.py starting")
    return True

if __name__ == "__main__":
    main()
