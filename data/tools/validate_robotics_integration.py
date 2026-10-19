"""
tools validate_robotics_integration module.

# This module provides functionality for validate_robotics_integration.
"""

import os
import sys
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def validate_robotics_structure():
    """Validates robotics directory structure"""
    logger.info("️  Validating robotics directory structure...")

    # main directory
    robotics_dir = Path("capibara/data/datasets/robotics")
    if not robotics_dir.exists():
        logger.error(" ERROR: Directory robotics/ does not exist")
        return False

    # Required files
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
            logger.info(f"    {file} - Existe")
    
    if missing_files:
        logger.error(f" ERROR: Missing files: {missing_files}")
        return False

    logger.info(" Robotics directory structure: VALID")
    return True

def validate_robotics_imports():
    """Validates robotics module imports"""
    logger.info("\n Validating robotics imports...")

    try:
        # add path if necessary
        current_dir = Path.cwd()
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))
        
        # Test main import
        from capibara.data.datasets.robotics import (
            RoboticsPremiumDatasetManager,
            RoboTurkConfig,
            CalvinConfig,
            OpenXEmbodimentConfig
        )
        logger.info("    Main classes import - OK")
        
        # Test factory functions
        from capibara.data.datasets.robotics import (
            create_robotics_datasets_manager,
            get_robotics_datasets_summary,
            get_recommended_robotics_datasets_by_task
        )
        logger.info("    Import factory functions - OK")
        
    except ImportError as e:
        logger.error(f" ERROR Import: {e}")
        return False
    except Exception as e:
        logger.error(f" Unexpected ERROR: {e}")
        return False

    logger.info(" Robotics imports: VALID")
    return True

def validate_robotics_configs():
    """Validates robotics dataset configurations"""
    logger.info("\n️  Validating dataset configurations...")
    
    try:
        from capibara.data.datasets.robotics import (
            RoboTurkConfig, CalvinConfig, OpenXEmbodimentConfig
        )
        
        # Test RoboTurk Config
        roboturk = RoboTurkConfig()
        assert roboturk.quality_score == 9.8
        assert roboturk.total_demonstrations == 111000
        assert "imitation_learning" in roboturk.use_cases
        logger.info("    RoboTurk Config - Valid")
        
        # Test CALVIN Config  
        calvin = CalvinConfig()
        assert calvin.quality_score == 9.6
        assert calvin.total_episodes == 25000
        assert "language_conditioned_robotics" in calvin.use_cases
        logger.info("    CALVIN Config - Valid")
        
        # Test Open X-Embodiment Config
        open_x = OpenXEmbodimentConfig()
        assert open_x.quality_score == 9.9
        assert open_x.total_robot_types == 22
        assert "cross_embodiment_learning" in open_x.use_cases
        logger.info("    Open X-Embodiment Config - Valid")
        
    except Exception as e:
        logger.error(f" ERROR Configs: {e}")
        return False

    logger.info(" Dataset configurations: VALID")
    return True

def validate_robotics_manager():
    """Validates dataset manager functionality"""
    logger.info("\n Validando RoboticsPremiumDatasetManager...")
    
    try:
        from capibara.data.datasets.robotics import create_robotics_datasets_manager
        
        # create test manager
        manager = create_robotics_datasets_manager("test_robotics")
        
        # Test metadatos
        assert manager.metadata["total_datasets"] == 3
        assert manager.metadata["average_quality_score"] > 9.5
        logger.info("    Manager metadata - Válidos")
        
        # Test information datasets
        roboturk_info = manager.get_roboturk_info()
        assert "manipulation_tasks" in roboturk_info["capabilities"]
        logger.info("    RoboTurk info - Válida")
        
        calvin_info = manager.get_calvin_info()  
        assert "language_grounding" in calvin_info["capabilities"]
        logger.info("    CALVIN info - Válida")
        
        open_x_info = manager.get_open_x_info()
        assert "cross_embodiment" in open_x_info["capabilities"]
        logger.info("    Open X-Embodiment info - Válida")
        
        # Test resumen integration
        summary = manager.get_integration_summary()
        assert summary["integration_overview"]["total_datasets"] == 3
        assert "Google DeepMind Robotics" in summary["integration_overview"]["authoritative_sources"]
        logger.info("    Integration summary - Válido")
        
    except Exception as e:
        logger.error(f" ERROR Manager: {e}")
        return False
        
    logger.info(" RoboticsPremiumDatasetManager: FUNCIONAL")
    return True

def validate_robotics_functions():
    """Validates robotics utility functions"""
    logger.info("\n Validando funciones utilitarias...")
    
    try:
        from capibara.data.datasets.robotics import (
            get_robotics_datasets_summary,
            get_recommended_robotics_datasets_by_task
        )
        
        # Test summary function
        summary = get_robotics_datasets_summary()
        assert summary["integration_status"] == "COMPLETED - 3/3 datasets premium"
        assert "1.1M+ episodes" in summary["total_coverage"]["demonstrations"]
        logger.info("    get_robotics_datasets_summary - Funcional")
        
        # Test recommendations
        imitation_rec = get_recommended_robotics_datasets_by_task("imitation_learning")
        assert imitation_rec["recommendation"]["primary"] == "RoboTurk Dataset"
        logger.info("    get_recommended_robotics_datasets_by_task - Funcional")
        
        language_rec = get_recommended_robotics_datasets_by_task("language_conditioned")
        assert language_rec["recommendation"]["primary"] == "CALVIN Dataset"
        logger.info("    Recomendaciones por tarea - Funcionales")
        
    except Exception as e:
        logger.error(f" ERROR Funciones: {e}")
        return False
        
    logger.info(" Funciones utilitarias: FUNCIONALES")
    return True

def validate_integration_in_main_datasets():
    """Validates integration in main datasets module"""
    logger.info("\n Validando integración en datasets principal...")
    
    try:
        from capibara.data.datasets import get_available_categories, get_robotics_summary
        
        # Test categorías disponibles
        categories = get_available_categories()
        assert "robotics" in categories
        logger.info("    Categoría 'robotics' incluida - OK")
        
        # Test resumen robótica
        robotics_summary = get_robotics_summary()
        assert robotics_summary["status"] == "NUEVA DIMENSIÓN INTEGRADA"
        assert "RoboTurk (Berkeley)" in robotics_summary["datasets"]
        logger.info("    Resumen robótica disponible - OK")
        
    except Exception as e:
        logger.error(f" ERROR Integración principal: {e}")
        return False
        
    logger.info(" Integración en datasets principal: COMPLETA")
    return True

def main():
    # Main function for this module.
    logger.info("Module validate_robotics_integration.py starting")
    return True

if __name__ == "__main__":
    main()
