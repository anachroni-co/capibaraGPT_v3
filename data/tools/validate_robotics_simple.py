"""
tools validate_robotics_simple module.

# This module provides functionality for validate_robotics_simple.
"""

import os
import sys
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def validate_robotics_files():
    """Validates robotics files"""
    logger.info(" Validando archivos robótica...")
    
    robotics_dir = Path("capibara/data/datasets/robotics")
    required_files = [
        "__init__.py",
        "robotics_premium_datasets.py"
    ]
    
    success = True
    for file in required_files:
        file_path = robotics_dir / file
        if file_path.exists():
            logger.info(f"    {file} - OK")
        else:
            logger.info(f"    {file} - FALTANTE")
            success = False
    
    return success

def validate_robotics_syntax():
    """Validates basic Python file syntax"""
    logger.info("\n Validando sintaxis archivos...")
    
    robotics_files = [
        "capibara/data/datasets/robotics/__init__.py",
        "capibara/data/datasets/robotics/robotics_premium_datasets.py"
    ]
    
    success = True
    for file_path in robotics_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Compilar for verify sintaxis
            compile(code, file_path, 'exec')
            logger.info(f"    {Path(file_path).name} - Sintaxis válida")
            
        except SyntaxError as e:
            logger.error(f"    {Path(file_path).name} - Error sintaxis: {e}")
            success = False
        except Exception as e:
            logger.error(f"   ️  {Path(file_path).name} - Error: {e}")
    
    return success

def validate_imports_standalone():
    """Validates imports in isolation"""
    logger.info("\n Validando imports standalone...")
    
    # add path
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    try:
        # Import directo del file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "robotics_premium_datasets", 
            "capibara/data/datasets/robotics/robotics_premium_datasets.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        logger.info("    robotics_premium_datasets.py - Import OK")
        
        # verify classes principales
        required_classes = [
            'RoboTurkConfig',
            'CalvinConfig', 
            'OpenXEmbodimentConfig',
            'RoboticsPremiumDatasetManager'
        ]
        
        for cls_name in required_classes:
            if hasattr(module, cls_name):
                logger.info(f"    {cls_name} - Disponible")
            else:
                logger.info(f"    {cls_name} - Faltante")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"    Error import: {e}")
        return False

def validate_content_quality():
    """Validates content quality"""
    logger.info("\n Validando calidad contenido...")
    
    try:
        with open("capibara/data/datasets/robotics/robotics_premium_datasets.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # verify palabras key importantes
        keywords = [
            "RoboTurk", "CALVIN", "Open X-Embodiment",
            "Berkeley", "TU Berlin", "Google DeepMind",
            "quality_score", "total_demonstrations", "total_episodes"
        ]
        
        missing = []
        for keyword in keywords:
            if keyword not in content:
                missing.append(keyword)
        
        if missing:
            logger.warning(f"   ️  Palabras clave faltantes: {missing}")
        else:
            logger.info("    Contenido completo - Todas las palabras clave presentes")
        
        # verify métricas
        if "9.8" in content and "9.6" in content and "9.9" in content:
            logger.info("    Quality scores - Presentes")
        else:
            logger.info("   ️  Quality scores - Incompletos")
            
        return len(missing) == 0
        
    except Exception as e:
        logger.error(f"    Error validación contenido: {e}")
        return False

def validate_structure_integration():
    """Validates integration en structure principal"""
    logger.info("\n Validando integración estructura...")
    
    try:
        # verify que robotics está en datasets __init__.py
        with open("capibara/data/datasets/__init__.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "robotics" in content and "" in content:
            logger.info("    Integración en datasets/__init__.py - OK")
        else:
            logger.info("    Integración en datasets/__init__.py - Faltante")
            return False
            
        # verify directory structure
        robotics_dir = Path("capibara/data/datasets/robotics")
        if robotics_dir.exists() and robotics_dir.is_dir():
            logger.info("    Directorio robotics/ - Existe")
        else:
            logger.info("    Directorio robotics/ - No existe")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"    Error validación estructura: {e}")
        return False

def main():
    # Main function for this module.
    logger.info("Module validate_robotics_simple.py starting")
    return True

if __name__ == "__main__":
    main()
