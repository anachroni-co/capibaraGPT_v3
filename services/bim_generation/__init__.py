"""
️ CAPIBARA TEXT-TO-BIM SERVICE
===============================

Servicio revolucionario de modelado BIM arquitectónico con IA.
Genera modelos BIM completos desde descripciones en lenguaje natural.

Capacidades:
- Modelado arquitectónico 3D automático
- Integración con Revit, AutoCAD, SketchUp, FreeCAD
- Generación de planos técnicos
- Cálculo de materiales y presupuestos
- Análisis estructural básico
- Exportación a múltiples formatos (IFC, DWG, SKP, etc.)
- Cumplimiento de códigos de construcción
- Renderizado fotorrealista

Herramientas integradas:
- Autodesk Revit (via API)
- AutoCAD (DXF/DWG generation)
- SketchUp (via Ruby API)
- FreeCAD (Python scripting)
- Blender (architectural visualization)
- IFC.js (web visualization)

Estándares soportados:
- IFC (Industry Foundation Classes)
- COBie (Construction Operations Building Information Exchange)
- BuildingSMART standards
- Local building codes (US, EU, etc.)
"""

from .text_to_bim_service import (
    CapibaraTextToBIM,
    BIMGenerationConfig,
    BIMRequest,
    BIMResult,
    BuildingType,
    BIMFormat,
    StructuralSystem,
    ArchitecturalStyle,
    ConstructionPhase
)

from .bim_manager import (
    BIMManager,
    BIMToolSelector,
    BIMValidator,
    BIMOptimizer,
    StructuralAnalyzer
)

from .architectural_parser import (
    ArchitecturalDescriptionParser,
    SpaceAnalyzer,
    MaterialSpecifier,
    CodeComplianceChecker
)

from .bim_generators import (
    RevitBIMGenerator,
    AutoCADBIMGenerator,
    SketchUpBIMGenerator,
    FreeCADBIMGenerator,
    BlenderBIMGenerator,
    MockBIMGenerator
)

from .bim_exporters import (
    IFCExporter,
    DWGExporter,
    SKPExporter,
    PDFExporter,
    RenderExporter
)

# Tool availability flags
try:
    import revitpythonshell
    REVIT_AVAILABLE = True
except ImportError:
    REVIT_AVAILABLE = False

try:
    import ezdxf
    AUTOCAD_DXF_AVAILABLE = True
except ImportError:
    AUTOCAD_DXF_AVAILABLE = False

try:
    import sketchup
    SKETCHUP_AVAILABLE = True
except ImportError:
    SKETCHUP_AVAILABLE = False

try:
    import FreeCAD
    FREECAD_AVAILABLE = True
except ImportError:
    FREECAD_AVAILABLE = False

try:
    import bpy  # Blender Python API
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

# E2B Integration for secure BIM generation
try:
    from e2b import Sandbox
    from .e2b_bim_integration import E2BBIMExecutor
    E2B_BIM_AVAILABLE = True
except ImportError:
    E2B_BIM_AVAILABLE = False

# Service factories
def create_text_to_bim_service(config=None):
    """Crea instancia principal del servicio Text-to-BIM"""
    return CapibaraTextToBIM(config)

def create_bim_manager():
    """Crea gestor inteligente de herramientas BIM"""
    return BIMManager()

def create_architectural_parser():
    """Crea parser de descripciones arquitectónicas"""
    return ArchitecturalDescriptionParser()

def create_bim_tool_selector():
    """Crea selector automático de herramientas BIM"""
    return BIMToolSelector()

def get_available_bim_tools():
    """Retorna herramientas BIM disponibles en el sistema"""
    tools = {
        "mock_bim": True,  # Siempre disponible para testing
        "freecad": FREECAD_AVAILABLE,
        "blender": BLENDER_AVAILABLE,
        "autocad_dxf": AUTOCAD_DXF_AVAILABLE,
        "revit": REVIT_AVAILABLE,
        "sketchup": SKETCHUP_AVAILABLE,
        "e2b_sandbox": E2B_BIM_AVAILABLE
    }
    return tools

__all__ = [
    'CapibaraTextToBIM',
    'BIMGenerationConfig',
    'BIMRequest', 
    'BIMResult',
    'BuildingType',
    'BIMFormat',
    'StructuralSystem',
    'ArchitecturalStyle',
    'ConstructionPhase',
    'BIMManager',
    'ArchitecturalDescriptionParser',
    'create_text_to_bim_service',
    'create_bim_manager',
    'create_architectural_parser',
    'get_available_bim_tools',
    'REVIT_AVAILABLE',
    'AUTOCAD_DXF_AVAILABLE',
    'FREECAD_AVAILABLE',
    'BLENDER_AVAILABLE',
    'E2B_BIM_AVAILABLE'
]