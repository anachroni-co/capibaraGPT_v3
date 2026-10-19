#!/usr/bin/env python3
"""
️ CAPIBARA TEXT-TO-PRINT3D SERVICE
===================================

Main service for generating 3D models optimized for 3D printing
from natural language descriptions.

Supports:
- Technologies: FDM, SLA, SLS, Multi-Jet Fusion
- Materials: PLA, ABS, PETG, Resin, Nylon, TPU
- Formats: STL, OBJ, 3MF, AMF
- Automatic print optimization
- Intelligent support generation
- Time and cost calculation
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import math
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import numpy for STL generation
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("️ NumPy not available, using basic STL generation")

class PrintTechnology(Enum):
    """Supported 3D printing technologies"""
    FDM = "fdm"                    # Fused Deposition Modeling
    SLA = "sla"                    # Stereolithography
    SLS = "sls"                    # Selective Laser Sintering
    MULTI_JET = "multi_jet"        # Multi Jet Fusion
    POLYJET = "polyjet"            # PolyJet Technology
    BINDER_JETTING = "binder_jet"  # Binder Jetting

class MaterialType(Enum):
    """Material types for 3D printing"""
    PLA = "pla"                    # Polylactic Acid (FDM)
    ABS = "abs"                    # Acrylonitrile Butadiene Styrene (FDM)
    PETG = "petg"                  # Polyethylene Terephthalate Glycol (FDM)
    TPU = "tpu"                    # Thermoplastic Polyurethane (FDM flexible)
    NYLON = "nylon"                # Nylon (FDM/SLS)
    RESIN_STANDARD = "resin_std"   # Standard Resin (SLA)
    RESIN_TOUGH = "resin_tough"    # Tough Resin (SLA)
    RESIN_FLEXIBLE = "resin_flex"  # Flexible Resin (SLA)
    METAL_STEEL = "metal_steel"    # Metal Steel (SLS/Binder Jet)
    CERAMIC = "ceramic"            # Ceramic (Binder Jetting)

class PrintQuality(Enum):
    """Print qualities"""
    DRAFT = "draft"                # 0.3mm+ layer height, fast
    STANDARD = "standard"          # 0.2mm layer height, balanced
    HIGH = "high"                  # 0.1mm layer height, detailed
    ULTRA = "ultra"                # 0.05mm layer height, maximum detail

@dataclass
class Print3DGenerationConfig:
    """Configuration for Print3D model generation"""
    # General settings
    output_directory: str = "./generated_print3d"
    default_technology: PrintTechnology = PrintTechnology.FDM
    default_material: MaterialType = MaterialType.PLA
    default_quality: PrintQuality = PrintQuality.STANDARD
    
    # Print bed settings
    print_bed_size_mm: Tuple[float, float, float] = (220, 220, 250)  # Prusa i3 MK3
    print_bed_center: bool = True
    
    # Layer settings
    layer_heights = {
        PrintQuality.DRAFT: 0.3,
        PrintQuality.STANDARD: 0.2,
        PrintQuality.HIGH: 0.1,
        PrintQuality.ULTRA: 0.05
    }
    
    # Material properties
    material_properties = {
        MaterialType.PLA: {"temp": 210, "bed_temp": 60, "density": 1.24},
        MaterialType.ABS: {"temp": 250, "bed_temp": 100, "density": 1.04},
        MaterialType.PETG: {"temp": 240, "bed_temp": 80, "density": 1.27},
        MaterialType.TPU: {"temp": 220, "bed_temp": 50, "density": 1.20},
        MaterialType.NYLON: {"temp": 270, "bed_temp": 110, "density": 1.15},
    }
    
    # Optimization settings
    auto_support_generation: bool = True
    support_overhang_angle: float = 45.0  # degrees
    auto_infill_optimization: bool = True
    default_infill_percentage: float = 20.0
    
    # Wall settings
    wall_thickness_mm: float = 1.2
    top_bottom_layers: int = 3
    
    # STL generation
    stl_precision: float = 0.01  # mm precision
    mesh_resolution: int = 32    # vertices per circle
    
    # E2B Integration
    use_e2b_openscad: bool = True
    use_e2b_blender: bool = True
    use_e2b_freecad: bool = True
    use_optimizer: bool = True
    preferred_tool: Optional[str] = "auto"  # "openscad", "blender", "freecad", "auto"
    
    # Export formats
    export_formats: List[str] = field(default_factory=lambda: ["stl", "obj"])
    generate_gcode: bool = False  # Requires slicer integration
    
    # Cost calculation
    material_cost_per_kg: Dict[str, float] = field(default_factory=lambda: {
        "pla": 25.0, "abs": 30.0, "petg": 35.0, "tpu": 50.0, "nylon": 60.0
    })
    print_time_cost_per_hour: float = 2.0  # USD per hour

@dataclass
class Print3DRequest:
    """Request for Print3D model generation"""
    description: str
    technology: PrintTechnology = PrintTechnology.FDM
    material: MaterialType = MaterialType.PLA
    quality: PrintQuality = PrintQuality.STANDARD
    dimensions: Optional[Dict[str, float]] = None
    scale_factor: float = 1.0
    infill_percentage: Optional[float] = None
    supports_required: Optional[bool] = None
    custom_parameters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    project_name: Optional[str] = None

@dataclass
class Print3DResult:
    """Print3D model generation result"""
    success: bool
    stl_file_path: Optional[str] = None
    obj_file_path: Optional[str] = None
    file_paths: List[str] = field(default_factory=list)
    
    # Print analysis
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    print_time_hours: float = 0.0
    material_weight_g: float = 0.0
    material_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    
    # Technical info
    layer_count: int = 0
    support_volume_mm3: float = 0.0
    infill_volume_mm3: float = 0.0
    
    # Quality metrics
    printability_score: float = 0.0  # 0-1, higher is better
    overhangs_detected: bool = False
    supports_generated: bool = False
    
    # Slicer settings
    recommended_layer_height: float = 0.2
    recommended_print_speed: float = 50.0  # mm/s
    recommended_temperature: float = 210.0
    
    # Generation info
    generation_time_seconds: float = 0.0
    tool_used: Optional[str] = None
    error: Optional[str] = None

# Try to import E2B integrations
try:
    from .e2b_openscad_print3d import E2BOpenSCADPrint3D, OpenSCADPrint3DConfig
    OPENSCAD_E2B_AVAILABLE = True
except ImportError:
    OPENSCAD_E2B_AVAILABLE = False

try:
    from .e2b_blender_print3d import E2BBlenderPrint3D, BlenderPrint3DConfig
    BLENDER_E2B_AVAILABLE = True
except ImportError:
    BLENDER_E2B_AVAILABLE = False

try:
    from .e2b_freecad_print3d import E2BFreeCADPrint3D, FreeCADPrint3DConfig
    FREECAD_E2B_AVAILABLE = True
except ImportError:
    FREECAD_E2B_AVAILABLE = False

try:
    from .print3d_optimizer import Print3DOptimizer, OptimizationConfig
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

class Print3DDescriptionParser:
    """Parser to extract 3D printing specifications from natural language descriptions"""
    
    def __init__(self):
        self.object_keywords = {
            "toy": ["toy", "juguete", "figura", "miniatura"],
            "tool": ["tool", "herramienta", "llave", "destornillador"],
            "container": ["container", "caja", "recipiente", "vaso", "bowl"],
            "decorative": ["decorative", "ornament", "decorativo", "adorno"],
            "functional": ["functional", "funcional", "útil", "práctica"],
            "miniature": ["miniature", "miniatura", "pequeño", "tiny"],
            "prototype": ["prototype", "prototipo", "test", "prueba"],
            "replacement": ["replacement", "repuesto", "spare", "reemplazo"]
        }
        
        self.complexity_keywords = {
            "simple": ["simple", "básico", "easy", "fácil"],
            "medium": ["medium", "normal", "standard", "estándar"],
            "complex": ["complex", "complejo", "detailed", "detallado", "intricate"]
        }
        
        self.technology_keywords = {
            PrintTechnology.FDM: ["fdm", "filament", "filamento", "extruded"],
            PrintTechnology.SLA: ["sla", "resin", "resina", "liquid"],
            PrintTechnology.SLS: ["sls", "powder", "polvo", "sintered"],
        }
        
        self.material_keywords = {
            MaterialType.PLA: ["pla", "biodegradable", "eco"],
            MaterialType.ABS: ["abs", "strong", "fuerte", "resistant"],
            MaterialType.PETG: ["petg", "clear", "transparent", "chemical"],
            MaterialType.TPU: ["tpu", "flexible", "rubber", "elastic"],
            MaterialType.NYLON: ["nylon", "tough", "durable", "engineering"]
        }
    
    def parse_description(self, description: str) -> Dict[str, Any]:
        """Extract 3D printing specifications from the description"""
        description_lower = description.lower()
        
        # Detect object type
        object_type = "general"
        for obj_type, keywords in self.object_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                object_type = obj_type
                break
        
        # Detect complexity
        complexity = "medium"
        for comp_level, keywords in self.complexity_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                complexity = comp_level
                break
        
        # Detect preferred technology
        detected_tech = PrintTechnology.FDM  # Default
        for tech, keywords in self.technology_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_tech = tech
                break
        
        # Detect preferred material
        detected_material = MaterialType.PLA  # Default
        for material, keywords in self.material_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_material = material
                break
        
        # Extract dimensions
        dimensions = self._extract_dimensions(description)
        
        # Determine automatic configurations
        auto_config = self._determine_auto_config(object_type, complexity)
        
        return {
            "object_type": object_type,
            "complexity": complexity,
            "technology": detected_tech,
            "material": detected_material,
            "dimensions": dimensions,
            "auto_config": auto_config
        }
    
    def _extract_dimensions(self, description: str) -> Dict[str, float]:
        """Extract dimensions from the description."""
        import re
        
        dimensions = {}
        
        # Dimension patterns
        size_patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:mm|millimeter|centimeter|cm)",
            r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:mm|cm)",
            r"diameter\s*(\d+(?:\.\d+)?)\s*(?:mm|cm)",
            r"height\s*(\d+(?:\.\d+)?)\s*(?:mm|cm)",
            r"width\s*(\d+(?:\.\d+)?)\s*(?:mm|cm)",
            r"length\s*(\d+(?:\.\d+)?)\s*(?:mm|cm)"
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, description.lower())
            if matches:
                if "diameter" in pattern:
                    dimensions["diameter"] = float(matches[0])
                elif "height" in pattern:
                    dimensions["height"] = float(matches[0])
                elif "width" in pattern:
                    dimensions["width"] = float(matches[0])
                elif "length" in pattern:
                    dimensions["length"] = float(matches[0])
                elif "x" in pattern:
                    dimensions["width"] = float(matches[0][0])
                    dimensions["length"] = float(matches[0][1])
                else:
                    dimensions["size"] = float(matches[0])
                break
        
        return dimensions
    
    def _determine_auto_config(self, object_type: str, complexity: str) -> Dict[str, Any]:
        """Determine automatic configuration based on type and complexity."""
        config = {}
        
        # Configuration by object type
        if object_type == "miniature":
            config.update({
                "quality": PrintQuality.HIGH,
                "infill": 15.0,
                "supports": True
            })
        elif object_type == "tool":
            config.update({
                "quality": PrintQuality.STANDARD,
                "infill": 50.0,
                "material": MaterialType.ABS
            })
        elif object_type == "decorative":
            config.update({
                "quality": PrintQuality.HIGH,
                "infill": 10.0,
                "supports": True
            })
        elif object_type == "container":
            config.update({
                "quality": PrintQuality.STANDARD,
                "infill": 25.0,
                "material": MaterialType.PETG
            })
        
        # Adjust by complexity
        if complexity == "simple":
            config["quality"] = PrintQuality.DRAFT
            config["supports"] = False
        elif complexity == "complex":
            config["quality"] = PrintQuality.HIGH
            config["supports"] = True
        
        return config

class MockPrint3DGenerator:
    """Mock generator for Print3D models when E2B is not available."""
    
    def __init__(self, config: Print3DGenerationConfig):
        self.config = config
        
    async def generate_print3d_mock(self, request: Print3DRequest) -> Print3DResult:
        """Generate a mock 3D model optimized for printing."""
        logger.info(f"️ Generating mock Print3D model: {request.description[:100]}...")
        
        # Simulate generation time
        await asyncio.sleep(2.0)
        
        # Generate file names
        project_name = request.project_name or "print3d_model"
        stl_file = f"{project_name}.stl"
        obj_file = f"{project_name}.obj"
        
        # Simulate print analysis
        volume = self._calculate_mock_volume(request)
        surface_area = volume ** (2/3) * 6  # Approximation
        
        # Material properties
        material_props = self.config.material_properties.get(
            request.material, 
            self.config.material_properties[MaterialType.PLA]
        )
        
        material_weight = volume * material_props["density"] / 1000  # grams
        material_cost = material_weight / 1000 * self.config.material_cost_per_kg.get(
            request.material.value, 25.0
        )
        
        # Estimated print time
        layer_height = self.config.layer_heights[request.quality]
        estimated_height = request.dimensions.get("height", 50) if request.dimensions else 50
        layer_count = int(estimated_height / layer_height)
        print_time = layer_count * 2.5 / 60  # estimated hours
        
        total_cost = material_cost + (print_time * self.config.print_time_cost_per_hour)
        
        # Printability analysis
        printability_score = self._calculate_printability_score(request)
        overhangs_detected = "overhang" in request.description.lower() or printability_score < 0.7
        supports_needed = overhangs_detected or request.supports_required
        
        return Print3DResult(
            success=True,
            stl_file_path=stl_file,
            obj_file_path=obj_file,
            file_paths=[stl_file, obj_file],
            volume_mm3=volume,
            surface_area_mm2=surface_area,
            print_time_hours=print_time,
            material_weight_g=material_weight,
            material_cost_usd=material_cost,
            total_cost_usd=total_cost,
            layer_count=layer_count,
            support_volume_mm3=volume * 0.05 if supports_needed else 0.0,
            infill_volume_mm3=volume * (request.infill_percentage or 20.0) / 100,
            printability_score=printability_score,
            overhangs_detected=overhangs_detected,
            supports_generated=supports_needed,
            recommended_layer_height=layer_height,
            recommended_print_speed=50.0,
            recommended_temperature=material_props["temp"],
            generation_time_seconds=2.0,
            tool_used="mock_generator"
        )
    
    def _calculate_mock_volume(self, request: Print3DRequest) -> float:
        """Calculate estimated volume based on description and dimensions"""
        if request.dimensions:
            if "diameter" in request.dimensions and "height" in request.dimensions:
                # Cylinder
                r = request.dimensions["diameter"] / 2
                h = request.dimensions["height"]
                return math.pi * r * r * h
            elif all(k in request.dimensions for k in ["width", "length", "height"]):
                # Box
                return (request.dimensions["width"] * 
                       request.dimensions["length"] * 
                       request.dimensions["height"])
            elif "size" in request.dimensions:
                # Cube
                size = request.dimensions["size"]
                return size ** 3
        
        # Default volume based on description
        description_lower = request.description.lower()
        if any(word in description_lower for word in ["large", "big", "grande"]):
            return 50000.0  # 50cm³
        elif any(word in description_lower for word in ["small", "tiny", "pequeño"]):
            return 5000.0   # 5cm³
        else:
            return 20000.0  # 20cm³
    
    def _calculate_printability_score(self, request: Print3DRequest) -> float:
        """Calculate printability score (0-1)"""
        score = 0.8  # Base score
        
        description_lower = request.description.lower()
        
        # Penalize problematic features
        if any(word in description_lower for word in ["overhang", "bridge", "cantilever"]):
            score -= 0.2
        if any(word in description_lower for word in ["thin", "delicate", "fine"]):
            score -= 0.1
        if any(word in description_lower for word in ["complex", "intricate", "detailed"]):
            score -= 0.1
        
        # Bonus for favorable features
        if any(word in description_lower for word in ["solid", "thick", "robust"]):
            score += 0.1
        if request.material in [MaterialType.PLA, MaterialType.PETG]:
            score += 0.05
        
        return max(0.0, min(1.0, score))

class CapibaraTextToPrint3D:
    """Main Text-to-Print3D generation service."""
    
    def __init__(self, config: Optional[Print3DGenerationConfig] = None):
        self.config = config or Print3DGenerationConfig()
        
        # Initialize parser and mock generator
        self.description_parser = Print3DDescriptionParser()
        self.mock_generator = MockPrint3DGenerator(self.config)
        
        # Initialize E2B generators
        self.openscad_generator = None
        self.blender_generator = None
        self.freecad_generator = None
        self.optimizer = None
        
        if self.config.use_e2b_openscad and OPENSCAD_E2B_AVAILABLE:
            self._init_openscad_generator()
            
        if self.config.use_e2b_blender and BLENDER_E2B_AVAILABLE:
            self._init_blender_generator()
            
        if self.config.use_e2b_freecad and FREECAD_E2B_AVAILABLE:
            self._init_freecad_generator()
            
        if self.config.use_optimizer and OPTIMIZER_AVAILABLE:
            self._init_optimizer()
        
        # Configure directories
        self._setup_directories()
        
        logger.info(" CapibaraTextToPrint3D initialized")
    
    def _init_openscad_generator(self):
        """Initialize OpenSCAD generator for Print3D"""
        try:
            openscad_config = OpenSCADPrint3DConfig()
            self.openscad_generator = E2BOpenSCADPrint3D(openscad_config)
            logger.info(" OpenSCAD Print3D generator initialized")
        except Exception as e:
            logger.error(f" Error initializing OpenSCAD Print3D: {e}")
            self.openscad_generator = None
    
    def _init_blender_generator(self):
        """Initialize Blender generator for Print3D"""
        try:
            blender_config = BlenderPrint3DConfig()
            self.blender_generator = E2BBlenderPrint3D(blender_config)
            logger.info(" Blender Print3D generator initialized")
        except Exception as e:
            logger.error(f" Error initializing Blender Print3D: {e}")
            self.blender_generator = None
    
    def _init_freecad_generator(self):
        """Initialize FreeCAD generator for Print3D"""
        try:
            freecad_config = FreeCADPrint3DConfig()
            self.freecad_generator = E2BFreeCADPrint3D(freecad_config)
            logger.info(" FreeCAD Print3D generator initialized")
        except Exception as e:
            logger.error(f" Error initializing FreeCAD Print3D: {e}")
            self.freecad_generator = None
    
    def _init_optimizer(self):
        """Initialize Print3D optimizer"""
        try:
            optimizer_config = OptimizationConfig()
            self.optimizer = Print3DOptimizer(optimizer_config)
            logger.info(" Print3D optimizer initialized")
        except Exception as e:
            logger.error(f" Error initializing Print3D optimizer: {e}")
            self.optimizer = None
    
    def _setup_directories(self):
        """Configure output directories"""
        try:
            os.makedirs(self.config.output_directory, exist_ok=True)
            logger.info(f" Print3D output directory: {self.config.output_directory}")
        except Exception as e:
            logger.error(f" Error creating directories: {e}")
    
    async def generate_print3d(self, request: Print3DRequest) -> Print3DResult:
        """Generate 3D model optimized for printing from natural language description"""
        start_time = datetime.now()
        
        try:
            logger.info(f"️ Generating Print3D model: {request.description[:100]}...")
            
            # Parse description for auto-configuration
            parsed_specs = self.description_parser.parse_description(request.description)
            
            # Apply auto-configuration if not specified
            if not request.dimensions:
                request.dimensions = parsed_specs.get("dimensions", {})
            
            auto_config = parsed_specs.get("auto_config", {})
            if "quality" in auto_config and request.quality == PrintQuality.STANDARD:
                request.quality = auto_config["quality"]
            if "material" in auto_config and request.material == MaterialType.PLA:
                request.material = auto_config["material"]
            if request.infill_percentage is None and "infill" in auto_config:
                request.infill_percentage = auto_config["infill"]
            if request.supports_required is None and "supports" in auto_config:
                request.supports_required = auto_config["supports"]
            
            # Priority 1: Tool selection based on object type and complexity
            complexity = parsed_specs.get("complexity", "medium")
            object_type = parsed_specs.get("object_type", "general")
            
            selected_tool = self._select_optimal_tool(object_type, complexity, request)
            
            if selected_tool == "openscad" and self.openscad_generator:
                logger.info(" Using OpenSCAD for procedural Print3D generation...")
                result = await self.openscad_generator.generate_print3d_with_openscad(
                    description=request.description,
                    technology=request.technology,
                    material=request.material,
                    quality=request.quality,
                    dimensions=request.dimensions or {},
                    user_id=request.user_id
                )
                
                if result.get("success"):
                    return self._convert_e2b_result_to_print3d_result(result, request, start_time, "openscad_e2b")
            
            elif selected_tool == "blender" and self.blender_generator:
                logger.info(" Using Blender for artistic Print3D generation...")
                result = await self.blender_generator.generate_print3d_with_blender(
                    description=request.description,
                    technology=request.technology,
                    material=request.material,
                    quality=request.quality,
                    dimensions=request.dimensions or {},
                    user_id=request.user_id
                )
                
                if result.get("success"):
                    return self._convert_e2b_result_to_print3d_result(result, request, start_time, "blender_e2b")
            
            elif selected_tool == "freecad" and self.freecad_generator:
                logger.info("️ Using FreeCAD for parametric Print3D generation...")
                result = await self.freecad_generator.generate_print3d_with_freecad(
                    description=request.description,
                    technology=request.technology,
                    material=request.material,
                    quality=request.quality,
                    dimensions=request.dimensions or {},
                    user_id=request.user_id
                )
                
                if result.get("success"):
                    return self._convert_e2b_result_to_print3d_result(result, request, start_time, "freecad_e2b")
            
            # Fallback to mock generation
            logger.info(" Using mock Print3D generation (E2B tools not available)")
            result = await self.mock_generator.generate_print3d_mock(request)
            result.generation_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Apply optimization if available
            if self.optimizer and result.success:
                logger.info(" Applying Print3D optimizations...")
                optimization_result = await self.optimizer.optimize_for_printing(
                    stl_file=result.stl_file_path,
                    technology=request.technology,
                    material=request.material,
                    quality=request.quality
                )
                
                if optimization_result.get("success"):
                    result.printability_score = optimization_result.get("printability_score", result.printability_score)
                    result.supports_generated = optimization_result.get("supports_added", result.supports_generated)
                    result.support_volume_mm3 = optimization_result.get("support_volume", result.support_volume_mm3)
            
            return result
            
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f" Print3D generation error: {e}")
            
            return Print3DResult(
                success=False,
                generation_time_seconds=generation_time,
                error=str(e)
            )
    
    def _select_optimal_tool(self, object_type: str, complexity: str, request: Print3DRequest) -> str:
        """Select the optimal tool for the object type."""
        if self.config.preferred_tool and self.config.preferred_tool != "auto":
            return self.config.preferred_tool
        
        # Smart selection based on object type
        if object_type in ["tool", "container", "functional"]:
            return "freecad"  # Best for functional and parametric objects
        elif object_type in ["decorative", "toy", "miniature"]:
            return "blender"  # Best for artistic and detailed objects
        elif complexity == "simple" or object_type == "prototype":
            return "openscad"  # Best for simple and procedural objects
        
        # Default: FreeCAD for general cases
        return "freecad"
    
    def _convert_e2b_result_to_print3d_result(self, e2b_result: Dict[str, Any], 
                                            request: Print3DRequest, 
                                            start_time: datetime, 
                                            tool_used: str) -> Print3DResult:
        """Convert E2B result to Print3DResult"""
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return Print3DResult(
            success=True,
            stl_file_path=e2b_result.get("stl_file"),
            obj_file_path=e2b_result.get("obj_file"),
            file_paths=e2b_result.get("file_paths", []),
            volume_mm3=e2b_result.get("volume_mm3", 0.0),
            surface_area_mm2=e2b_result.get("surface_area_mm2", 0.0),
            print_time_hours=e2b_result.get("print_time_hours", 0.0),
            material_weight_g=e2b_result.get("material_weight_g", 0.0),
            material_cost_usd=e2b_result.get("material_cost_usd", 0.0),
            total_cost_usd=e2b_result.get("total_cost_usd", 0.0),
            layer_count=e2b_result.get("layer_count", 0),
            support_volume_mm3=e2b_result.get("support_volume_mm3", 0.0),
            infill_volume_mm3=e2b_result.get("infill_volume_mm3", 0.0),
            printability_score=e2b_result.get("printability_score", 0.8),
            overhangs_detected=e2b_result.get("overhangs_detected", False),
            supports_generated=e2b_result.get("supports_generated", False),
            recommended_layer_height=e2b_result.get("layer_height", 0.2),
            recommended_print_speed=e2b_result.get("print_speed", 50.0),
            recommended_temperature=e2b_result.get("temperature", 210.0),
            generation_time_seconds=generation_time,
            tool_used=tool_used
        )
    
    # Utility methods
    def is_openscad_available(self) -> bool:
        """Check if OpenSCAD is available"""
        return (self.config.use_e2b_openscad and 
                self.openscad_generator is not None and 
                self.openscad_generator.is_available())
    
    def is_blender_available(self) -> bool:
        """Check if Blender is available"""
        return (self.config.use_e2b_blender and 
                self.blender_generator is not None and 
                self.blender_generator.is_available())
    
    def is_freecad_available(self) -> bool:
        """Check if FreeCAD is available"""
        return (self.config.use_e2b_freecad and 
                self.freecad_generator is not None and 
                self.freecad_generator.is_available())
    
    def is_optimizer_available(self) -> bool:
        """Check if Optimizer is available"""
        return (self.config.use_optimizer and 
                self.optimizer is not None and 
                self.optimizer.is_available())
    
    async def test_all_tools(self) -> Dict[str, Any]:
        """Test availability of all Print3D tools"""
        results = {}
        
        if self.openscad_generator:
            results["openscad"] = await self.openscad_generator.test_openscad_availability()
        else:
            results["openscad"] = {"available": False, "error": "Not initialized"}
        
        if self.blender_generator:
            results["blender"] = await self.blender_generator.test_blender_availability()
        else:
            results["blender"] = {"available": False, "error": "Not initialized"}
        
        if self.freecad_generator:
            results["freecad"] = await self.freecad_generator.test_freecad_availability()
        else:
            results["freecad"] = {"available": False, "error": "Not initialized"}
        
        if self.optimizer:
            results["optimizer"] = await self.optimizer.test_optimizer_availability()
        else:
            results["optimizer"] = {"available": False, "error": "Not initialized"}
        
        results["mock_generator"] = {"available": True, "status": "Always available"}
        
        return results
    
    def get_supported_materials(self, technology: PrintTechnology) -> List[MaterialType]:
        """Get supported materials for a specific technology"""
        material_compatibility = {
            PrintTechnology.FDM: [MaterialType.PLA, MaterialType.ABS, MaterialType.PETG, 
                                MaterialType.TPU, MaterialType.NYLON],
            PrintTechnology.SLA: [MaterialType.RESIN_STANDARD, MaterialType.RESIN_TOUGH, 
                                MaterialType.RESIN_FLEXIBLE],
            PrintTechnology.SLS: [MaterialType.NYLON, MaterialType.METAL_STEEL],
            PrintTechnology.BINDER_JETTING: [MaterialType.METAL_STEEL, MaterialType.CERAMIC]
        }
        
        return material_compatibility.get(technology, [MaterialType.PLA])
    
    def estimate_print_cost(self, volume_mm3: float, material: MaterialType, 
                          print_time_hours: float) -> Dict[str, float]:
        """Estimate printing costs"""
        # Material density (g/cm3)
        material_density = self.config.material_properties.get(
            material, {"density": 1.24}
        )["density"]
        
        # Material weight in grams
        material_weight_g = volume_mm3 * material_density / 1000
        
        # Material cost
        material_cost_per_kg = self.config.material_cost_per_kg.get(material.value, 25.0)
        material_cost = (material_weight_g / 1000) * material_cost_per_kg
        
        # Print time cost
        time_cost = print_time_hours * self.config.print_time_cost_per_hour
        
        # Total cost
        total_cost = material_cost + time_cost
        
        return {
            "material_weight_g": material_weight_g,
            "material_cost_usd": material_cost,
            "time_cost_usd": time_cost,
            "total_cost_usd": total_cost
        }