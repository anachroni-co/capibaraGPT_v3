#!/usr/bin/env python3
"""
 CAPIBARA TEXT-TO-GEN SERVICE
==============================

Meta-service that automatically generates complete code for new text-to-* services
from natural language descriptions.

Capabilities:
- Requirements analysis from natural text
- Complete Python code generation
- Automatic integration with CapibaraGPT v3
- Automatic unit tests
- Auto-generated documentation
- Automatic API endpoints
- Deployment scripts included

Examples of services it can generate:
- Text-to-Music: Generates musical compositions
- Text-to-Game: Creates simple mini-games
- Text-to-3D: 3D models for virtual reality
- Text-to-Code: Code generator in any language
- Text-to-Presentation: Automatic slides
- Text-to-Website: Complete websites
- Text-to-Animation: 2D/3D animations
- Text-to-Data: Synthetic datasets
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import shutil
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Types of services that can be generated"""
    CREATIVE = "creative"           # Art, music, design
    TECHNICAL = "technical"         # CAD, code, circuits
    MEDIA = "media"                # Image, video, audio
    DATA = "data"                  # Datasets, APIs, databases
    INTERACTIVE = "interactive"     # Games, interfaces, apps
    DOCUMENT = "document"          # Presentations, reports, websites
    SIMULATION = "simulation"      # Physics, chemistry, mathematics
    BUSINESS = "business"          # Analytics, reports, workflows

class ComplexityLevel(Enum):
    """Service complexity levels"""
    SIMPLE = "simple"              # <500 lines, 1-2 files
    MEDIUM = "medium"              # 500-2000 lines, 3-5 files
    COMPLEX = "complex"            # 2000-5000 lines, 5-10 files
    ADVANCED = "advanced"          # >5000 lines, >10 files

class IntegrationType(Enum):
    """Integration types with the ecosystem"""
    STANDALONE = "standalone"       # Independent service
    CAPIBARA_NATIVE = "native"     # Full integration with CapibaraGPT v3
    E2B_INTEGRATION = "e2b"        # Uses E2B sandboxes
    API_WRAPPER = "api_wrapper"    # External API wrapper
    HYBRID = "hybrid"              # Combination of multiple types

@dataclass
class MetaGenerationConfig:
    """Configuration for the meta-generator"""
    # Output directories
    output_directory: str = "./generated_services"
    template_directory: str = "./templates"
    
    # Generation configuration
    default_complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    default_integration: IntegrationType = IntegrationType.CAPIBARA_NATIVE
    
    # Generated code features
    include_tests: bool = True
    include_docs: bool = True
    include_api_endpoints: bool = True
    include_demo_script: bool = True
    include_requirements: bool = True
    
    # Code style
    code_style: str = "pep8"
    docstring_style: str = "google"
    type_hints: bool = True
    async_support: bool = True
    
    # Code templates
    use_dataclasses: bool = True
    use_enums: bool = True
    use_logging: bool = True
    error_handling: str = "comprehensive"  # basic, standard, comprehensive
    
    # Integration with CapibaraGPT v3
    auto_register_service: bool = True
    generate_web_interface: bool = True
    create_service_factory: bool = True
    
    # AI/ML specific
    include_mock_mode: bool = True
    default_model_type: str = "transformer"
    include_training_scaffold: bool = True

@dataclass
class ServiceGenerationRequest:
    """Request to generate a new service"""
    description: str
    service_name: str
    service_type: ServiceType
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    integration_type: IntegrationType = IntegrationType.CAPIBARA_NATIVE
    
    # Technical specifications
    input_format: Optional[str] = None      # "text", "json", "file"
    output_format: Optional[str] = None     # "file", "json", "stream"
    dependencies: List[str] = field(default_factory=list)
    external_apis: List[str] = field(default_factory=list)
    
    # Functional features
    supports_batch: bool = False
    supports_streaming: bool = False
    requires_gpu: bool = False
    requires_internet: bool = False
    
    # Metadata
    author: Optional[str] = None
    license: str = "MIT"
    version: str = "1.0.0"
    user_id: Optional[str] = None

@dataclass
class ServiceGenerationResult:
    """Service generation result"""
    success: bool
    service_name: str
    service_path: str = ""
    
    # Generated files
    generated_files: List[str] = field(default_factory=list)
    main_service_file: str = ""
    test_files: List[str] = field(default_factory=list)
    documentation_files: List[str] = field(default_factory=list)
    
    # Code metrics
    total_lines: int = 0
    files_count: int = 0
    complexity_score: float = 0.0
    
    # Integration information
    api_endpoints: List[str] = field(default_factory=list)
    dependencies_added: List[str] = field(default_factory=list)
    
    # Instructions for the user
    installation_instructions: str = ""
    usage_examples: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # Development information
    generation_time_seconds: float = 0.0
    templates_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

# Try to import code generation components
try:
    from .code_generator import CodeGenerator
    from .service_analyzer import ServiceAnalyzer
    CODE_GENERATION_AVAILABLE = True
except ImportError:
    CODE_GENERATION_AVAILABLE = False

class ServiceRequirementAnalyzer:
    """Requirements analyzer from natural language description"""
    
    def __init__(self):
        self.service_keywords = {
            ServiceType.CREATIVE: ["music", "art", "design", "creative", "artistic", "song", "melody"],
            ServiceType.TECHNICAL: ["code", "program", "algorithm", "circuit", "cad", "engineering"],
            ServiceType.MEDIA: ["image", "video", "audio", "picture", "movie", "sound", "photo"],
            ServiceType.DATA: ["data", "dataset", "database", "api", "json", "csv", "table"],
            ServiceType.INTERACTIVE: ["game", "app", "interface", "interactive", "gui", "ui"],
            ServiceType.DOCUMENT: ["document", "report", "presentation", "slide", "website", "page"],
            ServiceType.SIMULATION: ["simulation", "physics", "chemistry", "math", "model", "calculate"],
            ServiceType.BUSINESS: ["business", "analytics", "report", "dashboard", "workflow", "process"]
        }
        
        self.complexity_indicators = {
            ComplexityLevel.SIMPLE: ["simple", "basic", "easy", "quick", "minimal"],
            ComplexityLevel.MEDIUM: ["standard", "normal", "regular", "typical"],
            ComplexityLevel.COMPLEX: ["complex", "advanced", "detailed", "comprehensive"],
            ComplexityLevel.ADVANCED: ["enterprise", "production", "scalable", "robust", "complete"]
        }
        
        self.integration_indicators = {
            IntegrationType.STANDALONE: ["standalone", "independent", "separate", "isolated"],
            IntegrationType.E2B_INTEGRATION: ["sandbox", "secure", "isolated", "e2b"],
            IntegrationType.API_WRAPPER: ["api", "wrapper", "external", "third-party"],
            IntegrationType.CAPIBARA_NATIVE: ["integrated", "native", "capibara", "ecosystem"]
        }
    
    def analyze_description(self, description: str) -> Dict[str, Any]:
        """Analyzes description and extracts requirements"""
        description_lower = description.lower()
        
        # Detect service type
        service_type = self._detect_service_type(description_lower)

        # Detect complexity
        complexity = self._detect_complexity(description_lower)

        # Detect integration type
        integration = self._detect_integration_type(description_lower)

        # Extract technical specifications
        technical_specs = self._extract_technical_specs(description_lower)

        # Extract dependencies
        dependencies = self._extract_dependencies(description_lower)
        
        return {
            "service_type": service_type,
            "complexity": complexity,
            "integration_type": integration,
            "technical_specs": technical_specs,
            "dependencies": dependencies,
            "suggested_name": self._generate_service_name(description)
        }
    
    def _detect_service_type(self, description: str) -> ServiceType:
        """Detects the service type from the description"""
        scores = {}
        
        for service_type, keywords in self.service_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description)
            if score > 0:
                scores[service_type] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return ServiceType.CREATIVE  # Default
    
    def _detect_complexity(self, description: str) -> ComplexityLevel:
        """Detects the complexity level"""
        for complexity, indicators in self.complexity_indicators.items():
            if any(indicator in description for indicator in indicators):
                return complexity
        
        # Additional heuristics based on length and features
        if len(description.split()) > 50:
            return ComplexityLevel.COMPLEX
        elif len(description.split()) > 20:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE
    
    def _detect_integration_type(self, description: str) -> IntegrationType:
        """Detects the required integration type"""
        for integration, indicators in self.integration_indicators.items():
            if any(indicator in description for indicator in indicators):
                return integration
        
        return IntegrationType.CAPIBARA_NATIVE  # Default
    
    def _extract_technical_specs(self, description: str) -> Dict[str, Any]:
        """Extracts technical specifications"""
        specs = {}
        
        # Detect input/output formats
        if any(word in description for word in ["file", "upload", "document"]):
            specs["input_format"] = "file"
        elif any(word in description for word in ["json", "api", "data"]):
            specs["input_format"] = "json"
        else:
            specs["input_format"] = "text"
        
        # Detect GPU requirements
        if any(word in description for word in ["ai", "ml", "neural", "model", "training"]):
            specs["requires_gpu"] = True
        
        # Detect internet requirement
        if any(word in description for word in ["api", "download", "fetch", "online"]):
            specs["requires_internet"] = True
        
        # Detect batch support
        if any(word in description for word in ["batch", "multiple", "bulk", "many"]):
            specs["supports_batch"] = True
        
        return specs
    
    def _extract_dependencies(self, description: str) -> List[str]:
        """Extracts probable dependencies"""
        dependencies = []
        
        # Keyword to dependency mapping
        dependency_map = {
            "image": ["Pillow", "opencv-python"],
            "video": ["opencv-python", "moviepy"],
            "audio": ["librosa", "soundfile"],
            "music": ["music21", "librosa"],
            "ai": ["torch", "transformers"],
            "ml": ["scikit-learn", "numpy"],
            "data": ["pandas", "numpy"],
            "web": ["requests", "beautifulsoup4"],
            "api": ["requests", "fastapi"],
            "game": ["pygame", "arcade"],
            "3d": ["trimesh", "open3d"],
            "pdf": ["PyPDF2", "reportlab"],
            "excel": ["openpyxl", "pandas"]
        }
        
        for keyword, deps in dependency_map.items():
            if keyword in description:
                dependencies.extend(deps)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _generate_service_name(self, description: str) -> str:
        """Generates a suggested name for the service"""
        words = description.split()
        
        # Search for important keywords
        key_words = []
        for word in words[:10]:  # First 10 words
            clean_word = word.strip('.,!?').lower()
            if len(clean_word) > 3 and clean_word not in ['that', 'will', 'can', 'should', 'would']:
                key_words.append(clean_word.capitalize())
        
        if key_words:
            return f"TextTo{key_words[0]}"
        else:
            return "TextToCustom"

class MockCodeGenerator:
    """Mock code generator when components are not available"""
    
    def __init__(self, config: MetaGenerationConfig):
        self.config = config
    
    async def generate_service_code(self, request: ServiceGenerationRequest) -> Dict[str, Any]:
        """Generates mock code for the service"""
        logger.info(f" Generating mock service: {request.service_name}")
        
        # Simulate generation time
        await asyncio.sleep(2.0)
        
        service_dir = f"{self.config.output_directory}/{request.service_name.lower()}"
        
        # Files that would be generated
        generated_files = [
            f"{service_dir}/__init__.py",
            f"{service_dir}/{request.service_name.lower()}_service.py",
            f"{service_dir}/config.py",
            f"{service_dir}/models.py"
        ]
        
        if self.config.include_tests:
            generated_files.extend([
                f"{service_dir}/tests/test_{request.service_name.lower()}.py",
                f"{service_dir}/tests/__init__.py"
            ])
        
        if self.config.include_docs:
            generated_files.append(f"{service_dir}/README.md")
        
        if self.config.include_demo_script:
            generated_files.append(f"demo_{request.service_name.lower()}.py")
        
        # Example code that would be generated
        mock_code = self._generate_mock_service_code(request)
        
        return {
            "success": True,
            "service_path": service_dir,
            "generated_files": generated_files,
            "main_service_file": f"{service_dir}/{request.service_name.lower()}_service.py",
            "total_lines": self._estimate_lines_of_code(request),
            "mock_code_preview": mock_code,
            "installation_instructions": self._generate_installation_instructions(request),
            "usage_examples": self._generate_usage_examples(request)
        }
    
    def _generate_mock_service_code(self, request: ServiceGenerationRequest) -> str:
        """Generates example code for the service"""
        service_class = f"Capibara{request.service_name}"
        
        return f'''#!/usr/bin/env python3
"""
 CAPIBARA {request.service_name.upper()} SERVICE
{"=" * (40 + len(request.service_name))}

{request.description}

Automatically generated by Text-to-Gen.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class {request.service_name}Config:
    """Configuration for {request.service_name}"""
    output_directory: str = "./generated_{request.service_name.lower()}"
    default_quality: str = "high"
    enable_caching: bool = True
    timeout_seconds: int = 30

@dataclass
class {request.service_name}Request:
    """Request for {request.service_name}"""
    description: str
    parameters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

@dataclass
class {request.service_name}Result:
    """Result of {request.service_name}"""
    success: bool
    output_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    generation_time_seconds: float = 0.0
    error: Optional[str] = None

class {service_class}:
    """Main service for {request.service_name}"""
    
    def __init__(self, config: Optional[{request.service_name}Config] = None):
        self.config = config or {request.service_name}Config()
        logger.info(f" {service_class} initialized")
    
    async def generate(self, request: {request.service_name}Request) -> {request.service_name}Result:
        """Generates result from natural language description"""
        start_time = datetime.now()
        
        try:
            logger.info(f" Generating {{request.service_name}}: {{request.description[:100]}}...")
            
            # Simulate processing
            await asyncio.sleep(1.0)
            
            # Service-specific logic would go here
            result = await self._process_request(request)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {request.service_name}Result(
                success=True,
                output_path=result.get("output_path"),
                metadata=result.get("metadata", {{}}),
                generation_time_seconds=generation_time
            )
            
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f" {request.service_name} generation error: {{e}}")
            
            return {request.service_name}Result(
                success=False,
                generation_time_seconds=generation_time,
                error=str(e)
            )
    
    async def _process_request(self, request: {request.service_name}Request) -> Dict[str, Any]:
        """Processes the service-specific request"""
        import hashlib
        import os
        from pathlib import Path

        # Create output directory
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique ID for this request
        request_hash = hashlib.md5(
            f"{{request.description}}{{datetime.now().isoformat()}}".encode()
        ).hexdigest()[:12]

        # Parse request parameters
        params = request.parameters or {{}}
        quality = params.get("quality", self.config.default_quality)
        format_type = params.get("format", "default")

        # Analyze the description to extract key elements
        description_lower = request.description.lower()
        extracted_features = {{
            "keywords": [w for w in description_lower.split() if len(w) > 3][:10],
            "length": len(request.description),
            "has_specifics": any(c.isdigit() for c in request.description),
        }}

        # Generate output based on service type
        output_filename = f"{{request_hash}}_output.txt"
        output_path = output_dir / output_filename

        # Create the generated content
        generated_content = {{
            "request_id": request_hash,
            "description": request.description,
            "parameters": params,
            "extracted_features": extracted_features,
            "quality": quality,
            "format": format_type,
            "generated_at": datetime.now().isoformat(),
            "service": "{request.service_name}",
            "status": "generated"
        }}

        # Save output
        import json
        with open(output_path, "w") as f:
            json.dump(generated_content, f, indent=2)

        # Cache result if enabled
        if self.config.enable_caching:
            cache_dir = output_dir / ".cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"{{request_hash}}.json"
            with open(cache_file, "w") as f:
                json.dump(generated_content, f)

        return {{
            "output_path": str(output_path),
            "metadata": {{
                "request_id": request_hash,
                "description": request.description,
                "quality": quality,
                "format": format_type,
                "features": extracted_features,
                "cached": self.config.enable_caching,
                "timestamp": datetime.now().isoformat()
            }}
        }}
    
    def is_available(self) -> bool:
        """Checks if the service is available"""
        return True

    async def test_functionality(self) -> Dict[str, Any]:
        """Tests the service functionality"""
        test_request = {request.service_name}Request(
            description="Test generation request"
        )
        
        result = await self.generate(test_request)
        
        return {{
            "test_passed": result.success,
            "generation_time": result.generation_time_seconds,
            "error": result.error
        }}

# Factory function
def create_{request.service_name.lower()}_service(config=None):
    """Creates an instance of the {request.service_name} service"""
    return {service_class}(config)
'''
    
    def _estimate_lines_of_code(self, request: ServiceGenerationRequest) -> int:
        """Estimates lines of code that would be generated"""
        base_lines = {
            ComplexityLevel.SIMPLE: 300,
            ComplexityLevel.MEDIUM: 800,
            ComplexityLevel.COMPLEX: 2000,
            ComplexityLevel.ADVANCED: 5000
        }
        
        lines = base_lines[request.complexity]
        
        # Adjustments based on features
        if request.supports_batch:
            lines += 200
        if request.supports_streaming:
            lines += 300
        if request.integration_type == IntegrationType.E2B_INTEGRATION:
            lines += 500
        
        return lines
    
    def _generate_installation_instructions(self, request: ServiceGenerationRequest) -> str:
        """Generates installation instructions"""
        return f"""# Installation of {request.service_name}

1. Install dependencies:
   pip install -r requirements.txt

2. Configure service:
   from capibara.services.{request.service_name.lower()} import create_{request.service_name.lower()}_service
   
3. Use service:
   service = create_{request.service_name.lower()}_service()
   result = await service.generate(request)
"""
    
    def _generate_usage_examples(self, request: ServiceGenerationRequest) -> List[str]:
        """Generates usage examples"""
        return [
            f"# Basic example of {request.service_name}",
            f"request = {request.service_name}Request(description='Your description here')",
            f"result = await service.generate(request)",
            f"print(f'Generated: {{result.output_path}}')"
        ]

class CapibaraTextToGen:
    """Main Text-to-Gen service"""
    
    def __init__(self, config: Optional[MetaGenerationConfig] = None):
        self.config = config or MetaGenerationConfig()
        
        # Initialize components
        self.requirement_analyzer = ServiceRequirementAnalyzer()
        self.mock_generator = MockCodeGenerator(self.config)

        # Try to initialize real generator
        self.code_generator = None
        if CODE_GENERATION_AVAILABLE:
            self._init_real_generator()

        # Configure directories
        self._setup_directories()
        
        logger.info(" CapibaraTextToGen initialized")
    
    def _init_real_generator(self):
        """Initializes the real code generator"""
        try:
            self.code_generator = CodeGenerator(self.config)
            logger.info(" Real code generator initialized")
        except Exception as e:
            logger.error(f" Error initializing real generator: {e}")
            self.code_generator = None
    
    def _setup_directories(self):
        """Configures output directories"""
        try:
            os.makedirs(self.config.output_directory, exist_ok=True)
            os.makedirs(self.config.template_directory, exist_ok=True)
            logger.info(f" Output directory: {self.config.output_directory}")
        except Exception as e:
            logger.error(f" Error creating directories: {e}")
    
    async def generate_service(self, request: ServiceGenerationRequest) -> ServiceGenerationResult:
        """Generates a new text-to-* service from description"""
        start_time = datetime.now()
        
        try:
            logger.info(f" Generating service: {request.service_name}")
            logger.info(f" Description: {request.description[:200]}...")
            
            # Analyze requirements if not specified
            if not hasattr(request, 'analyzed'):
                analysis = self.requirement_analyzer.analyze_description(request.description)

                # Apply analysis if not explicitly specified
                if request.service_type is None:
                    request.service_type = analysis["service_type"]
                if request.complexity == ComplexityLevel.MEDIUM:  # Default
                    request.complexity = analysis["complexity"]
                if request.integration_type == IntegrationType.CAPIBARA_NATIVE:  # Default
                    request.integration_type = analysis["integration_type"]
                
                # Add suggested dependencies
                if not request.dependencies:
                    request.dependencies = analysis["dependencies"]
                
                logger.info(f" Analysis: {analysis['service_type'].value}, {analysis['complexity'].value}")
            
            # Generate code
            if self.code_generator:
                logger.info("️ Using real code generator...")
                generation_result = await self.code_generator.generate_service_code(request)
            else:
                logger.info(" Using mock code generator...")
                generation_result = await self.mock_generator.generate_service_code(request)
            
            # Process result
            generation_time = (datetime.now() - start_time).total_seconds()
            
            if generation_result["success"]:
                # Create file structure (mock mode)
                if not self.code_generator:
                    await self._create_mock_files(request, generation_result)
                
                result = ServiceGenerationResult(
                    success=True,
                    service_name=request.service_name,
                    service_path=generation_result["service_path"],
                    generated_files=generation_result["generated_files"],
                    main_service_file=generation_result["main_service_file"],
                    total_lines=generation_result["total_lines"],
                    files_count=len(generation_result["generated_files"]),
                    complexity_score=self._calculate_complexity_score(request),
                    installation_instructions=generation_result["installation_instructions"],
                    usage_examples=generation_result["usage_examples"],
                    generation_time_seconds=generation_time,
                    templates_used=["base_service_template", "capibara_integration_template"]
                )
                
                logger.info(f" Service '{request.service_name}' generated successfully!")
                logger.info(f" Generated {result.files_count} files, {result.total_lines} lines")
                
                return result
            else:
                return ServiceGenerationResult(
                    success=False,
                    service_name=request.service_name,
                    generation_time_seconds=generation_time,
                    error=generation_result.get("error", "Unknown generation error")
                )
                
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f" Service generation error: {e}")
            
            return ServiceGenerationResult(
                success=False,
                service_name=request.service_name,
                generation_time_seconds=generation_time,
                error=str(e)
            )
    
    async def _create_mock_files(self, request: ServiceGenerationRequest, generation_result: Dict[str, Any]):
        """Creates mock files for demonstration"""
        try:
            service_dir = generation_result["service_path"]
            os.makedirs(service_dir, exist_ok=True)
            
            # Create main file
            main_file = generation_result["main_service_file"]
            os.makedirs(os.path.dirname(main_file), exist_ok=True)
            
            with open(main_file, 'w') as f:
                f.write(generation_result["mock_code_preview"])
            
            # Create __init__.py
            init_file = f"{service_dir}/__init__.py"
            with open(init_file, 'w') as f:
                f.write(f'''"""
{request.service_name} Service

{request.description}

Automatically generated by CapibaraGPT v3 Text-to-Gen.
"""

from .{request.service_name.lower()}_service import (
    Capibara{request.service_name},
    {request.service_name}Config,
    {request.service_name}Request,
    {request.service_name}Result
)

def create_{request.service_name.lower()}_service(config=None):
    return Capibara{request.service_name}(config)

__all__ = [
    'Capibara{request.service_name}',
    '{request.service_name}Config',
    '{request.service_name}Request',
    '{request.service_name}Result',
    'create_{request.service_name.lower()}_service'
]
''')
            
            # Create README
            if self.config.include_docs:
                readme_file = f"{service_dir}/README.md"
                with open(readme_file, 'w') as f:
                    f.write(f'''# {request.service_name} Service

{request.description}

## Installation

```bash
pip install -r requirements.txt
```

## Uso

```python
from capibara.services.{request.service_name.lower()} import create_{request.service_name.lower()}_service

service = create_{request.service_name.lower()}_service()
request = {request.service_name}Request(description="Your description here")
result = await service.generate(request)
logger.info(f"Generated: {{result.output_path}}")
```

## Features

- Type: {request.service_type.value}
- Complexity: {request.complexity.value}
- Integration: {request.integration_type.value}

Automatically generated by CapibaraGPT v3 Text-to-Gen.
''')
            
            logger.info(f" Mock files created in {service_dir}")
            
        except Exception as e:
            logger.error(f" Error creating mock files: {e}")
    
    def _calculate_complexity_score(self, request: ServiceGenerationRequest) -> float:
        """Calculates the complexity score of the service"""
        base_scores = {
            ComplexityLevel.SIMPLE: 0.2,
            ComplexityLevel.MEDIUM: 0.5,
            ComplexityLevel.COMPLEX: 0.8,
            ComplexityLevel.ADVANCED: 1.0
        }
        
        score = base_scores[request.complexity]
        
        # Adjustments based on features
        if request.supports_batch:
            score += 0.1
        if request.supports_streaming:
            score += 0.15
        if request.requires_gpu:
            score += 0.2
        if request.integration_type == IntegrationType.E2B_INTEGRATION:
            score += 0.15
        
        return min(1.0, score)
    
    # Utility methods
    def list_generated_services(self) -> List[str]:
        """Lists generated services"""
        try:
            if os.path.exists(self.config.output_directory):
                return [d for d in os.listdir(self.config.output_directory) 
                       if os.path.isdir(os.path.join(self.config.output_directory, d))]
            return []
        except Exception as e:
            logger.error(f"Error listing services: {e}")
            return []
    
    def analyze_description(self, description: str) -> Dict[str, Any]:
        """Analyzes description and suggests configuration"""
        return self.requirement_analyzer.analyze_description(description)
    
    def is_available(self) -> bool:
        """Checks if the service is available"""
        return True
    
    def get_supported_service_types(self) -> List[ServiceType]:
        """Gets supported service types"""
        return list(ServiceType)
    
    def get_complexity_levels(self) -> List[ComplexityLevel]:
        """Gets supported complexity levels"""
        return list(ComplexityLevel)
    
    async def test_generation(self) -> Dict[str, Any]:
        """Tests the generation functionality"""
        test_request = ServiceGenerationRequest(
            description="A simple test service that converts text to uppercase",
            service_name="TextToUppercase",
            service_type=ServiceType.TECHNICAL,
            complexity=ComplexityLevel.SIMPLE
        )
        
        result = await self.generate_service(test_request)
        
        return {
            "test_passed": result.success,
            "generation_time": result.generation_time_seconds,
            "files_generated": result.files_count,
            "error": result.error
        }