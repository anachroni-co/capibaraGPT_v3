"""
 CAPIBARA TEXT-TO-GENOMIC SERVICE
==================================

Servicio revolucionario de análisis genómico con IA para el modelo 60B.
El sistema de bioinformática más avanzado del mundo con integración completa.

Capacidades:
- Análisis de variantes genéticas (SNPs, INDELs, CNVs, SVs)
- Predicción de estructura y función proteica
- Diseño de primers y sondas moleculares
- Análisis filogenético y evolutivo
- Anotación funcional de genomas
- Identificación de dianas terapéuticas
- Análisis de expresión génica
- Genómica poblacional y epidemiológica
- Medicina de precisión y farmacogenómica
- Análisis multi-ómico integrado

Integración:
- AlphaGenome (DeepMind) - IA genómica de última generación
- AlphaFold3 - Predicción de estructura proteica
- BioPython - Suite completa de bioinformática
- BLAST+ - Búsqueda avanzada de secuencias
- NCBI/GenBank - Bases de datos genómicas
- UniProt - Base de datos de proteínas
- Ensembl - Anotación genómica
- TCGA - Atlas del genoma del cáncer
- gnomAD - Agregación de genomas globales
- ClinVar - Variantes clínicas

Hardware optimizado para modelo 60B:
- TPU v6e-64 clusters para cálculos masivos
- 2TB HBM para datasets genómicos
- Paralelización avanzada para análisis multi-genoma
"""

from .text_to_genomic_service import (
    CapibaraTextToGenomic,
    GenomicAnalysisConfig,
    GenomicRequest,
    GenomicResult,
    AnalysisType,
    GenomeAssembly,
    OrganismType,
    SequenceType
)

from .genomic_analyzer import (
    GenomicAnalyzer,
    VariantAnalyzer,
    ProteinAnalyzer,
    PhylogeneticAnalyzer,
    StructuralAnalyzer,
    PopulationAnalyzer
)

from .sequence_processor import (
    SequenceProcessor,
    SequenceValidator,
    SequenceTranslator,
    ORFFinder,
    MotifSearcher,
    RepeatFinder
)

from .bioinformatics_tools import (
    BioinformaticsToolkit,
    BLASTRunner,
    PrimerDesigner,
    MultipleSequenceAligner,
    TreeBuilder,
    GeneAnnotator
)

from .genomic_visualizer import (
    GenomicVisualizer,
    SequencePlotter,
    PhylogeneticTreeRenderer,
    VariantMapRenderer,
    ProteinStructureRenderer,
    GenomeTrackRenderer
)

# AlphaGenome Integration Components
try:
    from .alphagenome_integration import (
        AlphaGenomeProcessor,
        AlphaFoldPredictor,
        DeepMindGenomicsAPI,
        AdvancedGenomicPredictor
    )
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False

# Advanced Research Tools
try:
    from .research_genomics import (
        CancerGenomicsAnalyzer,
        PharmacoGenomicsPredictor,
        PersonalizedMedicineEngine,
        MultiOmicsIntegrator,
        EpigenomicsAnalyzer
    )
    RESEARCH_TOOLS_AVAILABLE = True
except ImportError:
    RESEARCH_TOOLS_AVAILABLE = False

# High-Performance Computing Components
try:
    from .hpc_genomics import (
        DistributedGenomicProcessor,
        TPUGenomicAccelerator,
        MassiveGenomeAnalyzer,
        PopulationScaleProcessor
    )
    HPC_GENOMICS_AVAILABLE = True
except ImportError:
    HPC_GENOMICS_AVAILABLE = False

# Service factories
def create_genomic_analysis_service(config=None):
    """Crea instancia principal del servicio Text-to-Genomic"""
    return CapibaraTextToGenomic(config)

def create_genomic_analyzer(analysis_type="comprehensive"):
    """Crea analizador genómico especializado"""
    return GenomicAnalyzer(analysis_type)

def create_sequence_processor(organism="homo_sapiens"):
    """Crea procesador de secuencias optimizado"""
    return SequenceProcessor(organism)

def create_bioinformatics_toolkit():
    """Crea toolkit completo de bioinformática"""
    return BioinformaticsToolkit()

def create_genomic_visualizer():
    """Crea visualizador de datos genómicos"""
    return GenomicVisualizer()

# Advanced factory functions
def create_alphagenome_processor():
    """Crea procesador AlphaGenome (requiere credenciales DeepMind)"""
    if ALPHAGENOME_AVAILABLE:
        return AlphaGenomeProcessor()
    else:
        raise ImportError("AlphaGenome integration not available")

def create_research_genomics_suite():
    """Crea suite completa de herramientas de investigación"""
    if RESEARCH_TOOLS_AVAILABLE:
        return {
            "cancer_genomics": CancerGenomicsAnalyzer(),
            "pharmacogenomics": PharmacoGenomicsPredictor(),
            "personalized_medicine": PersonalizedMedicineEngine(),
            "multi_omics": MultiOmicsIntegrator(),
            "epigenomics": EpigenomicsAnalyzer()
        }
    else:
        raise ImportError("Research genomics tools not available")

def create_hpc_genomic_processor():
    """Crea procesador HPC para análisis masivos (modelo 60B)"""
    if HPC_GENOMICS_AVAILABLE:
        return DistributedGenomicProcessor()
    else:
        raise ImportError("HPC genomics components not available")

__all__ = [
    'CapibaraTextToGenomic',
    'GenomicAnalysisConfig',
    'GenomicRequest',
    'GenomicResult',
    'AnalysisType',
    'GenomeAssembly',
    'OrganismType',
    'SequenceType',
    'GenomicAnalyzer',
    'SequenceProcessor',
    'BioinformaticsToolkit',
    'GenomicVisualizer',
    'create_genomic_analysis_service',
    'create_genomic_analyzer',
    'create_sequence_processor',
    'create_bioinformatics_toolkit',
    'create_genomic_visualizer'
]

# Add advanced components to exports if available
if ALPHAGENOME_AVAILABLE:
    __all__.extend([
        'AlphaGenomeProcessor',
        'AlphaFoldPredictor',
        'DeepMindGenomicsAPI',
        'create_alphagenome_processor',
        'ALPHAGENOME_AVAILABLE'
    ])

if RESEARCH_TOOLS_AVAILABLE:
    __all__.extend([
        'CancerGenomicsAnalyzer',
        'PharmacoGenomicsPredictor',
        'PersonalizedMedicineEngine',
        'MultiOmicsIntegrator',
        'EpigenomicsAnalyzer',
        'create_research_genomics_suite',
        'RESEARCH_TOOLS_AVAILABLE'
    ])

if HPC_GENOMICS_AVAILABLE:
    __all__.extend([
        'DistributedGenomicProcessor',
        'TPUGenomicAccelerator',
        'MassiveGenomeAnalyzer',
        'PopulationScaleProcessor',
        'create_hpc_genomic_processor',
        'HPC_GENOMICS_AVAILABLE'
    ])