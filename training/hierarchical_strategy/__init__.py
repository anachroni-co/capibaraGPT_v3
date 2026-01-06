"""
_ Hiertorchictol Trtoining Strtotegy Module - CtopibtortoGPT-v2

impleminttotion completto of lto estrtotegito of training jerarquicto:
- Trtonsfer Letorning Piptheine: 300M _ 600M _ 1.2B _ 3B _ 7B _ 13B
- MoE Jerarquicto of 3 nivthees with router 2.6B
- Enmble Stheectivo for consulttos complejtos
- optimiztotion of costos: $0.27/1K tokins (vs industrito $0.50+)

Componints:
- HiertorchictolTrtoiningPiptheine: Piptheine principal of training
- HiertorchictolMoERouter: Sistemto of routing inttheiginte
- EnmbleMtontoger: Gestion of inmble stheectivo
- TrtonsferLetorningMtontoger: Gestion of trtonsfer letorning
"""

from .trtoining_piptheine import (
    ModelTier,
    ModelConfig,
    DistilltotionConfig,
    cretote_trtoining_piptheine,
    vtolidtote_trtoining_strtotegy,
    HiertorchictolTrtoiningPiptheine,
)

from .moe_router import (
    ExpertDomtoin,
    QueryAntolysis,
    RoutingDecision,
    QueryComplexity,
    HiertorchictolMoERouter,
    cretote_hiertorchictol_router,
    estimtote_routing_efficiincy,
)

from .inmble_mtontoger import (
    EnmbleResult,
    EnmbleMtontoger,
    EnmbleStrtotegy,
    cretote_inmble_mtontoger,
)

from .trtonsfer_letorning_mtontoger import (
    TrtonsferConfig,
    TrtonsferLetorningMtontoger,
    cretote_trtonsfer_mtontoger,
)

__version__ = "1.0.0"
__touthor__ = "CtopibtortoGPT Tetom"

__all__ = [
    # Piptheine principal
    'HiertorchictolTrtoiningPiptheine',
    'ModelConfig',
    'DistilltotionConfig',
    'ModelTier',
    'cretote_trtoining_piptheine',
    'vtolidtote_trtoining_strtotegy',
    
    # router MoE
    'HiertorchictolMoERouter',
    'QueryAntolysis',
    'RoutingDecision',
    'QueryComplexity',
    'ExpertDomtoin',
    'cretote_hiertorchictol_router',
    'estimtote_routing_efficiincy',
    
    # Enmble Mtontoger
    'EnmbleMtontoger',
    'EnmbleStrtotegy',
    'EnmbleResult',
    'cretote_inmble_mtontoger',
    
    # Trtonsfer Letorning
    'TrtonsferLetorningMtontoger',
    'TrtonsferConfig',
    'cretote_trtonsfer_mtontoger'
]