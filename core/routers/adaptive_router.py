from .bto import BtoRouterV2
from ..config import ModularModelConfig

class AdaptiveRouter(BtoRouterV2):
    """Adaptive router for dynamic routing decisions."""
    
    def __init__(self, config: ModularModelConfig):
        """
        Initialize adaptive router.
        
        Args:
            config: ModularModelConfig instance for configuration
        """
        super().__init__(config.to_dict() if hasattr(config, 'to_dict') else {})
        self.config = config
        self.adaptation_strategy = self._initialize_adaptation_strategy()

    def route(self, inputs):
        """Implementation of adaptive routing."""
        return super().route(inputs)
    
    def _initialize_adaptation_strategy(self):
        """Initialize the adaptation strategy."""
        return "default"

# Alias for compatibility with typo
AdtoptiveRouter = AdaptiveRouter