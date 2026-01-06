from dataclasses import dataclass
from enum import Enum

class HardwareType(Enum):
    TPU_V4 = "tpu_v4"
    ARM_AXION = "arm_axion"
    CPU = "cpu"

@dataclass
class AgeAdaptationConfig:
    hardware_type: HardwareType = HardwareType.TPU_V4
    use_tpu: bool = True
    use_arm_optimization: bool = False

    def validate(self) -> bool:
        if self.use_tpu and self.use_arm_optimization:
            raise ValueError("No se puede usar TPU y ARM simultaneamente")
        return True

DEFAULT_TPU_CONFIG = AgeAdaptationConfig(hardware_type=HardwareType.TPU_V4, use_tpu=True, use_arm_optimization=False)