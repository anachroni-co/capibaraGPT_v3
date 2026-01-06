"""
Hardware Compatibility Adapter

Automatically detects available hardware and adapts system configuration
to optimize performance based on specific hardware capabilities
(TPU, GPU, CPU, memory, etc.).
"""

import logging
import platform
import subprocess
import re
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import time

from .base_adapter import BaseAdapter, AdapterConfig
from .adapter_registry import register_adapter_decorator, AdapterType

logger = logging.getLogger(__name__)

class HardwareType(Enum):
    """Detectable hardware types."""
    TPU_V4 = "tpu_v4"
    TPU_V5 = "tpu_v5"
    TPU_V6 = "tpu_v6"
    GPU_NVIDIA = "gpu_nvidia"
    GPU_AMD = "gpu_amd"
    CPU_INTEL = "cpu_intel"
    CPU_AMD = "cpu_amd"
    CPU_ARM = "cpu_arm"
    MEMORY_DDR4 = "memory_ddr4"
    MEMORY_DDR5 = "memory_ddr5"
    MEMORY_HBM = "memory_hbm"
    STORAGE_NVME = "storage_nvme"
    STORAGE_SSD = "storage_ssd"
    NETWORK_INFINIBAND = "network_infiniband"
    NETWORK_ETHERNET = "network_ethernet"

class OptimizationLevel(Enum):
    """Hardware optimization levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class HardwareCapability:
    """Specific hardware capability."""
    hardware_type: HardwareType
    name: str
    version: str = ""
    memory_gb: float = 0.0
    compute_units: int = 0
    peak_performance_tflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    power_consumption_watts: float = 0.0
    supported_precisions: List[str] = field(default_factory=list)
    special_features: List[str] = field(default_factory=list)
    driver_version: str = ""
    availability_score: float = 1.0  # 0-1, where 1 = fully available

@dataclass
class HardwareProfile:
    """Complete system hardware profile."""
    system_name: str
    detection_timestamp: float
    capabilities: List[HardwareCapability]
    total_memory_gb: float = 0.0
    total_compute_tflops: float = 0.0
    system_architecture: str = ""
    operating_system: str = ""
    optimization_recommendations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HardwareOptimization:
    """Hardware-specific optimization."""
    target_hardware: HardwareType
    optimization_type: str
    parameter_name: str
    recommended_value: Any
    expected_improvement: float  # Expected improvement percentage
    compatibility_requirements: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high

class HardwareDetector:
    """System hardware detector."""

    def __init__(self):
        self.detection_cache = {}
        self.cache_ttl = 300  # 5 minutes

    def detect_all_hardware(self) -> HardwareProfile:
        """Detects all available system hardware."""
        detection_start = time.time()
        
        capabilities = []

        # Detect TPUs
        tpu_capabilities = self._detect_tpus()
        capabilities.extend(tpu_capabilities)

        # Detect GPUs
        gpu_capabilities = self._detect_gpus()
        capabilities.extend(gpu_capabilities)

        # Detect CPUs
        cpu_capabilities = self._detect_cpus()
        capabilities.extend(cpu_capabilities)

        # Detect memory
        memory_capabilities = self._detect_memory()
        capabilities.extend(memory_capabilities)

        # Detect storage
        storage_capabilities = self._detect_storage()
        capabilities.extend(storage_capabilities)

        # Detect network
        network_capabilities = self._detect_network()
        capabilities.extend(network_capabilities)

        # Calculate totals
        total_memory = sum(cap.memory_gb for cap in capabilities)
        total_compute = sum(cap.peak_performance_tflops for cap in capabilities)

        # Create profile
        profile = HardwareProfile(
            system_name=platform.node(),
            detection_timestamp=detection_start,
            capabilities=capabilities,
            total_memory_gb=total_memory,
            total_compute_tflops=total_compute,
            system_architecture=platform.machine(),
            operating_system=platform.system()
        )

        # Generate optimization recommendations
        profile.optimization_recommendations = self._generate_optimization_recommendations(profile)

        detection_time = time.time() - detection_start
        logger.info(f"Hardware detection completed in {detection_time:.2f}s")
        logger.info(f"Detected {len(capabilities)} hardware capabilities")

        return profile

    def _detect_tpus(self) -> List[HardwareCapability]:
        """Detects available TPUs."""
        capabilities = []
        
        try:
            # Try detecting TPUs using JAX
            import jax
            tpu_devices = [d for d in jax.devices() if d.platform == 'tpu']

            for i, device in enumerate(tpu_devices):
                # Determine TPU version
                device_kind = getattr(device, 'device_kind', 'unknown')
                tpu_version = self._parse_tpu_version(device_kind)

                capability = HardwareCapability(
                    hardware_type=tpu_version,
                    name=f"TPU {i}",
                    version=device_kind,
                    memory_gb=32.0,  # Typical TPU v4
                    compute_units=1,
                    peak_performance_tflops=275.0,  # TPU v4
                    memory_bandwidth_gbps=1200.0,
                    supported_precisions=["bfloat16", "float32", "int8"],
                    special_features=["xla_optimization", "systolic_arrays"],
                    availability_score=1.0
                )
                capabilities.append(capability)

        except ImportError:
            logger.debug("JAX not available for TPU detection")
        except Exception as e:
            logger.debug(f"TPU detection failed: {e}")

        # Try alternative detection using gcloud
        try:
            result = subprocess.run(['gcloud', 'compute', 'tpus', 'list'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'READY' in result.stdout:
                # Parse TPU info from gcloud output
                tpu_info = self._parse_gcloud_tpu_output(result.stdout)
                for info in tpu_info:
                    capabilities.append(info)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("gcloud not available for TPU detection")
        except Exception as e:
            logger.debug(f"gcloud TPU detection failed: {e}")
        
        return capabilities
    
    def _detect_gpus(self) -> List[HardwareCapability]:
        """Detects available GPUs."""
        capabilities = []

        # Detect NVIDIA GPUs
        nvidia_gpus = self._detect_nvidia_gpus()
        capabilities.extend(nvidia_gpus)

        # Detect AMD GPUs
        amd_gpus = self._detect_amd_gpus()
        capabilities.extend(amd_gpus)

        return capabilities

    def _detect_nvidia_gpus(self) -> List[HardwareCapability]:
        """Detects NVIDIA GPUs."""
        capabilities = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get additional information
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"

                # Estimate performance based on GPU name
                peak_tflops = self._estimate_gpu_performance(name)
                
                capability = HardwareCapability(
                    hardware_type=HardwareType.GPU_NVIDIA,
                    name=name,
                    version=compute_capability,
                    memory_gb=memory_info.total / (1024**3),
                    compute_units=1,
                    peak_performance_tflops=peak_tflops,
                    memory_bandwidth_gbps=self._estimate_gpu_bandwidth(name),
                    supported_precisions=["float32", "float16", "int8"],
                    special_features=["cuda", "tensor_cores"] if "RTX" in name or "A100" in name else ["cuda"],
                    availability_score=1.0
                )
                capabilities.append(capability)
                
        except ImportError:
            logger.debug("pynvml not available for NVIDIA GPU detection")
        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")
        
        return capabilities
    
    def _detect_amd_gpus(self) -> List[HardwareCapability]:
        """Detects AMD GPUs."""
        capabilities = []
        
        try:
            # Try using rocm-smi to detect AMD GPUs
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'GPU' in line and ':' in line:
                        gpu_name = line.split(':', 1)[1].strip()
                        
                        capability = HardwareCapability(
                            hardware_type=HardwareType.GPU_AMD,
                            name=gpu_name,
                            memory_gb=8.0,  # Default estimation
                            compute_units=1,
                            peak_performance_tflops=10.0,  # Estimación
                            supported_precisions=["float32", "float16"],
                            special_features=["rocm"],
                            availability_score=0.8  # Menor que NVIDIA por compatibilidad
                        )
                        capabilities.append(capability)
                        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("rocm-smi not available for AMD GPU detection")
        except Exception as e:
            logger.debug(f"AMD GPU detection failed: {e}")
        
        return capabilities
    
    def _detect_cpus(self) -> List[HardwareCapability]:
        """Detects available CPUs."""
        capabilities = []
        
        try:
            import psutil
            
            # Información básica del CPU
            cpu_info = platform.processor()
            cpu_count = psutil.cpu_count(logical=False)
            logical_count = psutil.cpu_count(logical=True)
            
            # Determine CPU type
            cpu_type = HardwareType.CPU_INTEL
            if 'AMD' in cpu_info.upper():
                cpu_type = HardwareType.CPU_AMD
            elif 'ARM' in cpu_info.upper() or platform.machine().lower().startswith('arm'):
                cpu_type = HardwareType.CPU_ARM
            
            # Obtener frecuencia
            try:
                freq_info = psutil.cpu_freq()
                max_freq_ghz = freq_info.max / 1000 if freq_info else 3.0
            except:
                max_freq_ghz = 3.0
            
            # Estimar rendimiento
            estimated_tflops = self._estimate_cpu_performance(cpu_info, cpu_count, max_freq_ghz)
            
            capability = HardwareCapability(
                hardware_type=cpu_type,
                name=cpu_info or f"{cpu_count}-core CPU",
                compute_units=cpu_count,
                peak_performance_tflops=estimated_tflops,
                supported_precisions=["float32", "float64", "int32", "int64"],
                special_features=self._detect_cpu_features(),
                availability_score=1.0
            )
            capabilities.append(capability)
            
        except ImportError:
            logger.debug("psutil not available for CPU detection")
        except Exception as e:
            logger.debug(f"CPU detection failed: {e}")
        
        return capabilities
    
    def _detect_memory(self) -> List[HardwareCapability]:
        """Detects system memory."""
        capabilities = []
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            # Determine memory type (basic estimation)
            memory_type = HardwareType.MEMORY_DDR4  # By default
            
            capability = HardwareCapability(
                hardware_type=memory_type,
                name="System Memory",
                memory_gb=memory.total / (1024**3),
                memory_bandwidth_gbps=50.0,  # Estimación DDR4
                availability_score=1.0
            )
            capabilities.append(capability)
            
        except ImportError:
            logger.debug("psutil not available for memory detection")
        except Exception as e:
            logger.debug(f"Memory detection failed: {e}")
        
        return capabilities
    
    def _detect_storage(self) -> List[HardwareCapability]:
        """Detects system storage."""
        capabilities = []
        
        try:
            import psutil
            
            for disk in psutil.disk_partitions():
                if disk.fstype:  # Only partitions with filesystem
                    usage = psutil.disk_usage(disk.mountpoint)
                    
                    # Determine storage type (basic heuristic)
                    storage_type = HardwareType.STORAGE_SSD
                    if '/dev/nvme' in disk.device:
                        storage_type = HardwareType.STORAGE_NVME
                    
                    capability = HardwareCapability(
                        hardware_type=storage_type,
                        name=f"Storage {disk.device}",
                        memory_gb=usage.total / (1024**3),
                        special_features=["persistent_storage"],
                        availability_score=0.9
                    )
                    capabilities.append(capability)
                    break  # Solo el primer disco para evitar duplicados
                    
        except ImportError:
            logger.debug("psutil not available for storage detection")
        except Exception as e:
            logger.debug(f"Storage detection failed: {e}")
        
        return capabilities
    
    def _detect_network(self) -> List[HardwareCapability]:
        """Detects network interfaces."""
        capabilities = []
        
        try:
            import psutil
            
            for interface, addresses in psutil.net_if_addrs().items():
                if interface != 'lo':  # Exclude loopback
                    # Detect network type (basic heuristic)
                    network_type = HardwareType.NETWORK_ETHERNET
                    if 'ib' in interface.lower():
                        network_type = HardwareType.NETWORK_INFINIBAND
                    
                    capability = HardwareCapability(
                        hardware_type=network_type,
                        name=f"Network {interface}",
                        memory_bandwidth_gbps=1.0,  # 1 Gbps por defecto
                        special_features=["networking"],
                        availability_score=0.8
                    )
                    capabilities.append(capability)
                    break  # Solo la primera interfaz
                    
        except ImportError:
            logger.debug("psutil not available for network detection")
        except Exception as e:
            logger.debug(f"Network detection failed: {e}")
        
        return capabilities
    
    def _parse_tpu_version(self, device_kind: str) -> HardwareType:
        """Parses TPU version from device kind."""
        if 'v4' in device_kind.lower():
            return HardwareType.TPU_V4
        elif 'v5' in device_kind.lower():
            return HardwareType.TPU_V5
        elif 'v6' in device_kind.lower():
            return HardwareType.TPU_V6
        else:
            return HardwareType.TPU_V4  # By default
    
    def _estimate_gpu_performance(self, gpu_name: str) -> float:
        """Estimates GPU performance based on name."""
        name_upper = gpu_name.upper()
        
        # Estimates based on known models
        if 'A100' in name_upper:
            return 312.0  # TFLOPS para A100
        elif 'V100' in name_upper:
            return 125.0
        elif 'RTX 4090' in name_upper:
            return 83.0
        elif 'RTX 3090' in name_upper:
            return 71.0
        elif 'RTX' in name_upper:
            return 30.0  # Generic estimation RTX
        elif 'GTX' in name_upper:
            return 15.0  # Generic estimation GTX
        else:
            return 10.0  # Default estimation
    
    def _estimate_gpu_bandwidth(self, gpu_name: str) -> float:
        """Estimates GPU memory bandwidth."""
        name_upper = gpu_name.upper()
        
        if 'A100' in name_upper:
            return 1555.0  # GB/s
        elif 'V100' in name_upper:
            return 900.0
        elif 'RTX 4090' in name_upper:
            return 1008.0
        elif 'RTX 3090' in name_upper:
            return 936.0
        else:
            return 500.0  # Default estimation
    
    def _estimate_cpu_performance(self, cpu_info: str, cores: int, freq_ghz: float) -> float:
        """Estimates CPU performance."""
        # Very basic estimation: cores * frequency * factor
        base_performance = cores * freq_ghz * 0.1  # Conservative factor
        
        # Architecture-based adjustments
        if 'Intel' in cpu_info and ('i7' in cpu_info or 'i9' in cpu_info or 'Xeon' in cpu_info):
            base_performance *= 1.2
        elif 'AMD' in cpu_info and ('Ryzen' in cpu_info or 'EPYC' in cpu_info):
            base_performance *= 1.1
        
        return base_performance
    
    def _detect_cpu_features(self) -> List[str]:
        """Detects special CPU features."""
        features = []
        
        try:
            # Detectar AVX, AVX2, etc.
            if platform.machine() in ['x86_64', 'AMD64']:
                features.extend(['sse', 'avx'])  # Assume basic support
            
            # Detectar ARM features
            if platform.machine().lower().startswith('arm'):
                features.append('neon')
            
        except Exception:
            pass
        
        return features
    
    def _parse_gcloud_tpu_output(self, output: str) -> List[HardwareCapability]:
        """Parses gcloud TPU list output."""
        capabilities = []
        
        lines = output.strip().split('\n')[1:]  # Skip header
        for line in lines:
            if 'READY' in line:
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    zone = parts[1]
                    accelerator_type = parts[2]
                    
                    # Determinar versión de TPU
                    tpu_version = HardwareType.TPU_V4
                    if 'v5' in accelerator_type:
                        tpu_version = HardwareType.TPU_V5
                    elif 'v6' in accelerator_type:
                        tpu_version = HardwareType.TPU_V6
                    
                    capability = HardwareCapability(
                        hardware_type=tpu_version,
                        name=f"TPU {name}",
                        version=accelerator_type,
                        memory_gb=32.0,
                        peak_performance_tflops=275.0,
                        special_features=["cloud_tpu", "xla_optimization"],
                        availability_score=0.9
                    )
                    capabilities.append(capability)
        
        return capabilities
    
    def _generate_optimization_recommendations(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Generates optimization recommendations based on hardware profile."""
        recommendations = {
            'kernel_backend_priority': [],
            'memory_optimizations': [],
            'compute_optimizations': [],
            'precision_recommendations': []
        }
        
        # Backend priority based on available hardware
        for capability in profile.capabilities:
            if capability.hardware_type in [HardwareType.TPU_V4, HardwareType.TPU_V5, HardwareType.TPU_V6]:
                recommendations['kernel_backend_priority'].append('tpu')
            elif capability.hardware_type in [HardwareType.GPU_NVIDIA, HardwareType.GPU_AMD]:
                recommendations['kernel_backend_priority'].append('gpu')
        
        if not recommendations['kernel_backend_priority']:
            recommendations['kernel_backend_priority'].append('cpu')
        
        # Memory recommendations
        total_memory = profile.total_memory_gb
        if total_memory < 16:
            recommendations['memory_optimizations'].extend([
                'enable_gradient_checkpointing',
                'use_memory_efficient_attention',
                'reduce_batch_size'
            ])
        elif total_memory > 64:
            recommendations['memory_optimizations'].extend([
                'increase_batch_size',
                'enable_model_parallelism'
            ])
        
        # Precision recommendations
        has_tensor_cores = any('tensor_cores' in cap.special_features for cap in profile.capabilities)
        if has_tensor_cores:
            recommendations['precision_recommendations'].append('bfloat16')
        else:
            recommendations['precision_recommendations'].append('float32')
        
        return recommendations

@register_adapter_decorator(
    adapter_type=AdapterType.HARDWARE_COMPATIBILITY,
    priority=85,
    capabilities=["hardware_detection", "automatic_optimization", "multi_platform_support"],
    metadata={"version": "1.0", "supports_all_hardware_types": True}
)
class HardwareCompatibilityAdapter(BaseAdapter):
    """
    Adapter que detecta automáticamente el hardware disponible
    and optimizes the system configuration para máximo rendimiento.
    """
    
    def __init__(self, 
                 config: Optional[AdapterConfig] = None,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        super().__init__(config)
        self.optimization_level = optimization_level
        self.detector = HardwareDetector()
        self.hardware_profile: Optional[HardwareProfile] = None
        self.optimization_cache: Dict[str, List[HardwareOptimization]] = {}
        self.last_detection_time = 0.0
        self.detection_interval = 600.0  # 10 minutos
        
    def _initialize_impl(self) -> bool:
        """Initializes adapter by detecting hardware."""
        try:
            # Detect initial hardware
            self.hardware_profile = self.detector.detect_all_hardware()
            self.last_detection_time = time.time()
            
            # Generate initial optimizations
            self._generate_optimizations()
            
            self.logger.info(f"Hardware compatibility adapter initialized")
            self.logger.info(f"Detected {len(self.hardware_profile.capabilities)} hardware components")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware compatibility adapter: {e}")
            return False
    
    def _execute_impl(self, operation_type: str = "detect", *args, **kwargs) -> Dict[str, Any]:
        """Executes hardware compatibility adapter operations."""
        if operation_type == "detect":
            return self._execute_detection()
        elif operation_type == "optimize":
            return self._execute_optimization(*args, **kwargs)
        elif operation_type == "get_profile":
            return self._get_hardware_profile()
        elif operation_type == "get_recommendations":
            return self._get_optimization_recommendations()
        else:
            return {"error": f"Unknown operation: {operation_type}"}
    
    def _execute_detection(self) -> Dict[str, Any]:
        """Executes hardware detection."""
        current_time = time.time()
        
        # Re-detect if enough time has passed
        if (current_time - self.last_detection_time) > self.detection_interval:
            self.hardware_profile = self.detector.detect_all_hardware()
            self.last_detection_time = current_time
            self._generate_optimizations()
            
            return {
                "status": "detection_completed",
                "hardware_profile": self._serialize_profile(self.hardware_profile),
                "detection_time": current_time
            }
        else:
            return {
                "status": "using_cached_profile",
                "hardware_profile": self._serialize_profile(self.hardware_profile),
                "last_detection": self.last_detection_time
            }
    
    def _execute_optimization(self, target_component: str = "", **kwargs) -> Dict[str, Any]:
        """Executes hardware optimizations."""
        if not self.hardware_profile:
            return {"error": "Hardware profile not available"}
        
        applied_optimizations = []
        
        # Get relevant optimizations
        optimizations = self.optimization_cache.get(target_component, [])
        if not optimizations and target_component:
            # Generate component-specific optimizations
            optimizations = self._generate_component_optimizations(target_component)
        
        # Apply optimizations according to level
        max_optimizations = self._get_max_optimizations_for_level()
        
        for optimization in optimizations[:max_optimizations]:
            if self._should_apply_optimization(optimization):
                success = self._apply_optimization(optimization)
                if success:
                    applied_optimizations.append({
                        "type": optimization.optimization_type,
                        "parameter": optimization.parameter_name,
                        "value": optimization.recommended_value,
                        "expected_improvement": optimization.expected_improvement
                    })
        
        return {
            "status": "optimization_completed",
            "applied_optimizations": applied_optimizations,
            "optimization_level": self.optimization_level.value
        }
    
    def _get_hardware_profile(self) -> Dict[str, Any]:
        """Gets complete hardware profile."""
        if not self.hardware_profile:
            return {"error": "Hardware profile not available"}
        
        return self._serialize_profile(self.hardware_profile)
    
    def _get_optimization_recommendations(self) -> Dict[str, Any]:
        """Gets optimization recommendations."""
        if not self.hardware_profile:
            return {"error": "Hardware profile not available"}
        
        return {
            "recommendations": self.hardware_profile.optimization_recommendations,
            "optimization_level": self.optimization_level.value,
            "cached_optimizations": {
                component: len(opts) for component, opts in self.optimization_cache.items()
            }
        }
    
    def _generate_optimizations(self):
        """Generates optimizations based on hardware profile."""
        if not self.hardware_profile:
            return
        
        self.optimization_cache.clear()
        
        # Optimizaciones para kernels
        kernel_optimizations = self._generate_kernel_optimizations()
        self.optimization_cache["kernel"] = kernel_optimizations
        
        # Optimizaciones para memoria
        memory_optimizations = self._generate_memory_optimizations()
        self.optimization_cache["memory"] = memory_optimizations
        
        # Optimizaciones para compute
        compute_optimizations = self._generate_compute_optimizations()
        self.optimization_cache["compute"] = compute_optimizations
        
        self.logger.info(f"Generated {sum(len(opts) for opts in self.optimization_cache.values())} optimizations")
    
    def _generate_kernel_optimizations(self) -> List[HardwareOptimization]:
        """Generates kernel-specific optimizations."""
        optimizations = []
        
        # Check available TPUs
        tpu_capabilities = [cap for cap in self.hardware_profile.capabilities 
                           if cap.hardware_type in [HardwareType.TPU_V4, HardwareType.TPU_V5, HardwareType.TPU_V6]]
        
        if tpu_capabilities:
            optimizations.append(HardwareOptimization(
                target_hardware=tpu_capabilities[0].hardware_type,
                optimization_type="kernel_backend_selection",
                parameter_name="preferred_backend",
                recommended_value="tpu",
                expected_improvement=40.0,
                compatibility_requirements=["jax", "tpu_driver"],
                risk_level="low"
            ))
            
            optimizations.append(HardwareOptimization(
                target_hardware=tpu_capabilities[0].hardware_type,
                optimization_type="precision_optimization",
                parameter_name="default_precision",
                recommended_value="bfloat16",
                expected_improvement=25.0,
                compatibility_requirements=["tpu"],
                risk_level="low"
            ))
        
        # Check NVIDIA GPUs
        nvidia_gpus = [cap for cap in self.hardware_profile.capabilities 
                      if cap.hardware_type == HardwareType.GPU_NVIDIA]
        
        if nvidia_gpus:
            gpu_cap = nvidia_gpus[0]
            if "tensor_cores" in gpu_cap.special_features:
                optimizations.append(HardwareOptimization(
                    target_hardware=HardwareType.GPU_NVIDIA,
                    optimization_type="precision_optimization",
                    parameter_name="use_tensor_cores",
                    recommended_value=True,
                    expected_improvement=30.0,
                    compatibility_requirements=["cuda", "tensor_cores"],
                    risk_level="low"
                ))
        
        return optimizations
    
    def _generate_memory_optimizations(self) -> List[HardwareOptimization]:
        """Generates memory optimizations."""
        optimizations = []
        
        total_memory = self.hardware_profile.total_memory_gb
        
        if total_memory < 16:
            optimizations.extend([
                HardwareOptimization(
                    target_hardware=HardwareType.MEMORY_DDR4,
                    optimization_type="memory_efficiency",
                    parameter_name="enable_gradient_checkpointing",
                    recommended_value=True,
                    expected_improvement=40.0,
                    risk_level="low"
                ),
                HardwareOptimization(
                    target_hardware=HardwareType.MEMORY_DDR4,
                    optimization_type="batch_size_optimization",
                    parameter_name="max_batch_size",
                    recommended_value=16,
                    expected_improvement=20.0,
                    risk_level="medium"
                )
            ])
        elif total_memory > 64:
            optimizations.append(HardwareOptimization(
                target_hardware=HardwareType.MEMORY_DDR4,
                optimization_type="batch_size_optimization",
                parameter_name="max_batch_size",
                recommended_value=128,
                expected_improvement=15.0,
                risk_level="low"
            ))
        
        return optimizations
    
    def _generate_compute_optimizations(self) -> List[HardwareOptimization]:
        """Generates compute optimizations."""
        optimizations = []
        
        # CPU-based optimizations
        cpu_capabilities = [cap for cap in self.hardware_profile.capabilities 
                           if cap.hardware_type in [HardwareType.CPU_INTEL, HardwareType.CPU_AMD, HardwareType.CPU_ARM]]
        
        if cpu_capabilities:
            cpu_cap = cpu_capabilities[0]
            
            if cpu_cap.compute_units >= 8:
                optimizations.append(HardwareOptimization(
                    target_hardware=cpu_cap.hardware_type,
                    optimization_type="parallelization",
                    parameter_name="num_threads",
                    recommended_value=min(cpu_cap.compute_units, 16),
                    expected_improvement=cpu_cap.compute_units * 5,
                    risk_level="low"
                ))
            
            if "avx" in cpu_cap.special_features:
                optimizations.append(HardwareOptimization(
                    target_hardware=cpu_cap.hardware_type,
                    optimization_type="vectorization",
                    parameter_name="enable_avx",
                    recommended_value=True,
                    expected_improvement=20.0,
                    compatibility_requirements=["avx_support"],
                    risk_level="low"
                ))
        
        return optimizations
    
    def _generate_component_optimizations(self, component: str) -> List[HardwareOptimization]:
        """Generates component-specific optimizations."""
        all_optimizations = []
        for opts in self.optimization_cache.values():
            all_optimizations.extend(opts)
        
        # Filtrar por componente relevante
        relevant_optimizations = [opt for opt in all_optimizations 
                                if component.lower() in opt.optimization_type.lower()]
        
        return relevant_optimizations
    
    def _get_max_optimizations_for_level(self) -> int:
        """Gets maximum number of optimizations according to level."""
        level_limits = {
            OptimizationLevel.CONSERVATIVE: 2,
            OptimizationLevel.BALANCED: 5,
            OptimizationLevel.AGGRESSIVE: 10,
            OptimizationLevel.MAXIMUM: 20
        }
        return level_limits.get(self.optimization_level, 5)
    
    def _should_apply_optimization(self, optimization: HardwareOptimization) -> bool:
        """Determines if an optimization should be applied."""
        # Verificar nivel de riesgo vs nivel de optimización
        risk_tolerance = {
            OptimizationLevel.CONSERVATIVE: ["low"],
            OptimizationLevel.BALANCED: ["low", "medium"],
            OptimizationLevel.AGGRESSIVE: ["low", "medium", "high"],
            OptimizationLevel.MAXIMUM: ["low", "medium", "high"]
        }
        
        allowed_risks = risk_tolerance.get(self.optimization_level, ["low"])
        return optimization.risk_level in allowed_risks
    
    def _apply_optimization(self, optimization: HardwareOptimization) -> bool:
        """Applies a specific optimization."""
        try:
            # En un system real, esto interactuaría con los componentes correspondientes
            self.logger.info(f"Applying optimization: {optimization.optimization_type} = {optimization.recommended_value}")
            
            # Simulación de aplicación exitosa
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization {optimization.optimization_type}: {e}")
            return False
    
    def _serialize_profile(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Serializa un perfil de hardware para JSON."""
        return {
            "system_name": profile.system_name,
            "detection_timestamp": profile.detection_timestamp,
            "total_memory_gb": profile.total_memory_gb,
            "total_compute_tflops": profile.total_compute_tflops,
            "system_architecture": profile.system_architecture,
            "operating_system": profile.operating_system,
            "capabilities": [
                {
                    "hardware_type": cap.hardware_type.value,
                    "name": cap.name,
                    "version": cap.version,
                    "memory_gb": cap.memory_gb,
                    "compute_units": cap.compute_units,
                    "peak_performance_tflops": cap.peak_performance_tflops,
                    "memory_bandwidth_gbps": cap.memory_bandwidth_gbps,
                    "supported_precisions": cap.supported_precisions,
                    "special_features": cap.special_features,
                    "availability_score": cap.availability_score
                } for cap in profile.capabilities
            ],
            "optimization_recommendations": profile.optimization_recommendations
        }
    
    def set_optimization_level(self, level: OptimizationLevel):
        """Changes optimization level."""
        self.optimization_level = level
        self._generate_optimizations()  # Re-generar optimizaciones
        self.logger.info(f"Optimization level changed to: {level.value}")
    
    def force_hardware_detection(self) -> Dict[str, Any]:
        """Forces new hardware detection."""
        self.hardware_profile = self.detector.detect_all_hardware()
        self.last_detection_time = time.time()
        self._generate_optimizations()
        
        return self._serialize_profile(self.hardware_profile)
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Gets detected hardware summary."""
        if not self.hardware_profile:
            return {"error": "Hardware profile not available"}
        
        summary = {
            "total_components": len(self.hardware_profile.capabilities),
            "total_memory_gb": self.hardware_profile.total_memory_gb,
            "total_compute_tflops": self.hardware_profile.total_compute_tflops,
            "hardware_types": {},
            "optimization_opportunities": len(sum(self.optimization_cache.values(), []))
        }
        
        # Count hardware types
        for capability in self.hardware_profile.capabilities:
            hw_type = capability.hardware_type.value
            if hw_type not in summary["hardware_types"]:
                summary["hardware_types"][hw_type] = 0
            summary["hardware_types"][hw_type] += 1
        
        return summary


# Global adapter instance
hardware_adapter = HardwareCompatibilityAdapter()

# Funciones de utilidad
def detect_system_hardware():
    """Detects system hardware."""
    if hardware_adapter.get_status().value == "ready":
        return hardware_adapter.execute("detect")
    else:
        return {"error": "hardware_adapter_not_ready"}

def get_hardware_optimizations(component: str = ""):
    """Gets hardware optimizations."""
    if hardware_adapter.get_status().value == "ready":
        return hardware_adapter.execute("optimize", target_component=component)
    else:
        return {"error": "hardware_adapter_not_ready"}

def get_system_capabilities():
    """Gets system capabilities."""
    if hardware_adapter.get_status().value == "ready":
        return hardware_adapter.get_hardware_summary()
    else:
        return {"error": "hardware_adapter_not_ready"}