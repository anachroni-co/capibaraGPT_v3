#!/usr/bin/env python3
"""
ARM Compatibility Validator for CapibaraGPT
Valida que los modelos sean compatibles with ARM Axion C4A

Autor: CapibaraGPT Team
Versión: 3.3.0
"""

import logging
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from jax import numpy as jnp
from capibara.core.arm_optimizations import ARMOptimizationSuite
from capibara.utils.system_info import SystemMonitor

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Niveles de validation ARM."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    EXHAUSTIVE = "exhaustive"

class ValidationResult(Enum):
    """Resultados de validation."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIP = "skip"

@dataclass
class ValidationConfig:
    """setup for validation ARM."""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_sve2_check: bool = True
    enable_memory_check: bool = True
    enable_quantization_check: bool = True
    enable_onnx_check: bool = True
    enable_performance_check: bool = True
    timeout_seconds: int = 300
    memory_threshold_gb: float = 32.0
    performance_threshold_percent: float = 10.0

@dataclass
class ValidationReport:
    """Reporte de validation ARM."""
    overall_result: ValidationResult
    validation_time: float
    checks_performed: List[str]
    checks_passed: List[str]
    checks_failed: List[str]
    checks_warnings: List[str]
    performance_metrics: Dict[str, Any]
    memory_metrics: Dict[str, Any]
    recommendations: List[str]
    system_info: Dict[str, Any]

class ARMCompatibilityValidator:
    """Validador de compatibilidad ARM Axion."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.system_monitor = SystemMonitor()
        self.arm_suite = ARMOptimizationSuite()
        
        # Métricas de validation
        self.validation_metrics = {
            "start_time": None,
            "end_time": None,
            "total_time": None,
            "checks_performed": [],
            "checks_passed": [],
            "checks_failed": [],
            "checks_warnings": []
        }
    
    def validate_model(self, model_path: str) -> ValidationReport:
        """Valida compatibilidad completa del model with ARM."""
        logger.info(f" Iniciando validación ARM para: {model_path}")
        
        self.validation_metrics["start_time"] = time.time()
        
        try:
            # carry model
            model_state = self._load_model(model_path)
            
            # execute validaciones
            validation_results = {}
            
            if self.config.enable_sve2_check:
                validation_results["sve2"] = self._validate_sve2_compatibility(model_state)
            
            if self.config.enable_memory_check:
                validation_results["memory"] = self._validate_memory_efficiency(model_state)
            
            if self.config.enable_quantization_check:
                validation_results["quantization"] = self._validate_quantization_readiness(model_state)
            
            if self.config.enable_onnx_check:
                validation_results["onnx"] = self._validate_onnx_compatibility(model_state)
            
            if self.config.enable_performance_check:
                validation_results["performance"] = self._validate_performance_optimization(model_state)
            
            # generate reporte
            report = self._generate_validation_report(validation_results)
            
            self.validation_metrics["end_time"] = time.time()
            self.validation_metrics["total_time"] = (
                self.validation_metrics["end_time"] - self.validation_metrics["start_time"]
            )
            
            logger.info(f" Validación ARM completada en {self.validation_metrics['total_time']:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f" Error en validación ARM: {e}")
            return self._generate_error_report(str(e))
    
    def _load_model(self, model_path: str) -> Dict[str, Any]:
        """load el model for validation."""
        try:
            model_file = Path(model_path)
            
            if model_file.suffix == '.jax':
                return self._load_jax_model(model_path)
            elif model_file.suffix == '.onnx':
                return self._load_onnx_model(model_path)
            elif model_file.suffix == '.tflite':
                return self._load_tflite_model(model_path)
            else:
                raise ValueError(f"Formato de modelo no soportado: {model_file.suffix}")
                
        except Exception as e:
            logger.error(f" Error cargando modelo: {e}")
            raise
    
    def _load_jax_model(self, model_path: str) -> Dict[str, Any]:
        """load model JAX."""
        try:
            import pickle
            
            with open(model_path, 'rb') as f:
                model_state = pickle.load(f)  # nosec B301 — trusted model
            
            logger.info(f" Modelo JAX cargado: {len(model_state.get('params', {}))} parámetros")
            return model_state
            
        except Exception as e:
            logger.error(f" Error cargando modelo JAX: {e}")
            raise
    
    def _load_onnx_model(self, model_path: str) -> Dict[str, Any]:
        """load model ONNX."""
        try:
            import onnx
            
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # Extraer information del model ONNX
            model_info = {
                "model_type": "onnx",
                "inputs": [input.name for input in model.graph.input],
                "outputs": [output.name for output in model.graph.output],
                "nodes_count": len(model.graph.node),
                "parameters_count": sum(len(node.input) for node in model.graph.node)
            }
            
            logger.info(f" Modelo ONNX cargado: {model_info['nodes_count']} nodos")
            return model_info
            
        except Exception as e:
            logger.error(f" Error cargando modelo ONNX: {e}")
            raise
    
    def _load_tflite_model(self, model_path: str) -> Dict[str, Any]:
        """load model TFLite."""
        try:
            import tensorflow as tf
            
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            model_info = {
                "model_type": "tflite",
                "inputs": [detail['name'] for detail in input_details],
                "outputs": [detail['name'] for detail in output_details],
                "input_shapes": [detail['shape'] for detail in input_details],
                "output_shapes": [detail['shape'] for detail in output_details]
            }
            
            logger.info(f" Modelo TFLite cargado: {len(input_details)} inputs, {len(output_details)} outputs")
            return model_info
            
        except Exception as e:
            logger.error(f" Error cargando modelo TFLite: {e}")
            raise
    
    def _validate_sve2_compatibility(self, model_state: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """Valida compatibilidad with SVE2."""
        logger.info(" Validando compatibilidad SVE2...")
        
        try:
            # verify if SVE2 está available
            sve_status = self.arm_suite.get_comprehensive_status()
            if not sve_status.get("sve_optimizations", {}).get("available", False):
                return ValidationResult.SKIP, "SVE2 no disponible en este sistema"
            
            # validate parámetros del model
            if "params" in model_state:
                params = model_state["params"]
                sve_compatible_count = 0
                total_params = 0
                
                for param_name, param_value in params.items():
                    if isinstance(param_value, jnp.ndarray):
                        total_params += 1
                        # verify que el size sea compatible with SVE2 (múltiplo de 16)
                        if param_value.size > 0 and param_value.size % 16 == 0:
                            sve_compatible_count += 1
                        else:
                            logger.warning(f"Parámetro {param_name} no es compatible con SVE2 (tamaño: {param_value.size})")
                
                compatibility_ratio = sve_compatible_count / total_params if total_params > 0 else 0
                
                if compatibility_ratio >= 0.95:
                    return ValidationResult.PASS, f"SVE2 compatible ({compatibility_ratio:.1%})"
                elif compatibility_ratio >= 0.8:
                    return ValidationResult.WARNING, f"SVE2 parcialmente compatible ({compatibility_ratio:.1%})"
                else:
                    return ValidationResult.FAIL, f"SVE2 no compatible ({compatibility_ratio:.1%})"
            
            return ValidationResult.PASS, "SVE2 compatible (sin parámetros para validar)"
            
        except Exception as e:
            logger.error(f" Error validando SVE2: {e}")
            return ValidationResult.FAIL, f"Error validando SVE2: {e}"
    
    def _validate_memory_efficiency(self, model_state: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """Valida eficiencia de memory."""
        logger.info(" Validando eficiencia de memoria...")
        
        try:
            # calculate uso de memory estimado
            estimated_memory_gb = 0
            
            if "params" in model_state:
                params = model_state["params"]
                total_params = 0
                
                for param_value in params.values():
                    if isinstance(param_value, jnp.ndarray):
                        # estimate memory (float32 = 4 bytes by parameter)
                        total_params += param_value.size
                
                estimated_memory_gb = (total_params * 4) / (1024**3)
            
            # obtain information del sistema
            system_info = self.system_monitor.get_system_info()
            available_memory_gb = system_info.get("memory_gb", 32.0)
            
            # validate against umbrales
            memory_usage_ratio = estimated_memory_gb / available_memory_gb
            
            if memory_usage_ratio <= 0.5:
                return ValidationResult.PASS, f"Memoria eficiente ({estimated_memory_gb:.1f}GB / {available_memory_gb:.1f}GB)"
            elif memory_usage_ratio <= 0.8:
                return ValidationResult.WARNING, f"Memoria moderada ({estimated_memory_gb:.1f}GB / {available_memory_gb:.1f}GB)"
            else:
                return ValidationResult.FAIL, f"Memoria insuficiente ({estimated_memory_gb:.1f}GB / {available_memory_gb:.1f}GB)"
            
        except Exception as e:
            logger.error(f" Error validando memoria: {e}")
            return ValidationResult.FAIL, f"Error validando memoria: {e}"
    
    def _validate_quantization_readiness(self, model_state: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """Valida preparación for cuantización."""
        logger.info(" Validando preparación para cuantización...")
        
        try:
            if "params" in model_state:
                params = model_state["params"]
                quantization_ready_count = 0
                total_params = 0
                
                for param_name, param_value in params.items():
                    if isinstance(param_value, jnp.ndarray):
                        total_params += 1
                        
                        # verify que not haya NaN or Inf
                        if not (jnp.any(jnp.isnan(param_value)) or jnp.any(jnp.isinf(param_value))):
                            # verify rank de valores
                            param_min = jnp.min(param_value)
                            param_max = jnp.max(param_value)
                            
                            # verify que el rank sea razonable for cuantización
                            if param_min >= -100 and param_max <= 100:
                                quantization_ready_count += 1
                            else:
                                logger.warning(f"Parámetro {param_name} tiene rango extremo: [{param_min:.3f}, {param_max:.3f}]")
                
                readiness_ratio = quantization_ready_count / total_params if total_params > 0 else 0
                
                if readiness_ratio >= 0.95:
                    return ValidationResult.PASS, f"Listo para cuantización ({readiness_ratio:.1%})"
                elif readiness_ratio >= 0.8:
                    return ValidationResult.WARNING, f"Parcialmente listo para cuantización ({readiness_ratio:.1%})"
                else:
                    return ValidationResult.FAIL, f"No listo para cuantización ({readiness_ratio:.1%})"
            
            return ValidationResult.PASS, "Listo para cuantización (sin parámetros para validar)"
            
        except Exception as e:
            logger.error(f" Error validando cuantización: {e}")
            return ValidationResult.FAIL, f"Error validando cuantización: {e}"
    
    def _validate_onnx_compatibility(self, model_state: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """Valida compatibilidad with ONNX."""
        logger.info(" Validando compatibilidad ONNX...")
        
        try:
            # verify if ONNX Runtime ARM está available
            onnx_status = self.arm_suite.get_comprehensive_status()
            if not onnx_status.get("onnx_runtime", {}).get("available", False):
                return ValidationResult.SKIP, "ONNX Runtime ARM no disponible"
            
            # validate operaciones del model
            if model_state.get("model_type") == "onnx":
                return ValidationResult.PASS, "Modelo ONNX nativo"
            elif model_state.get("model_type") == "tflite":
                return ValidationResult.PASS, "Modelo TFLite (compatible con ONNX)"
            else:
                # for modelos JAX, verify operaciones básicas
                if "params" in model_state:
                    return ValidationResult.PASS, "Modelo JAX (conversión a ONNX posible)"
                else:
                    return ValidationResult.WARNING, "Modelo con formato desconocido"
            
        except Exception as e:
            logger.error(f" Error validando ONNX: {e}")
            return ValidationResult.FAIL, f"Error validando ONNX: {e}"
    
    def _validate_performance_optimization(self, model_state: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """Valida optimizaciones de rendimiento."""
        logger.info(" Validando optimizaciones de rendimiento...")
        
        try:
            # obtain information del sistema ARM
            system_info = self.system_monitor.get_system_info()
            arm_optimizations = self.arm_suite.get_comprehensive_status()
            
            optimization_score = 0
            total_checks = 0
            
            # verify optimizaciones disponibles
            if arm_optimizations.get("sve_optimizations", {}).get("available", False):
                optimization_score += 1
            total_checks += 1
            
            if arm_optimizations.get("memory_pool", {}).get("available", False):
                optimization_score += 1
            total_checks += 1
            
            if arm_optimizations.get("quantization", {}).get("available", False):
                optimization_score += 1
            total_checks += 1
            
            if arm_optimizations.get("onnx_runtime", {}).get("available", False):
                optimization_score += 1
            total_checks += 1
            
            optimization_ratio = optimization_score / total_checks if total_checks > 0 else 0
            
            if optimization_ratio >= 0.75:
                return ValidationResult.PASS, f"Optimizaciones ARM disponibles ({optimization_ratio:.1%})"
            elif optimization_ratio >= 0.5:
                return ValidationResult.WARNING, f"Optimizaciones ARM limitadas ({optimization_ratio:.1%})"
            else:
                return ValidationResult.FAIL, f"Optimizaciones ARM insuficientes ({optimization_ratio:.1%})"
            
        except Exception as e:
            logger.error(f" Error validando rendimiento: {e}")
            return ValidationResult.FAIL, f"Error validando rendimiento: {e}"
    
    def _generate_validation_report(self, validation_results: Dict[str, Tuple[ValidationResult, str]]) -> ValidationReport:
        """Genera reporte de validation."""
        logger.info(" Generando reporte de validación...")
        
        # tell resultados
        passed = []
        failed = []
        warnings = []
        skipped = []
        
        for check_name, (result, message) in validation_results.items():
            self.validation_metrics["checks_performed"].append(check_name)
            
            if result == ValidationResult.PASS:
                passed.append(f"{check_name}: {message}")
                self.validation_metrics["checks_passed"].append(check_name)
            elif result == ValidationResult.FAIL:
                failed.append(f"{check_name}: {message}")
                self.validation_metrics["checks_failed"].append(check_name)
            elif result == ValidationResult.WARNING:
                warnings.append(f"{check_name}: {message}")
                self.validation_metrics["checks_warnings"].append(check_name)
            elif result == ValidationResult.SKIP:
                skipped.append(f"{check_name}: {message}")
        
        # determine result general
        if len(failed) == 0 and len(warnings) == 0:
            overall_result = ValidationResult.PASS
        elif len(failed) == 0:
            overall_result = ValidationResult.WARNING
        else:
            overall_result = ValidationResult.FAIL
        
        # generate recomendaciones
        recommendations = self._generate_recommendations(validation_results)
        
        # obtain métricas del sistema
        system_info = self.system_monitor.get_system_info()
        
        # obtain métricas de rendimiento
        performance_metrics = self._get_performance_metrics()
        memory_metrics = self._get_memory_metrics()
        
        return ValidationReport(
            overall_result=overall_result,
            validation_time=self.validation_metrics["total_time"] or 0,
            checks_performed=self.validation_metrics["checks_performed"],
            checks_passed=passed,
            checks_failed=failed,
            checks_warnings=warnings,
            performance_metrics=performance_metrics,
            memory_metrics=memory_metrics,
            recommendations=recommendations,
            system_info=system_info
        )
    
    def _generate_error_report(self, error_message: str) -> ValidationReport:
        """Genera reporte de error."""
        return ValidationReport(
            overall_result=ValidationResult.FAIL,
            validation_time=0,
            checks_performed=[],
            checks_passed=[],
            checks_failed=[f"Error general: {error_message}"],
            checks_warnings=[],
            performance_metrics={},
            memory_metrics={},
            recommendations=["Revisar configuración del sistema", "Verificar permisos de archivos"],
            system_info={}
        )
    
    def _generate_recommendations(self, validation_results: Dict[str, Tuple[ValidationResult, str]]) -> List[str]:
        """Genera recomendaciones basadas en los resultados de validation."""
        recommendations = []
        
        # Recomendaciones basadas en resultados específicos
        for check_name, (result, message) in validation_results.items():
            if result == ValidationResult.FAIL:
                if check_name == "sve2":
                    recommendations.append("Considerar reentrenamiento con tamaños de parámetros compatibles con SVE2")
                elif check_name == "memory":
                    recommendations.append("Reducir tamaño del modelo o usar cuantización")
                elif check_name == "quantization":
                    recommendations.append("Aplicar normalización de parámetros antes de cuantización")
                elif check_name == "onnx":
                    recommendations.append("Instalar ONNX Runtime ARM o usar formato alternativo")
                elif check_name == "performance":
                    recommendations.append("Instalar optimizaciones ARM adicionales")
            
            elif result == ValidationResult.WARNING:
                if check_name == "sve2":
                    recommendations.append("Optimizar algunos parámetros para mejor compatibilidad SVE2")
                elif check_name == "memory":
                    recommendations.append("Monitorear uso de memoria durante inferencia")
                elif check_name == "quantization":
                    recommendations.append("Considerar cuantización selectiva")
        
        # Recomendaciones generales
        if len([r for r in validation_results.values() if r[0] == ValidationResult.PASS]) >= 3:
            recommendations.append("Modelo bien optimizado para ARM Axion")
        
        if not recommendations:
            recommendations.append("Modelo compatible con ARM Axion")
        
        return recommendations
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento."""
        try:
            arm_status = self.arm_suite.get_comprehensive_status()
            return {
                "sve2_available": arm_status.get("sve_optimizations", {}).get("available", False),
                "memory_pool_available": arm_status.get("memory_pool", {}).get("available", False),
                "quantization_available": arm_status.get("quantization", {}).get("available", False),
                "onnx_runtime_available": arm_status.get("onnx_runtime", {}).get("available", False)
            }
        except Exception:
            return {}
    
    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de memory."""
        try:
            system_info = self.system_monitor.get_system_info()
            return {
                "total_memory_gb": system_info.get("memory_gb", 0),
                "available_memory_gb": system_info.get("available_memory_gb", 0),
                "memory_usage_percent": system_info.get("memory_usage_percent", 0)
            }
        except Exception:
            return {}
    
    def print_validation_report(self, report: ValidationReport):
        """Imprime reporte de validation en consola."""
        logger.info("=" * 60)
        logger.info(" REPORTE DE VALIDACIÓN ARM")
        logger.info("=" * 60)
        
        # result general
        status_emoji = {
            ValidationResult.PASS: "",
            ValidationResult.WARNING: "️",
            ValidationResult.FAIL: "",
            ValidationResult.SKIP: "️"
        }
        
        logger.info(f" Resultado General: {status_emoji[report.overall_result]} {report.overall_result.value.upper()}")
        logger.info(f"️ Tiempo de validación: {report.validation_time:.2f} segundos")
        logger.info(f" Validaciones realizadas: {len(report.checks_performed)}")
        
        # Detalles de validaciones
        logger.info("\n DETALLES DE VALIDACIÓN:")
        
        for check in report.checks_passed:
            logger.info(f"   {check}")
        
        for check in report.checks_warnings:
            logger.info(f"  ️ {check}")
        
        for check in report.checks_failed:
            logger.info(f"   {check}")
        
        # Métricas del sistema
        logger.info("\n INFORMACIÓN DEL SISTEMA:")
        for key, value in report.system_info.items():
            logger.info(f"  {key}: {value}")
        
        # Recomendaciones
        logger.info("\n RECOMENDACIONES:")
        for recommendation in report.recommendations:
            logger.info(f"  • {recommendation}")
        
        logger.info("=" * 60)

def validate_arm_compatibility(
    model_path: str,
    config: Optional[ValidationConfig] = None
) -> ValidationReport:
    """function de conveniencia for validate compatibilidad ARM."""
    if config is None:
        config = ValidationConfig()
    
    validator = ARMCompatibilityValidator(config)
    report = validator.validate_model(model_path)
    validator.print_validation_report(report)
    
    return report

def main():
    """function principal del script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ARM Compatibility Validator for CapibaraGPT"
    )
    
    parser.add_argument(
        "model_path",
        help="Ruta al modelo a validar"
    )
    
    parser.add_argument(
        "--level",
        choices=["basic", "standard", "strict", "exhaustive"],
        default="standard",
        help="Nivel de validación"
    )
    
    parser.add_argument(
        "--output",
        help="Archivo de salida para el reporte JSON"
    )
    
    parser.add_argument(
        "--disable-sve2",
        action="store_true",
        help="Deshabilitar validación SVE2"
    )
    
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Deshabilitar validación de memoria"
    )
    
    parser.add_argument(
        "--disable-quantization",
        action="store_true",
        help="Deshabilitar validación de cuantización"
    )
    
    parser.add_argument(
        "--disable-onnx",
        action="store_true",
        help="Deshabilitar validación ONNX"
    )
    
    parser.add_argument(
        "--disable-performance",
        action="store_true",
        help="Deshabilitar validación de rendimiento"
    )
    
    args = parser.parse_args()
    
    # create setup
    config = ValidationConfig(
        validation_level=ValidationLevel(args.level),
        enable_sve2_check=not args.disable_sve2,
        enable_memory_check=not args.disable_memory,
        enable_quantization_check=not args.disable_quantization,
        enable_onnx_check=not args.disable_onnx,
        enable_performance_check=not args.disable_performance
    )
    
    # execute validation
    report = validate_arm_compatibility(args.model_path, config)
    
    # export reporte if se especifica
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "overall_result": report.overall_result.value,
                "validation_time": report.validation_time,
                "checks_performed": report.checks_performed,
                "checks_passed": report.checks_passed,
                "checks_failed": report.checks_failed,
                "checks_warnings": report.checks_warnings,
                "performance_metrics": report.performance_metrics,
                "memory_metrics": report.memory_metrics,
                "recommendations": report.recommendations,
                "system_info": report.system_info
            }, f, indent=2)
        
        logger.info(f" Reporte exportado a: {args.output}")
    
    # Código de output
    if report.overall_result == ValidationResult.FAIL:
        sys.exit(1)
    elif report.overall_result == ValidationResult.WARNING:
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()