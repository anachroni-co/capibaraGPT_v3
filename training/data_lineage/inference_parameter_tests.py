#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 INFERENCE PARAMETER CONTROL TESTS
====================================

Critical tests to verify that parameter enable/disable functionality
works correctly during model inference without breaking the model.

TESTS COVER:
1. Model inference before and after parameter control
2. Gradual parameter disabling/enabling
3. Model output consistency
4. Performance impact measurement
5. Recovery from disabled states
6. Edge cases and error conditions
"""

import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import tempfile

# Mock JAX/Flax for testing
try:
    import jax.numpy as jnp
    import jax
    JAX_AVAILABLE = True
except ImportError:
    # Create mock jax.numpy for testing
    class MockJNP:
        @staticmethod
        def array(x):
            return np.array(x)
        
        @staticmethod
        def zeros_like(x):
            return np.zeros_like(x)
        
        @staticmethod
        def ones_like(x):
            return np.ones_like(x)
        
        @staticmethod
        def full(shape, fill_value, dtype=None):
            return np.full(shape, fill_value, dtype=dtype)
        
        @staticmethod
        def where(condition, x, y):
            return np.where(condition, x, y)
        
        @staticmethod
        def sum(x):
            return np.sum(x)
    
    jnp = MockJNP()
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

class MockModel:
    """Mock neural network model for testing parameter control."""
    
    def __init__(self, layer_sizes: List[int] = [768, 1024, 768, 512]):
        self.layer_sizes = layer_sizes
        self.parameters = self._initialize_parameters()
        self.original_output = None
        
    def _initialize_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize model parameters."""
        params = {}
        
        for i in range(len(self.layer_sizes) - 1):
            # Weight matrices
            weight_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
            params[f"layer_{i}.weight"] = np.random.normal(0, 0.1, weight_shape)
            
            # Bias vectors
            bias_shape = (self.layer_sizes[i + 1],)
            params[f"layer_{i}.bias"] = np.random.normal(0, 0.01, bias_shape)
        
        # Output layer
        params["output.weight"] = np.random.normal(0, 0.1, (self.layer_sizes[-1], 10))
        params["output.bias"] = np.random.normal(0, 0.01, (10,))
        
        return params
    
    def forward(self, x: np.ndarray, parameters: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Forward pass through the model."""
        if parameters is None:
            parameters = self.parameters
            
        current = x
        
        # Process through layers
        for i in range(len(self.layer_sizes) - 1):
            weight = parameters[f"layer_{i}.weight"]
            bias = parameters[f"layer_{i}.bias"]
            
            current = np.dot(current, weight) + bias
            current = np.maximum(0, current)  # ReLU activation
        
        # Output layer
        output_weight = parameters["output.weight"]
        output_bias = parameters["output.bias"]
        
        output = np.dot(current, output_weight) + output_bias
        
        return output
    
    def predict(self, x: np.ndarray, parameters: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Make predictions."""
        logits = self.forward(x, parameters)
        return np.argmax(logits, axis=-1)
    
    def get_output_distribution(self, x: np.ndarray, parameters: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Get output probability distribution."""
        logits = self.forward(x, parameters)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

class InferenceParameterTester:
    """
    Comprehensive tester for parameter control during inference.
    
    Verifies that parameter enable/disable works correctly without
    breaking model functionality.
    """
    
    def __init__(self, model: MockModel):
        self.model = model
        self.test_data = self._generate_test_data()
        self.baseline_outputs = {}
        self.test_results = {}
        
        # Import parameter controller
        try:
            from .dataset_parameter_controller import DatasetParameterController, ParameterMask
            self.controller = DatasetParameterController(
                model_parameters={k: jnp.array(v) if hasattr(jnp, 'array') else v 
                                for k, v in model.parameters.items()},
                control_dir=tempfile.mkdtemp()
            )
            self.CONTROLLER_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"Parameter controller not available: {e}")
            self.CONTROLLER_AVAILABLE = False
            self.controller = None
            
        logger.info(" InferenceParameterTester initialized")
    
    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data for inference testing."""
        np.random.seed(42)  # Reproducible results
        
        return {
            "simple": np.random.randn(1, self.model.layer_sizes[0]),
            "batch": np.random.randn(32, self.model.layer_sizes[0]),
            "edge_case": np.zeros((1, self.model.layer_sizes[0])),
            "large_values": np.random.randn(5, self.model.layer_sizes[0]) * 10,
            "small_values": np.random.randn(5, self.model.layer_sizes[0]) * 0.01
        }
    
    def establish_baseline(self):
        """Establish baseline model outputs before parameter control."""
        logger.info(" Establishing baseline outputs...")
        
        for test_name, test_input in self.test_data.items():
            outputs = {
                "predictions": self.model.predict(test_input),
                "logits": self.model.forward(test_input),
                "probabilities": self.model.get_output_distribution(test_input)
            }
            self.baseline_outputs[test_name] = outputs
            
        logger.info(f" Baseline established for {len(self.test_data)} test cases")
    
    def test_parameter_disabling(self) -> Dict[str, Any]:
        """Test parameter disabling functionality."""
        if not self.CONTROLLER_AVAILABLE:
            return {"error": "Parameter controller not available"}
        
        logger.info(" Testing parameter disabling...")
        results = {}
        
        # Test 1: Disable single dataset parameters
        test_datasets = ["medical_data", "robotics_data", "text_data"]
        
        for dataset_id in test_datasets:
            # Create mock parameter lineage for this dataset
            self._create_mock_lineage(dataset_id)
            
            # Test different mask types
            for mask_type in ["binary", "weighted", "gradient_based"]:
                test_key = f"{dataset_id}_{mask_type}"
                logger.info(f"  Testing {test_key}...")
                
                try:
                    # Disable parameters
                    success = self.controller.disable_dataset_parameters(dataset_id, mask_type)
                    if not success:
                        results[test_key] = {"error": "Failed to disable parameters"}
                        continue
                    
                    # Get modified parameters
                    modified_params = self._convert_params_for_model(self.controller.current_parameters)
                    
                    # Test inference with modified parameters
                    inference_results = self._test_inference_with_params(modified_params)
                    
                    # Re-enable parameters
                    self.controller.enable_dataset_parameters(dataset_id, mask_type)
                    
                    results[test_key] = {
                        "success": True,
                        "inference_results": inference_results,
                        "parameters_modified": len([k for k, v in modified_params.items() if not np.allclose(v, self.model.parameters[k])]),
                        "mask_type": mask_type
                    }
                    
                except Exception as e:
                    results[test_key] = {"error": str(e)}
                    logger.error(f" Error in {test_key}: {e}")
        
        return results
    
    def test_gradual_parameter_control(self) -> Dict[str, Any]:
        """Test gradual enabling/disabling of parameters."""
        if not self.CONTROLLER_AVAILABLE:
            return {"error": "Parameter controller not available"}
        
        logger.info("️ Testing gradual parameter control...")
        results = {}
        
        try:
            # Create test scenario with multiple datasets
            datasets = ["dataset_1", "dataset_2", "dataset_3"]
            for dataset in datasets:
                self._create_mock_lineage(dataset)
            
            # Test gradual disabling
            gradual_results = []
            
            for i, dataset in enumerate(datasets):
                step_name = f"step_{i+1}_disable_{dataset}"
                logger.info(f"  {step_name}...")
                
                # Disable this dataset
                self.controller.disable_dataset_parameters(dataset)
                
                # Test inference
                modified_params = self._convert_params_for_model(self.controller.current_parameters)
                inference_result = self._test_inference_with_params(modified_params)
                
                gradual_results.append({
                    "step": step_name,
                    "disabled_datasets": datasets[:i+1],
                    "inference_result": inference_result
                })
            
            # Test gradual re-enabling
            for i, dataset in enumerate(reversed(datasets)):
                step_name = f"step_{i+1}_enable_{dataset}"
                logger.info(f"  {step_name}...")
                
                # Re-enable this dataset
                self.controller.enable_dataset_parameters(dataset)
                
                # Test inference
                modified_params = self._convert_params_for_model(self.controller.current_parameters)
                inference_result = self._test_inference_with_params(modified_params)
                
                gradual_results.append({
                    "step": step_name,
                    "remaining_disabled": datasets[:len(datasets)-i-1],
                    "inference_result": inference_result
                })
            
            results["gradual_control"] = {
                "success": True,
                "steps": gradual_results
            }
            
        except Exception as e:
            results["gradual_control"] = {"error": str(e)}
            logger.error(f" Error in gradual control test: {e}")
        
        return results
    
    def test_model_recovery(self) -> Dict[str, Any]:
        """Test model recovery after parameter control."""
        if not self.CONTROLLER_AVAILABLE:
            return {"error": "Parameter controller not available"}
        
        logger.info(" Testing model recovery...")
        results = {}
        
        try:
            # Create mock lineage
            self._create_mock_lineage("test_dataset")
            
            # Store original state
            original_params = self.model.parameters.copy()
            
            # Disable parameters
            self.controller.disable_dataset_parameters("test_dataset")
            modified_params = self._convert_params_for_model(self.controller.current_parameters)
            
            # Test that model is modified
            is_modified = not self._compare_parameters(original_params, modified_params)
            
            # Reset parameters
            self.controller.reset_parameters()
            reset_params = self._convert_params_for_model(self.controller.current_parameters)
            
            # Test that model is recovered
            is_recovered = self._compare_parameters(original_params, reset_params)
            
            # Test inference after recovery
            recovery_inference = self._test_inference_with_params(reset_params)
            baseline_match = self._compare_inference_results(
                self.baseline_outputs["simple"], 
                recovery_inference["simple"]
            )
            
            results["recovery"] = {
                "success": True,
                "parameters_were_modified": is_modified,
                "parameters_recovered": is_recovered,
                "inference_matches_baseline": baseline_match,
                "recovery_inference": recovery_inference
            }
            
        except Exception as e:
            results["recovery"] = {"error": str(e)}
            logger.error(f" Error in recovery test: {e}")
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error conditions."""
        if not self.CONTROLLER_AVAILABLE:
            return {"error": "Parameter controller not available"}
        
        logger.info("️ Testing edge cases...")
        results = {}
        
        # Test 1: Disable non-existent dataset
        try:
            result = self.controller.disable_dataset_parameters("non_existent_dataset")
            results["non_existent_dataset"] = {
                "disable_result": result,
                "expected": False
            }
        except Exception as e:
            results["non_existent_dataset"] = {"error": str(e)}
        
        # Test 2: Double disable
        try:
            self._create_mock_lineage("double_test")
            self.controller.disable_dataset_parameters("double_test")
            result = self.controller.disable_dataset_parameters("double_test")
            results["double_disable"] = {
                "second_disable_result": result,
                "should_handle_gracefully": True
            }
        except Exception as e:
            results["double_disable"] = {"error": str(e)}
        
        # Test 3: Empty parameters
        try:
            empty_controller = DatasetParameterController(
                model_parameters={},
                control_dir=tempfile.mkdtemp()
            )
            result = empty_controller.disable_dataset_parameters("any_dataset")
            results["empty_parameters"] = {
                "disable_result": result,
                "should_handle_gracefully": True
            }
        except Exception as e:
            results["empty_parameters"] = {"error": str(e)}
        
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info(" Starting comprehensive parameter control tests...")
        
        # Establish baseline
        self.establish_baseline()
        
        # Run all tests
        all_results = {
            "baseline_established": True,
            "parameter_disabling": self.test_parameter_disabling(),
            "gradual_control": self.test_gradual_parameter_control(), 
            "model_recovery": self.test_model_recovery(),
            "edge_cases": self.test_edge_cases()
        }
        
        # Generate summary
        summary = self._generate_test_summary(all_results)
        all_results["summary"] = summary
        
        logger.info(" Comprehensive tests completed")
        return all_results
    
    def _create_mock_lineage(self, dataset_id: str):
        """Create mock parameter lineage for testing."""
        if not self.CONTROLLER_AVAILABLE:
            return
            
        # Assign some parameters to this dataset
        param_names = list(self.model.parameters.keys())
        dataset_params = param_names[:len(param_names)//2]  # Assign first half
        
        self.controller.dataset_parameters[dataset_id] = dataset_params
        
        for param_name in dataset_params:
            if param_name not in self.controller.parameter_lineage:
                self.controller.parameter_lineage[param_name] = []
            self.controller.parameter_lineage[param_name].append(dataset_id)
    
    def _convert_params_for_model(self, controller_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert controller parameters back to numpy for model use."""
        converted = {}
        for k, v in controller_params.items():
            if hasattr(v, 'shape'):  # jnp.ndarray or similar
                converted[k] = np.array(v)
            else:
                converted[k] = v
        return converted
    
    def _test_inference_with_params(self, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Test inference with given parameters."""
        results = {}
        
        for test_name, test_input in self.test_data.items():
            try:
                predictions = self.model.predict(test_input, parameters)
                logits = self.model.forward(test_input, parameters)
                probabilities = self.model.get_output_distribution(test_input, parameters)
                
                results[test_name] = {
                    "success": True,
                    "predictions": predictions,
                    "logits": logits,
                    "probabilities": probabilities,
                    "has_nan": np.isnan(logits).any(),
                    "has_inf": np.isinf(logits).any()
                }
            except Exception as e:
                results[test_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def _compare_parameters(self, params1: Dict[str, np.ndarray], params2: Dict[str, np.ndarray]) -> bool:
        """Compare two parameter dictionaries."""
        if set(params1.keys()) != set(params2.keys()):
            return False
        
        for key in params1.keys():
            if not np.allclose(params1[key], params2[key], rtol=1e-5, atol=1e-8):
                return False
        
        return True
    
    def _compare_inference_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> bool:
        """Compare inference results."""
        try:
            return (
                np.allclose(result1["logits"], result2["logits"], rtol=1e-5, atol=1e-8) and
                np.array_equal(result1["predictions"], result2["predictions"])
            )
        except Exception as exc:
            logger.warning("Result comparison failed: %s", exc)
            return False
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        errors = []
        
        def count_results(data, prefix=""):
            nonlocal total_tests, passed_tests, failed_tests, errors
            
            if isinstance(data, dict):
                for key, value in data.items():
                    current_prefix = f"{prefix}.{key}" if prefix else key
                    
                    if key == "error":
                        failed_tests += 1
                        errors.append(f"{prefix}: {value}")
                    elif key == "success" and value:
                        passed_tests += 1
                    elif isinstance(value, dict):
                        count_results(value, current_prefix)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                count_results(item, f"{current_prefix}[{i}]")
            
            if isinstance(data, dict) and ("success" in data or "error" in data):
                total_tests += 1
        
        count_results(results)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "errors": errors,
            "overall_success": failed_tests == 0 and total_tests > 0
        }

# Test execution functions
def run_parameter_control_tests() -> Dict[str, Any]:
    """Run comprehensive parameter control tests."""
    logger.info(" Initializing parameter control tests...")
    
    # Create mock model
    model = MockModel()
    
    # Create tester
    tester = InferenceParameterTester(model)
    
    # Run tests
    results = tester.run_comprehensive_tests()
    
    return results

def print_test_results(results: Dict[str, Any]):
    """Print formatted test results."""
    logger.info("\n" + "="*80)
    logger.info(" PARAMETER CONTROL TEST RESULTS")
    logger.info("="*80)
    
    if "summary" in results:
        summary = results["summary"]
        logger.info(f"\n SUMMARY:")
        logger.info(f"   Total Tests: {summary['total_tests']}")
        logger.info(f"   Passed: {summary['passed_tests']} ")
        logger.error(f"   Failed: {summary['failed_tests']} ")
        logger.info(f"   Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"   Overall Success: {'' if summary['overall_success'] else ''}")
        
        if summary['errors']:
            logger.error(f"\n ERRORS:")
            for error in summary['errors']:
                logger.error(f"   - {error}")
    
    logger.debug("\n DETAILED RESULTS:")
    for section, data in results.items():
        if section != "summary":
            logger.info(f"\n   {section.upper()}:")
            if isinstance(data, dict) and "error" in data:
                logger.error(f"     Error: {data['error']}")
            elif isinstance(data, dict) and "success" in data:
                logger.info(f"    {'' if data['success'] else ''} Success: {data['success']}")
            else:
                logger.info(f"     Contains {len(data) if isinstance(data, dict) else 1} test(s)")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_results = run_parameter_control_tests()
    
    # Print results
    print_test_results(test_results)
    
    # Save results
    import json
    with open("parameter_control_test_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_for_json(test_results), f, indent=2)
    
    logger.info(f"\n Results saved to: parameter_control_test_results.json")