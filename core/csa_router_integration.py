#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSA Router Integration Module

This module provides integration between the CSA Expert and the main router system.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# Import dependencies with fallbacks
try:
    from capibara.sub_models.csa_expert import CSAExpert, CSAExpertConfig
    from capibara.interfaces.isub_models import ExpertContext, ExpertResult
    CSA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CSA Expert not available: {e}")
    CSA_AVAILABLE = False

class CSARouterIntegration:
    """Integration layer between CSA Expert and main router."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.csa_expert = None
        self.enabled = self.config.get("csa_expert_enabled", True)
        
        if CSA_AVAILABLE and self.enabled:
            self._initialize_csa_expert()
    
    def _initialize_csa_expert(self):
        """Initialize the CSA Expert with configuration."""
        try:
            csa_config = CSAExpertConfig(
                max_hypotheses=self.config.get("max_hypotheses", 6),
                max_rollout_steps=self.config.get("max_rollout_steps", 3),
                min_plausibility=self.config.get("min_plausibility", 0.5),
                min_utility=self.config.get("min_utility", 0.5),
                enable_logging=self.config.get("enable_logging", True),
                cache_size=self.config.get("cache_size", 100),
                temperature=self.config.get("temperature", 0.7)
            )
            
            self.csa_expert = CSAExpert(csa_config)
            logger.info("CSA Expert initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CSA Expert: {e}")
            self.csa_expert = None
            self.enabled = False
    
    def should_activate_csa(self, input_data: Union[str, Dict[str, Any]], 
                           context: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if CSA Expert should be activated for this request."""
        if not self.enabled or not self.csa_expert:
            return False
        
        # Convert input to ExpertContext format
        expert_context = self._convert_to_expert_context(input_data, context)
        
        # Use CSA's own support detection
        return self.csa_expert.supports(expert_context)
    
    async def process_with_csa(self, input_data: Union[str, Dict[str, Any]], 
                              context: Optional[Dict[str, Any]] = None) -> Optional[ExpertResult]:
        """Process request with CSA Expert."""
        if not self.enabled or not self.csa_expert:
            return None
        
        try:
            expert_context = self._convert_to_expert_context(input_data, context)
            result = await self.csa_expert.process(expert_context)
            
            logger.info(f"CSA Expert processed request: success={result.success}, "
                       f"confidence={result.confidence:.3f}, time={result.processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"CSA processing failed: {e}")
            return ExpertResult(
                success=False,
                output=[],
                confidence=0.0,
                processing_time=0.0,
                expert_name="CSA",
                error_message=str(e)
            )
    
    def _convert_to_expert_context(self, input_data: Union[str, Dict[str, Any]], 
                                  context: Optional[Dict[str, Any]] = None) -> ExpertContext:
        """Convert router input format to ExpertContext."""
        if isinstance(input_data, str):
            text = input_data
            task_hint = "general"
        elif isinstance(input_data, dict):
            text = input_data.get("text", "")
            task_hint = input_data.get("task_hint", "general")
        else:
            text = str(input_data)
            task_hint = "general"
        
        # Extract additional information from context
        constraints = {}
        flags = {}
        metadata = {}
        
        if context:
            constraints = context.get("constraints", {})
            flags = context.get("flags", {})
            metadata = context.get("metadata", {})
            
            # Infer task hint from context if not explicitly provided
            if task_hint == "general" and "task_type" in context:
                task_hint = context["task_type"]
        
        # Auto-detect task hint from text content
        if task_hint == "general":
            task_hint = self._infer_task_hint(text)
        
        return ExpertContext(
            text=text,
            task_hint=task_hint,
            constraints=constraints,
            flags=flags,
            metadata=metadata
        )
    
    def _infer_task_hint(self, text: str) -> str:
        """Infer task hint from text content."""
        text_lower = text.lower()
        
        # Diagnostic keywords
        if any(word in text_lower for word in 
               ["error", "falla", "problema", "diagnóstico", "diagnosis", "troubleshoot"]):
            return "diagnosis"
        
        # Planning keywords  
        if any(word in text_lower for word in 
               ["plan", "estrategia", "strategy", "cronograma", "schedule", "proyecto"]):
            return "planning"
        
        # Risk analysis keywords
        if any(word in text_lower for word in 
               ["riesgo", "risk", "peligro", "danger", "seguridad", "safety"]):
            return "risk_analysis"
        
        # Optimization keywords
        if any(word in text_lower for word in 
               ["optimiz", "mejora", "improve", "eficien", "rendimiento", "performance"]):
            return "optimization"
        
        # Design keywords
        if any(word in text_lower for word in 
               ["diseño", "design", "arquitectura", "architecture", "modelo", "model"]):
            return "design"
        
        return "general"
    
    def format_csa_results_for_router(self, csa_result: ExpertResult, 
                                     base_response: str = "") -> Dict[str, Any]:
        """Format CSA results for integration with router response."""
        if not csa_result.success or not csa_result.output:
            return {
                "enhanced_response": base_response,
                "csa_used": False,
                "csa_error": csa_result.error_message
            }
        
        # Format counterfactual results
        cf_results = csa_result.output
        alternatives = []
        
        for i, cf_result in enumerate(cf_results[:3]):  # Top 3 results
            alternative = {
                "scenario": f"Escenario {i+1}",
                "hypothesis": cf_result.hypothesis.delta,
                "consequences": cf_result.rollout.consequences,
                "plausibility": f"{cf_result.plausibility:.2f}",
                "utility": f"{cf_result.utility:.2f}", 
                "confidence": f"{cf_result.hypothesis.confidence:.2f}",
                "risk_assessment": f"{cf_result.risk_assessment:.2f}"
            }
            alternatives.append(alternative)
        
        # Create enhanced response
        if alternatives:
            csa_section = "\n\n**Análisis Contrafactual (CSA):**\n"
            csa_section += f"*Generados {len(alternatives)} escenarios alternativos con confianza {csa_result.confidence:.2f}*\n\n"
            
            for alt in alternatives:
                csa_section += f"**{alt['scenario']}**: {alt['hypothesis']}\n"
                csa_section += f"→ {alt['consequences']}\n"
                csa_section += f"*Plausibilidad: {alt['plausibility']}, Utilidad: {alt['utility']}, Riesgo: {alt['risk_assessment']}*\n\n"
            
            enhanced_response = base_response + csa_section
        else:
            enhanced_response = base_response
        
        return {
            "enhanced_response": enhanced_response,
            "csa_used": True,
            "csa_confidence": csa_result.confidence,
            "csa_processing_time": csa_result.processing_time,
            "csa_scenarios_count": len(alternatives),
            "csa_alternatives": alternatives
        }
    
    def get_csa_metrics(self) -> Dict[str, Any]:
        """Get CSA Expert metrics."""
        if not self.csa_expert:
            return {"csa_available": False}
        
        metrics = self.csa_expert.get_metrics()
        metrics["csa_available"] = True
        metrics["csa_enabled"] = self.enabled
        
        return metrics