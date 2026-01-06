"""
Concrete Observer Implementations
=================================

This module provides concrete implementations of request observers for different
patterns and use cases. Each observer specializes in detecting specific types
of requests that require expert activation.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque

from .request_observer import (
    RequestObserver,
    RequestEvent,
    RequestEventType,
    ExpertActivationEvent,
    create_expert_activation_event
)

logger = logging.getLogger(__name__)


class RequestPatternObserver(RequestObserver):
    """
    Observer that detects specific patterns in request text to activate experts.
    """
    
    def __init__(self, name: str = "PatternObserver", priority: int = 1):
        super().__init__(name, priority)
        
        # Pattern definitions for different experts
        self.expert_patterns = {
            "CSA": [
                # Spanish patterns
                r"(?i)\b(si|supongamos|qué pasaría si|en caso de que|alternativamente)\b",
                r"(?i)\b(hipótesis|escenario|suponer|asumir)\b",
                r"(?i)\b(contrafactual|análisis de escenarios)\b",
                # English patterns
                r"(?i)\b(what if|suppose|assuming|alternatively|hypothesis)\b",
                r"(?i)\b(scenario|counterfactual|if we|consider the case)\b",
                r"(?i)\b(diagnostic|troubleshoot|root cause|failure analysis)\b",
            ],
            "MathExpert": [
                r"(?i)\b(calcul[ao]|matemáticas?|ecuación|fórmula|algebra)\b",
                r"(?i)\b(solve|equation|formula|mathematics|algebra|calculus)\b",
                r"(?i)\b(derivad[ao]|integral|límite|función)\b",
                r"\b\d+[\+\-\*/\^]\d+\b",  # Mathematical expressions
            ],
            "CodeExpert": [
                r"(?i)\b(código|programar?|algoritmo|función|debug)\b",
                r"(?i)\b(code|program|algorithm|function|debug|implementation)\b",
                r"(?i)\b(python|javascript|java|c\+\+|rust|go)\b",
                r"```[\s\S]*?```",  # Code blocks
            ],
            "SpanishExpert": [
                r"(?i)\b(traducir|translation|español|spanish|castellano)\b",
                r"(?i)\b(idioma|language|multilingual|bilingüe)\b",
                # Spanish-specific patterns
                r"\b(el|la|los|las)\s+\w+\s+(es|son|está|están)\b",
                r"\b(por favor|gracias|de nada|lo siento)\b",
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for expert, patterns in self.expert_patterns.items():
            self.compiled_patterns[expert] = [re.compile(pattern) for pattern in patterns]
    
    async def observe(self, event: RequestEvent) -> List[ExpertActivationEvent]:
        """Observe request and activate experts based on pattern matching."""
        if not self.should_activate(event):
            return []
        
        activations = []
        request_text = event.request_text or ""
        
        # Check patterns for each expert
        for expert_name, patterns in self.compiled_patterns.items():
            matches = []
            confidence = 0.0
            
            for pattern in patterns:
                pattern_matches = pattern.findall(request_text)
                if pattern_matches:
                    matches.extend(pattern_matches)
                    confidence += 0.2  # Increase confidence for each matching pattern
            
            if matches:
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
                
                # Create activation event
                activation = create_expert_activation_event(
                    expert_name=expert_name,
                    reason=f"Pattern matching: {len(matches)} matches found",
                    confidence=confidence,
                    priority=1 if confidence > 0.7 else 2,
                    patterns=[str(match) for match in matches[:5]],  # Limit to 5 matches
                    context={
                        "pattern_matches": len(matches),
                        "matched_patterns": [str(match) for match in matches],
                        "task_hint": self._infer_task_hint(expert_name)
                    }
                )
                activations.append(activation)
        
        return activations
    
    def should_activate(self, event: RequestEvent) -> bool:
        """Activate for REQUEST_RECEIVED events with text content."""
        return (
            event.event_type == RequestEventType.REQUEST_RECEIVED and
            event.request_text is not None and
            len(event.request_text.strip()) > 10  # Minimum text length
        )
    
    def _infer_task_hint(self, expert_name: str) -> str:
        """Infer task hint based on expert name."""
        task_hints = {
            "CSA": "diagnosis",
            "MathExpert": "calculation",
            "CodeExpert": "programming",
            "SpanishExpert": "translation"
        }
        return task_hints.get(expert_name, "analysis")
    
    def get_supported_event_types(self) -> Set[RequestEventType]:
        """Support REQUEST_RECEIVED events."""
        return {RequestEventType.REQUEST_RECEIVED}


class ComplexityObserver(RequestObserver):
    """
    Observer that activates experts based on request complexity analysis.
    """
    
    def __init__(self, name: str = "ComplexityObserver", priority: int = 2):
        super().__init__(name, priority)
        
        # Complexity indicators
        self.complexity_indicators = {
            "length": {"weight": 0.1, "threshold": 200},
            "technical_terms": {"weight": 0.3, "threshold": 3},
            "question_complexity": {"weight": 0.2, "threshold": 2},
            "domain_specificity": {"weight": 0.4, "threshold": 0.5}
        }
        
        # Technical term patterns
        self.technical_patterns = [
            r"(?i)\b(algorithm|implementation|optimization|architecture)\b",
            r"(?i)\b(database|framework|library|api|sdk)\b",
            r"(?i)\b(machine learning|artificial intelligence|neural network)\b",
            r"(?i)\b(microservice|container|kubernetes|docker)\b",
            r"(?i)\b(blockchain|cryptocurrency|distributed system)\b"
        ]
        self.compiled_technical = [re.compile(p) for p in self.technical_patterns]
    
    async def observe(self, event: RequestEvent) -> List[ExpertActivationEvent]:
        """Analyze complexity and activate appropriate experts."""
        if not self.should_activate(event):
            return []
        
        complexity_score = self._calculate_complexity_score(event)
        
        if complexity_score < 0.6:  # Low complexity threshold
            return []
        
        activations = []
        
        # Determine which experts to activate based on complexity
        if complexity_score > 0.8:  # High complexity
            # Activate CSA for complex problem analysis
            activations.append(create_expert_activation_event(
                expert_name="CSA",
                reason=f"High complexity detected (score: {complexity_score:.2f})",
                confidence=complexity_score,
                priority=1,
                context={
                    "complexity_score": complexity_score,
                    "task_hint": "analysis",
                    "flags": {"high_complexity": True}
                }
            ))
        
        elif complexity_score > 0.6:  # Medium complexity
            # Check for specific domain indicators
            domain_activations = self._analyze_domain_complexity(event, complexity_score)
            activations.extend(domain_activations)
        
        return activations
    
    def should_activate(self, event: RequestEvent) -> bool:
        """Activate for REQUEST_RECEIVED and COMPLEXITY_DETECTED events."""
        return event.event_type in {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.COMPLEXITY_DETECTED
        }
    
    def _calculate_complexity_score(self, event: RequestEvent) -> float:
        """Calculate complexity score for the request."""
        if event.complexity_score is not None:
            return event.complexity_score
        
        request_text = event.request_text or ""
        score = 0.0
        
        # Length-based complexity
        length_score = min(len(request_text) / 500, 1.0)  # Normalize to 0-1
        score += length_score * self.complexity_indicators["length"]["weight"]
        
        # Technical terms complexity
        tech_matches = sum(1 for pattern in self.compiled_technical 
                          if pattern.search(request_text))
        tech_score = min(tech_matches / 5, 1.0)  # Normalize to 0-1
        score += tech_score * self.complexity_indicators["technical_terms"]["weight"]
        
        # Question complexity (multiple questions, nested queries)
        question_marks = request_text.count('?')
        question_words = len(re.findall(r'(?i)\b(qué|cómo|cuándo|dónde|por qué|what|how|when|where|why)\b', request_text))
        question_score = min((question_marks + question_words) / 5, 1.0)
        score += question_score * self.complexity_indicators["question_complexity"]["weight"]
        
        # Domain specificity (presence of specialized vocabulary)
        domain_score = self._calculate_domain_specificity(request_text)
        score += domain_score * self.complexity_indicators["domain_specificity"]["weight"]
        
        return min(score, 1.0)
    
    def _calculate_domain_specificity(self, text: str) -> float:
        """Calculate how domain-specific the text is."""
        domain_patterns = {
            "medical": r"(?i)\b(diagnosis|treatment|patient|symptom|disease|medical)\b",
            "legal": r"(?i)\b(contract|legal|law|regulation|compliance|court)\b",
            "financial": r"(?i)\b(investment|portfolio|risk|return|financial|trading)\b",
            "technical": r"(?i)\b(system|network|server|database|security|software)\b"
        }
        
        max_specificity = 0.0
        for domain, pattern in domain_patterns.items():
            matches = len(re.findall(pattern, text))
            specificity = min(matches / 3, 1.0)  # Normalize
            max_specificity = max(max_specificity, specificity)
        
        return max_specificity
    
    def _analyze_domain_complexity(self, event: RequestEvent, complexity_score: float) -> List[ExpertActivationEvent]:
        """Analyze domain-specific complexity and suggest experts."""
        activations = []
        request_text = event.request_text or ""
        
        # Check for mathematical content
        math_indicators = len(re.findall(r'\b\d+[\+\-\*/\^=]\d+\b|(?i)\b(equation|formula|calculate)\b', request_text))
        if math_indicators > 0:
            activations.append(create_expert_activation_event(
                expert_name="MathExpert",
                reason=f"Mathematical complexity detected ({math_indicators} indicators)",
                confidence=min(complexity_score + 0.1, 1.0),
                priority=2,
                context={"task_hint": "calculation", "math_indicators": math_indicators}
            ))
        
        # Check for code-related content
        code_indicators = len(re.findall(r'(?i)\b(function|class|method|variable|algorithm)\b|```', request_text))
        if code_indicators > 0:
            activations.append(create_expert_activation_event(
                expert_name="CodeExpert",
                reason=f"Programming complexity detected ({code_indicators} indicators)",
                confidence=min(complexity_score + 0.1, 1.0),
                priority=2,
                context={"task_hint": "programming", "code_indicators": code_indicators}
            ))
        
        return activations
    
    def get_supported_event_types(self) -> Set[RequestEventType]:
        """Support multiple event types."""
        return {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.COMPLEXITY_DETECTED,
            RequestEventType.REQUEST_ANALYZED
        }


class DomainSpecificObserver(RequestObserver):
    """
    Observer that specializes in detecting domain-specific requests.
    """
    
    def __init__(self, name: str = "DomainObserver", priority: int = 2):
        super().__init__(name, priority)
        
        # Domain-specific expert mappings
        self.domain_experts = {
            "mathematics": {
                "expert": "MathExpert",
                "keywords": ["math", "calculate", "equation", "formula", "algebra", "geometry", "calculus"],
                "patterns": [
                    r"\b\d+[\+\-\*/\^=]\d+\b",
                    r"(?i)\b(solve|equation|derivative|integral|limit)\b"
                ]
            },
            "programming": {
                "expert": "CodeExpert", 
                "keywords": ["code", "program", "algorithm", "function", "debug", "software"],
                "patterns": [
                    r"(?i)\b(python|java|javascript|c\+\+|rust|go|php)\b",
                    r"```[\s\S]*?```",
                    r"(?i)\b(class|method|variable|array|loop)\b"
                ]
            },
            "analysis": {
                "expert": "CSA",
                "keywords": ["analyze", "problem", "issue", "diagnosis", "troubleshoot"],
                "patterns": [
                    r"(?i)\b(what if|scenario|hypothesis|cause|effect)\b",
                    r"(?i)\b(problem|issue|error|failure|malfunction)\b"
                ]
            },
            "language": {
                "expert": "SpanishExpert",
                "keywords": ["translate", "spanish", "language", "idioma", "traducir"],
                "patterns": [
                    r"(?i)\b(translate|translation|español|spanish|idioma)\b",
                    r"\b(el|la|los|las)\s+\w+\s+(es|son|está|están)\b"
                ]
            }
        }
        
        # Compile patterns for efficiency
        for domain_info in self.domain_experts.values():
            domain_info["compiled_patterns"] = [
                re.compile(pattern) for pattern in domain_info["patterns"]
            ]
    
    async def observe(self, event: RequestEvent) -> List[ExpertActivationEvent]:
        """Detect domain-specific requests and activate appropriate experts."""
        if not self.should_activate(event):
            return []
        
        activations = []
        request_text = event.request_text or ""
        
        # Analyze each domain
        for domain_name, domain_info in self.domain_experts.items():
            domain_score = self._calculate_domain_score(request_text, domain_info)
            
            if domain_score > 0.4:  # Domain relevance threshold
                activation = create_expert_activation_event(
                    expert_name=domain_info["expert"],
                    reason=f"Domain-specific request detected: {domain_name} (score: {domain_score:.2f})",
                    confidence=domain_score,
                    priority=1 if domain_score > 0.7 else 2,
                    context={
                        "domain": domain_name,
                        "domain_score": domain_score,
                        "task_hint": self._get_task_hint_for_domain(domain_name)
                    }
                )
                activations.append(activation)
        
        return activations
    
    def should_activate(self, event: RequestEvent) -> bool:
        """Activate for domain identification events."""
        return event.event_type in {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.DOMAIN_IDENTIFIED,
            RequestEventType.REQUEST_ANALYZED
        }
    
    def _calculate_domain_score(self, text: str, domain_info: Dict[str, Any]) -> float:
        """Calculate relevance score for a specific domain."""
        score = 0.0
        text_lower = text.lower()
        
        # Keyword matching
        keyword_matches = sum(1 for keyword in domain_info["keywords"] 
                            if keyword in text_lower)
        keyword_score = min(keyword_matches / len(domain_info["keywords"]), 1.0)
        score += keyword_score * 0.6
        
        # Pattern matching
        pattern_matches = sum(1 for pattern in domain_info["compiled_patterns"]
                            if pattern.search(text))
        pattern_score = min(pattern_matches / len(domain_info["compiled_patterns"]), 1.0)
        score += pattern_score * 0.4
        
        return score
    
    def _get_task_hint_for_domain(self, domain: str) -> str:
        """Get appropriate task hint for domain."""
        task_hints = {
            "mathematics": "calculation",
            "programming": "programming", 
            "analysis": "diagnosis",
            "language": "translation"
        }
        return task_hints.get(domain, "analysis")
    
    def get_supported_event_types(self) -> Set[RequestEventType]:
        """Support domain-related events."""
        return {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.DOMAIN_IDENTIFIED,
            RequestEventType.REQUEST_ANALYZED
        }


class PerformanceObserver(RequestObserver):
    """
    Observer that monitors system performance and activates experts based on load.
    """
    
    def __init__(self, name: str = "PerformanceObserver", priority: int = 3):
        super().__init__(name, priority)
        
        self.performance_history = deque(maxlen=100)
        self.load_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        }
        
        # Track expert performance
        self.expert_performance = defaultdict(lambda: {
            "response_times": deque(maxlen=50),
            "success_rate": 0.0,
            "last_used": None
        })
    
    async def observe(self, event: RequestEvent) -> List[ExpertActivationEvent]:
        """Monitor performance and make load-based activation decisions."""
        if not self.should_activate(event):
            return []
        
        current_load = self._estimate_system_load()
        self.performance_history.append({
            "timestamp": time.time(),
            "load": current_load,
            "event_type": event.event_type
        })
        
        activations = []
        
        # Performance-based activation logic
        if current_load < self.load_thresholds["low"]:
            # Low load - can activate multiple experts
            if event.request_text and len(event.request_text) > 100:
                activations.append(create_expert_activation_event(
                    expert_name="CSA",
                    reason="Low system load allows comprehensive analysis",
                    confidence=0.7,
                    priority=2,
                    context={
                        "system_load": current_load,
                        "task_hint": "analysis",
                        "flags": {"comprehensive_analysis": True}
                    }
                ))
        
        elif current_load > self.load_thresholds["high"]:
            # High load - only activate high-priority experts
            if event.complexity_score and event.complexity_score > 0.8:
                activations.append(create_expert_activation_event(
                    expert_name="CSA",
                    reason="High complexity requires expert despite high load",
                    confidence=0.6,
                    priority=1,
                    context={
                        "system_load": current_load,
                        "task_hint": "diagnosis",
                        "flags": {"high_priority": True}
                    }
                ))
        
        return activations
    
    def should_activate(self, event: RequestEvent) -> bool:
        """Activate for performance-related events."""
        return event.event_type in {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.PERFORMANCE_THRESHOLD,
            RequestEventType.PROCESSING_COMPLETE
        }
    
    def _estimate_system_load(self) -> float:
        """Estimate current system load (simplified)."""
        if len(self.performance_history) < 5:
            return 0.3  # Default moderate load
        
        # Calculate load based on recent activity
        recent_events = list(self.performance_history)[-10:]
        time_window = 60.0  # 1 minute window
        current_time = time.time()
        
        recent_activity = [
            event for event in recent_events
            if current_time - event["timestamp"] < time_window
        ]
        
        # Simple load estimation based on event frequency
        load_estimate = min(len(recent_activity) / 20.0, 1.0)
        return load_estimate
    
    def update_expert_performance(self, expert_name: str, response_time: float, success: bool):
        """Update performance metrics for an expert."""
        perf = self.expert_performance[expert_name]
        perf["response_times"].append(response_time)
        perf["last_used"] = time.time()
        
        # Update success rate (simple moving average)
        current_rate = perf["success_rate"]
        perf["success_rate"] = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        current_load = self._estimate_system_load()
        
        expert_stats = {}
        for expert_name, perf in self.expert_performance.items():
            avg_response_time = (
                sum(perf["response_times"]) / len(perf["response_times"])
                if perf["response_times"] else 0.0
            )
            expert_stats[expert_name] = {
                "average_response_time": avg_response_time,
                "success_rate": perf["success_rate"],
                "last_used": perf["last_used"],
                "total_requests": len(perf["response_times"])
            }
        
        return {
            "current_system_load": current_load,
            "load_classification": self._classify_load(current_load),
            "expert_performance": expert_stats,
            "history_size": len(self.performance_history)
        }
    
    def _classify_load(self, load: float) -> str:
        """Classify system load level."""
        if load < self.load_thresholds["low"]:
            return "low"
        elif load < self.load_thresholds["medium"]:
            return "medium"
        elif load < self.load_thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    def get_supported_event_types(self) -> Set[RequestEventType]:
        """Support performance-related events."""
        return {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.PERFORMANCE_THRESHOLD,
            RequestEventType.PROCESSING_COMPLETE
        }


class AdaptiveObserver(RequestObserver):
    """
    Observer that learns from activation patterns and adapts its behavior.
    """
    
    def __init__(self, name: str = "AdaptiveObserver", priority: int = 1):
        super().__init__(name, priority)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.activation_weights = defaultdict(float)
        self.success_history = deque(maxlen=200)
        self.pattern_effectiveness = defaultdict(lambda: {"successes": 0, "failures": 0})
        
        # Initialize with some baseline weights
        self.activation_weights.update({
            "high_complexity": 0.8,
            "domain_specific": 0.7,
            "pattern_match": 0.6,
            "performance_based": 0.5
        })
    
    async def observe(self, event: RequestEvent) -> List[ExpertActivationEvent]:
        """Make adaptive activation decisions based on learned patterns."""
        if not self.should_activate(event):
            return []
        
        activations = []
        
        # Analyze request using learned patterns
        activation_signals = self._analyze_activation_signals(event)
        
        for expert_name, signal_data in activation_signals.items():
            # Calculate adaptive confidence
            base_confidence = signal_data["base_confidence"]
            learned_weight = self.activation_weights.get(f"expert_{expert_name}", 0.5)
            pattern_weight = self._get_pattern_effectiveness(signal_data["patterns"])
            
            adaptive_confidence = base_confidence * learned_weight * pattern_weight
            
            if adaptive_confidence > 0.5:  # Adaptive threshold
                activation = create_expert_activation_event(
                    expert_name=expert_name,
                    reason=f"Adaptive activation based on learned patterns (confidence: {adaptive_confidence:.2f})",
                    confidence=adaptive_confidence,
                    priority=1 if adaptive_confidence > 0.7 else 2,
                    patterns=signal_data["patterns"],
                    context={
                        "adaptive_confidence": adaptive_confidence,
                        "learned_weight": learned_weight,
                        "pattern_weight": pattern_weight,
                        "task_hint": signal_data.get("task_hint", "analysis")
                    }
                )
                activations.append(activation)
        
        return activations
    
    def should_activate(self, event: RequestEvent) -> bool:
        """Activate for most event types to learn from them."""
        return event.event_type in {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.REQUEST_ANALYZED,
            RequestEventType.COMPLEXITY_DETECTED,
            RequestEventType.DOMAIN_IDENTIFIED
        }
    
    def _analyze_activation_signals(self, event: RequestEvent) -> Dict[str, Dict[str, Any]]:
        """Analyze request for activation signals."""
        signals = {}
        request_text = event.request_text or ""
        
        # CSA activation signals
        csa_patterns = []
        csa_confidence = 0.0
        
        if event.complexity_score and event.complexity_score > 0.6:
            csa_patterns.append("high_complexity")
            csa_confidence += 0.3
        
        if any(keyword in request_text.lower() for keyword in 
               ["problem", "issue", "error", "what if", "scenario"]):
            csa_patterns.append("problem_analysis")
            csa_confidence += 0.4
        
        if csa_patterns:
            signals["CSA"] = {
                "base_confidence": csa_confidence,
                "patterns": csa_patterns,
                "task_hint": "diagnosis"
            }
        
        # Math expert signals
        math_patterns = []
        math_confidence = 0.0
        
        if re.search(r'\b\d+[\+\-\*/\^=]\d+\b', request_text):
            math_patterns.append("mathematical_expression")
            math_confidence += 0.5
        
        if any(keyword in request_text.lower() for keyword in 
               ["calculate", "solve", "equation", "formula"]):
            math_patterns.append("mathematical_keywords")
            math_confidence += 0.3
        
        if math_patterns:
            signals["MathExpert"] = {
                "base_confidence": math_confidence,
                "patterns": math_patterns,
                "task_hint": "calculation"
            }
        
        return signals
    
    def _get_pattern_effectiveness(self, patterns: List[str]) -> float:
        """Get effectiveness weight for patterns based on history."""
        if not patterns:
            return 0.5
        
        total_weight = 0.0
        for pattern in patterns:
            effectiveness = self.pattern_effectiveness[pattern]
            total_successes = effectiveness["successes"]
            total_attempts = total_successes + effectiveness["failures"]
            
            if total_attempts > 0:
                success_rate = total_successes / total_attempts
                total_weight += success_rate
            else:
                total_weight += 0.5  # Default weight for new patterns
        
        return total_weight / len(patterns)
    
    def learn_from_feedback(self, expert_name: str, patterns: List[str], success: bool):
        """Learn from activation feedback to improve future decisions."""
        # Update expert-specific weights
        expert_key = f"expert_{expert_name}"
        if success:
            self.activation_weights[expert_key] += self.learning_rate
        else:
            self.activation_weights[expert_key] -= self.learning_rate
        
        # Keep weights in reasonable bounds
        self.activation_weights[expert_key] = max(0.1, min(2.0, self.activation_weights[expert_key]))
        
        # Update pattern effectiveness
        for pattern in patterns:
            if success:
                self.pattern_effectiveness[pattern]["successes"] += 1
            else:
                self.pattern_effectiveness[pattern]["failures"] += 1
        
        # Store in success history
        self.success_history.append({
            "expert": expert_name,
            "patterns": patterns,
            "success": success,
            "timestamp": time.time()
        })
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning process."""
        # Calculate overall success rate
        if self.success_history:
            overall_success_rate = sum(1 for entry in self.success_history if entry["success"]) / len(self.success_history)
        else:
            overall_success_rate = 0.0
        
        # Get top performing patterns
        pattern_performance = {}
        for pattern, stats in self.pattern_effectiveness.items():
            total = stats["successes"] + stats["failures"]
            if total > 0:
                pattern_performance[pattern] = stats["successes"] / total
        
        top_patterns = sorted(pattern_performance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "overall_success_rate": overall_success_rate,
            "total_learning_samples": len(self.success_history),
            "expert_weights": dict(self.activation_weights),
            "top_performing_patterns": top_patterns,
            "learning_rate": self.learning_rate,
            "pattern_count": len(self.pattern_effectiveness)
        }
    
    def get_supported_event_types(self) -> Set[RequestEventType]:
        """Support multiple event types for comprehensive learning."""
        return {
            RequestEventType.REQUEST_RECEIVED,
            RequestEventType.REQUEST_ANALYZED,
            RequestEventType.COMPLEXITY_DETECTED,
            RequestEventType.DOMAIN_IDENTIFIED,
            RequestEventType.PATTERN_MATCHED
        }