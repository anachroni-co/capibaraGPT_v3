"""
Advanced Agent Behaviors - Extended Strategy Pattern - CapibaraGPT v3024
========================================================================

Additional advanced behaviors for the Strategy pattern:
- ResearchBehavior: Advanced research behavior
- CodingBehavior: Specialized programming behavior
- CommunicationBehavior: Inter-agent communication behavior
- MonitoringBehavior: Monitoring and metrics behavior
- LearningBehavior: Adaptive learning behavior
"""

import time
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from pathlib import Path

# Safe imports
try:
    from ..interfaces.iagent import (
        IAgentBehavior, AgentBehaviorType, AgentCapability,
        AgentContext, AgentResult, IAgent
    )
    from .behaviors import BaseBehavior
except ImportError:
    # Fallback imports
    from abc import ABC, abstractmethod
    from enum import Enum
    from dataclasses import dataclass
    
    class AgentBehaviorType(str, Enum):
        RESEARCH = "research"
        CODING = "coding"
        COMMUNICATION = "communication"
        MONITORING = "monitoring"
        LEARNING = "learning"
    
    class AgentCapability(str, Enum):
        INFORMATION_GATHERING = "information_gathering"
        CODE_GENERATION = "code_generation"
        INTER_AGENT_COMMUNICATION = "inter_agent_communication"
        PERFORMANCE_MONITORING = "performance_monitoring"
        ADAPTIVE_LEARNING = "adaptive_learning"
    
    @dataclass
    class AgentContext:
        task_id: str
        task_description: str
        requirements: Dict[str, Any]
    
    @dataclass
    class AgentResult:
        agent_id: str
        status: str
        result: Any
        execution_time_ms: float
        confidence: float = 0.0
    
    class BaseBehavior:
        def __init__(self, config=None):
            self.config = config or {}
            self.execution_count = 0
            self.success_count = 0
        
        def _update_metrics(self, time, success):
            self.execution_count += 1
            if success:
                self.success_count += 1

logger = logging.getLogger(__name__)


# ============================================================================
# Research Behavior - Strategy for Information Gathering
# ============================================================================

class ResearchBehavior(BaseBehavior):
    """Specialized behavior for research and information gathering."""
    
    @property
    def behavior_type(self) -> AgentBehaviorType:
        return AgentBehaviorType.RESEARCH
    
    @property
    def required_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.INFORMATION_GATHERING,
            AgentCapability.SOURCE_VALIDATION,
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.SYNTHESIS_GENERATION
        ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_sources = self.config.get("max_sources", 10)
        self.quality_threshold = self.config.get("quality_threshold", 0.7)
        self.use_data_integration = self.config.get("use_data_integration", True)
        self.research_cache = {}
        self.source_quality_scores = {}
    
    def execute_behavior(self, context: AgentContext, agent: IAgent) -> AgentResult:
        """Execute specialized research."""
        start_time = time.time()
        
        try:
            research_result = self._conduct_research(
                context.task_description,
                context.requirements
            )
            
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, research_result["success"])
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="success" if research_result["success"] else "failed",
                result=research_result,
                execution_time_ms=execution_time,
                confidence=research_result.get("confidence", 0.8),
                metadata={
                    "sources_consulted": research_result.get("sources_count", 0),
                    "quality_score": research_result.get("quality_score", 0.0)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, False)
            logger.error(f"Research behavior failed: {e}")
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="failed",
                result=None,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def _conduct_research(self, query: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct complete research."""

        # Phase 1: Research planning
        research_plan = self._plan_research(query, requirements)

        # Phase 2: Information gathering
        gathered_data = self._gather_information(research_plan)

        # Phase 3: Source validation
        validated_sources = self._validate_sources(gathered_data)

        # Phase 4: Analysis and synthesis
        analysis_result = self._analyze_and_synthesize(validated_sources)
        
        return {
            "success": True,
            "research_plan": research_plan,
            "sources_count": len(validated_sources),
            "quality_score": analysis_result.get("quality_score", 0.8),
            "findings": analysis_result.get("findings", []),
            "synthesis": analysis_result.get("synthesis", ""),
            "confidence": analysis_result.get("confidence", 0.8),
            "recommendations": analysis_result.get("recommendations", [])
        }
    
    def _plan_research(self, query: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Plan research strategy."""

        # Identify keywords and concepts
        keywords = self._extract_keywords(query)
        concepts = self._identify_concepts(query)
        
        # Determinar tipos de fuentes necesarias
        source_types = self._determine_source_types(query, requirements)
        
        # Establish quality criteria
        quality_criteria = self._define_quality_criteria(requirements)
        
        return {
            "keywords": keywords,
            "concepts": concepts,
            "source_types": source_types,
            "quality_criteria": quality_criteria,
            "search_strategy": self._create_search_strategy(keywords, concepts)
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords."""
        # Simplified implementation - in production would use advanced NLP
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
        words = [w.lower().strip(".,!?") for w in query.split()]
        return [w for w in words if len(w) > 3 and w not in stop_words]
    
    def _identify_concepts(self, query: str) -> List[str]:
        """Identify main concepts."""
        concepts = []

        # Technical concepts
        if any(term in query.lower() for term in ["algorithm", "model", "system", "architecture"]):
            concepts.append("technical")

        # Business concepts
        if any(term in query.lower() for term in ["business", "market", "strategy", "customer"]):
            concepts.append("business")

        # Scientific concepts
        if any(term in query.lower() for term in ["research", "study", "analysis", "data"]):
            concepts.append("scientific")
        
        return concepts if concepts else ["general"]
    
    def _determine_source_types(self, query: str, requirements: Dict[str, Any]) -> List[str]:
        """Determine required source types."""
        source_types = ["web", "documentation"]
        
        if requirements.get("academic_sources", False):
            source_types.append("academic")
        
        if "code" in query.lower() or "implementation" in query.lower():
            source_types.extend(["code_repositories", "technical_docs"])
        
        if requirements.get("real_time", False):
            source_types.append("news")
        
        return source_types
    
    def _define_quality_criteria(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Define quality criteria for sources."""
        return {
            "min_quality_score": requirements.get("min_quality", self.quality_threshold),
            "require_recent": requirements.get("require_recent", False),
            "prefer_authoritative": requirements.get("prefer_authoritative", True),
            "max_age_days": requirements.get("max_age_days", 365)
        }
    
    def _create_search_strategy(self, keywords: List[str], concepts: List[str]) -> Dict[str, Any]:
        """Create search strategy."""
        return {
            "primary_queries": [" ".join(keywords[:3])],
            "secondary_queries": [" ".join([keywords[0], concept]) for concept in concepts],
            "search_depth": "comprehensive" if len(keywords) > 5 else "standard",
            "parallel_search": len(keywords) > 3
        }
    
    def _gather_information(self, research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather information from multiple sources."""
        gathered_data = []

        # Gather from configured source types and keep deterministic limits.
        for source_type in research_plan["source_types"]:
            source_data = self._gather_from_source_type(source_type, research_plan)
            gathered_data.extend(source_data)

        return gathered_data[:self.max_sources]  # Limit number of sources
    
    def _gather_from_source_type(self, source_type: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather information from a specific source type."""
        if source_type == "web":
            return self._gather_web_sources(plan)
        if source_type == "documentation":
            return self._gather_local_documentation_sources(plan)
        if source_type == "academic":
            return self._gather_academic_sources(plan)

        return []

    def _gather_web_sources(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather web sources from Wikipedia search API."""
        query = " ".join(plan.get("keywords", [])[:4]).strip()
        if not query:
            return []

        encoded = urllib.parse.quote(query)
        api_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}&utf8=&format=json&srlimit=3"
        )

        try:
            with urllib.request.urlopen(api_url, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Web gathering failed for query '%s': %s", query, exc)
            return []

        results = []
        for item in payload.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if not title:
                continue
            results.append(
                {
                    "type": "web",
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                    "title": title,
                    "content": snippet,
                    "quality_indicators": {"authority": 0.8, "relevance": 0.8, "freshness": 0.6},
                }
            )
        return results

    def _gather_local_documentation_sources(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather documentation sources from local repository markdown/text files."""
        workspace_root = Path(os.getcwd())
        keywords = [k.lower() for k in plan.get("keywords", [])[:6]]
        if not keywords:
            return []

        candidates = []
        for file_path in workspace_root.rglob("*"):
            if file_path.suffix.lower() not in {".md", ".txt", ".rst"}:
                continue
            if any(part.startswith(".") for part in file_path.parts):
                continue
            candidates.append(file_path)
            if len(candidates) >= 200:
                break

        results = []
        for path in candidates:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            low = content.lower()
            match_count = sum(1 for kw in keywords if kw in low)
            if match_count == 0:
                continue

            rel_path = str(path.relative_to(workspace_root))
            excerpt = content[:500]
            relevance = min(1.0, 0.3 + (match_count / max(1, len(keywords))))
            results.append(
                {
                    "type": "documentation",
                    "source": rel_path,
                    "content": excerpt,
                    "quality_indicators": {"authority": 0.75, "relevance": relevance, "freshness": 0.7},
                }
            )

        results.sort(key=lambda s: s["quality_indicators"]["relevance"], reverse=True)
        return results[:5]

    def _gather_academic_sources(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather academic references from Crossref API."""
        query = " ".join(plan.get("keywords", [])[:4]).strip()
        if not query:
            return []

        encoded = urllib.parse.quote(query)
        api_url = f"https://api.crossref.org/works?query={encoded}&rows=2"
        try:
            with urllib.request.urlopen(api_url, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Academic gathering failed for query '%s': %s", query, exc)
            return []

        results = []
        items = payload.get("message", {}).get("items", [])
        for item in items:
            title_list = item.get("title", [])
            title = title_list[0] if title_list else "Untitled"
            authors = []
            for a in item.get("author", [])[:3]:
                authors.append(f"{a.get('given', '').strip()} {a.get('family', '').strip()}".strip())
            results.append(
                {
                    "type": "academic",
                    "title": title,
                    "authors": [a for a in authors if a],
                    "content": item.get("abstract", "")[:500],
                    "quality_indicators": {"authority": 0.9, "relevance": 0.8, "freshness": 0.5},
                }
            )
        return results

    def _gather_from_source_type_legacy(self, source_type: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Legacy simulated source gathering retained for compatibility reference."""
        sources = []

        if source_type == "web":
            sources.extend([
                {
                    "type": "web",
                    "url": f"https://example.com/article_{i}",
                    "title": f"Web Article {i}",
                    "content": f"Content related to {', '.join(plan['keywords'])}",
                    "quality_indicators": {"authority": 0.7, "relevance": 0.8, "freshness": 0.9}
                }
                for i in range(3)
            ])
        
        elif source_type == "documentation":
            sources.extend([
                {
                    "type": "documentation",
                    "source": f"Technical Documentation {i}",
                    "content": f"Technical details about {', '.join(plan['concepts'])}",
                    "quality_indicators": {"authority": 0.9, "relevance": 0.9, "freshness": 0.6}
                }
                for i in range(2)
            ])
        
        elif source_type == "academic":
            sources.extend([
                {
                    "type": "academic",
                    "title": f"Academic Paper on {plan['keywords'][0]}",
                    "authors": ["Dr. Smith", "Dr. Johnson"],
                    "content": f"Academic research on {', '.join(plan['keywords'])}",
                    "quality_indicators": {"authority": 0.95, "relevance": 0.85, "freshness": 0.4}
                }
            ])
        
        return sources
    
    def _validate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate quality and relevance of sources."""
        validated_sources = []
        
        for source in sources:
            quality_score = self._calculate_source_quality(source)
            
            if quality_score >= self.quality_threshold:
                source["quality_score"] = quality_score
                validated_sources.append(source)
                
                # Cache quality score
                source_id = source.get("url", source.get("title", "unknown"))
                self.source_quality_scores[source_id] = quality_score

        # Sort by quality
        validated_sources.sort(key=lambda x: x["quality_score"], reverse=True)
        
        return validated_sources
    
    def _calculate_source_quality(self, source: Dict[str, Any]) -> float:
        """Calculate quality score for a source."""
        indicators = source.get("quality_indicators", {})

        # Weights for different quality factors
        weights = {
            "authority": 0.4,
            "relevance": 0.4,
            "freshness": 0.2
        }
        
        quality_score = sum(
            indicators.get(factor, 0.5) * weight
            for factor, weight in weights.items()
        )

        # Bonus by source type
        source_type = source.get("type", "unknown")
        if source_type == "academic":
            quality_score += 0.1
        elif source_type == "documentation":
            quality_score += 0.05
        
        return min(1.0, quality_score)
    
    def _analyze_and_synthesize(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sources and synthesize findings."""

        # Extract key findings
        findings = self._extract_key_findings(sources)

        # Identify patterns and themes
        patterns = self._identify_patterns(findings)

        # Generate synthesis
        synthesis = self._generate_synthesis(findings, patterns)

        # Calculate confidence
        confidence = self._calculate_confidence(sources, findings)

        # Generate recommendations
        recommendations = self._generate_recommendations(synthesis, patterns)
        
        return {
            "findings": findings,
            "patterns": patterns,
            "synthesis": synthesis,
            "confidence": confidence,
            "quality_score": sum(s["quality_score"] for s in sources) / len(sources),
            "recommendations": recommendations
        }
    
    def _extract_key_findings(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key findings from sources."""
        findings = []

        for source in sources:
            content = source.get("content", "")

            # Simulation of findings extraction
            finding = {
                "source": source.get("title", source.get("url", "Unknown")),
                "key_points": self._extract_key_points(content),
                "importance": source.get("quality_score", 0.5),
                "type": source.get("type", "general")
            }
            findings.append(finding)
        
        return findings
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Simplified implementation
        sentences = content.split(". ")
        return [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
    
    def _identify_patterns(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns in findings."""
        patterns = []

        # Theme frequency analysis
        theme_frequency = defaultdict(int)
        for finding in findings:
            for point in finding.get("key_points", []):
                # Simulation of theme identification
                if "performance" in point.lower():
                    theme_frequency["performance"] += 1
                if "security" in point.lower():
                    theme_frequency["security"] += 1
                if "scalability" in point.lower():
                    theme_frequency["scalability"] += 1
        
        # Convertir a patrones
        for theme, frequency in theme_frequency.items():
            if frequency >= 2:
                patterns.append({
                    "pattern_type": "theme_frequency",
                    "theme": theme,
                    "frequency": frequency,
                    "confidence": min(1.0, frequency / len(findings))
                })
        
        return patterns
    
    def _generate_synthesis(self, findings: List[Dict[str, Any]], patterns: List[Dict[str, Any]]) -> str:
        """Generate research synthesis."""

        synthesis_parts = []

        # Summary of main findings
        high_importance_findings = [f for f in findings if f.get("importance", 0) > 0.7]
        if high_importance_findings:
            synthesis_parts.append(
                f"Based on {len(high_importance_findings)} high-quality sources, "
                f"the research reveals several key insights."
            )

        # Identified patterns
        if patterns:
            main_themes = [p["theme"] for p in patterns if p.get("confidence", 0) > 0.5]
            if main_themes:
                synthesis_parts.append(
                    f"The main themes that emerge are: {', '.join(main_themes)}."
                )

        # Conclusion
        synthesis_parts.append(
            f"The research provides comprehensive coverage with "
            f"{len(findings)} sources analyzed."
        )
        
        return " ".join(synthesis_parts)
    
    def _calculate_confidence(self, sources: List[Dict[str, Any]], findings: List[Dict[str, Any]]) -> float:
        """Calculate research confidence level."""

        # Confidence factors
        source_quality = sum(s.get("quality_score", 0.5) for s in sources) / len(sources)
        source_diversity = len(set(s.get("type", "unknown") for s in sources)) / 4  # Max 4 types
        findings_consistency = len(findings) / max(1, len(sources))  # Findings per source
        
        confidence = (source_quality * 0.5 + source_diversity * 0.3 + 
                     min(1.0, findings_consistency) * 0.2)
        
        return min(1.0, confidence)
    
    def _generate_recommendations(self, synthesis: str, patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate research-based recommendations."""
        recommendations = []

        # Recommendations based on patterns
        for pattern in patterns:
            theme = pattern.get("theme", "")
            confidence = pattern.get("confidence", 0.0)
            
            if confidence > 0.7:
                if theme == "performance":
                    recommendations.append("Consider performance optimization as a priority")
                elif theme == "security":
                    recommendations.append("Implement robust security measures")
                elif theme == "scalability":
                    recommendations.append("Plan for scalability from the beginning")

        # General recommendation
        if not recommendations:
            recommendations.append("Further research may be needed for specific implementation details")
        
        return recommendations


# ============================================================================
# Coding Behavior - Strategy for Code Generation and Development
# ============================================================================

class CodingBehavior(BaseBehavior):
    """Specialized behavior for programming and code development."""
    
    @property
    def behavior_type(self) -> AgentBehaviorType:
        return AgentBehaviorType.CODING
    
    @property
    def required_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CODE_GENERATION,
            AgentCapability.CODE_DEBUGGING,
            AgentCapability.CODE_OPTIMIZATION,
            AgentCapability.TESTING_FRAMEWORK
        ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_languages = self.config.get("languages", ["python", "javascript", "typescript"])
        self.code_style = self.config.get("code_style", "clean")
        self.include_tests = self.config.get("include_tests", True)
        self.include_docs = self.config.get("include_docs", True)
        self.code_templates = {}
    
    def execute_behavior(self, context: AgentContext, agent: IAgent) -> AgentResult:
        """Execute specialized code generation."""
        start_time = time.time()
        
        try:
            coding_result = self._generate_code_solution(
                context.task_description,
                context.requirements
            )
            
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, coding_result["success"])
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="success" if coding_result["success"] else "failed",
                result=coding_result,
                execution_time_ms=execution_time,
                confidence=coding_result.get("confidence", 0.85),
                metadata={
                    "language": coding_result.get("language", "python"),
                    "lines_of_code": coding_result.get("lines_of_code", 0),
                    "functions_created": coding_result.get("functions_created", 0)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, False)
            logger.error(f"Coding behavior failed: {e}")
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="failed",
                result=None,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def _generate_code_solution(self, task: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete code solution."""

        # Phase 1: Requirements analysis
        code_requirements = self._analyze_coding_requirements(task, requirements)

        # Phase 2: Architecture design
        architecture = self._design_code_architecture(code_requirements)

        # Phase 3: Code generation
        generated_code = self._generate_code(architecture, code_requirements)

        # Phase 4: Test generation
        tests = []
        if self.include_tests:
            tests = self._generate_tests(generated_code, code_requirements)

        # Phase 5: Documentation
        documentation = ""
        if self.include_docs:
            documentation = self._generate_documentation(generated_code, architecture)
        
        return {
            "success": True,
            "language": code_requirements.get("language", "python"),
            "architecture": architecture,
            "code": generated_code,
            "tests": tests,
            "documentation": documentation,
            "lines_of_code": len(generated_code.get("main_code", "").split("\n")),
            "functions_created": len(generated_code.get("functions", [])),
            "confidence": 0.85,
            "quality_metrics": self._calculate_code_quality(generated_code)
        }
    
    def _analyze_coding_requirements(self, task: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific programming requirements."""

        # Determine programming language
        language = self._determine_language(task, requirements)

        # Identify application type
        app_type = self._identify_application_type(task)

        # Extract required features
        features = self._extract_required_features(task)

        # Determine required design patterns
        design_patterns = self._suggest_design_patterns(task, features)
        
        return {
            "language": language,
            "app_type": app_type,
            "features": features,
            "design_patterns": design_patterns,
            "performance_requirements": requirements.get("performance", "standard"),
            "scalability_requirements": requirements.get("scalability", "medium")
        }
    
    def _determine_language(self, task: str, requirements: Dict[str, Any]) -> str:
        """Determine the best programming language for the task."""

        # Explicit preference
        if "language" in requirements:
            requested_lang = requirements["language"].lower()
            if requested_lang in self.supported_languages:
                return requested_lang
        
        # Context-based analysis
        task_lower = task.lower()
        
        if any(term in task_lower for term in ["web", "frontend", "react", "vue"]):
            return "javascript" if "javascript" in self.supported_languages else "typescript"
        elif any(term in task_lower for term in ["api", "backend", "server", "data"]):
            return "python"
        elif "mobile" in task_lower:
            return "typescript"  # For React Native
        else:
            return self.supported_languages[0]  # Default language
    
    def _identify_application_type(self, task: str) -> str:
        """Identify application type to develop."""
        task_lower = task.lower()
        
        if any(term in task_lower for term in ["api", "rest", "graphql", "backend"]):
            return "api"
        elif any(term in task_lower for term in ["web", "frontend", "ui", "interface"]):
            return "web_app"
        elif any(term in task_lower for term in ["script", "automation", "tool"]):
            return "script"
        elif any(term in task_lower for term in ["library", "package", "module"]):
            return "library"
        else:
            return "general"
    
    def _extract_required_features(self, task: str) -> List[str]:
        """Extract required code features."""
        features = []
        task_lower = task.lower()
        
        feature_keywords = {
            "authentication": ["auth", "login", "user", "password"],
            "database": ["database", "db", "store", "persist"],
            "validation": ["validate", "check", "verify"],
            "logging": ["log", "debug", "monitor"],
            "caching": ["cache", "redis", "memory"],
            "testing": ["test", "spec", "unit"],
            "documentation": ["doc", "readme", "comment"]
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                features.append(feature)
        
        return features if features else ["basic_functionality"]
    
    def _suggest_design_patterns(self, task: str, features: List[str]) -> List[str]:
        """Suggest appropriate design patterns."""
        patterns = []

        # Factory pattern for object creation
        if any(term in task.lower() for term in ["create", "generate", "factory"]):
            patterns.append("factory")

        # Strategy pattern for different behaviors
        if any(term in task.lower() for term in ["strategy", "behavior", "algorithm"]):
            patterns.append("strategy")

        # Observer pattern for events
        if any(term in task.lower() for term in ["event", "notify", "observer"]):
            patterns.append("observer")

        # Singleton for unique resources
        if "database" in features or "config" in task.lower():
            patterns.append("singleton")
        
        return patterns if patterns else ["simple"]
    
    def _design_code_architecture(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design code architecture."""
        
        app_type = requirements.get("app_type", "general")
        language = requirements.get("language", "python")
        features = requirements.get("features", [])
        
        architecture = {
            "structure": self._design_project_structure(app_type, language),
            "components": self._identify_components(features),
            "dependencies": self._identify_dependencies(features, language),
            "patterns": requirements.get("design_patterns", [])
        }
        
        return architecture
    
    def _design_project_structure(self, app_type: str, language: str) -> Dict[str, Any]:
        """Design project structure."""
        
        if language == "python":
            if app_type == "api":
                return {
                    "main_files": ["main.py", "app.py"],
                    "directories": ["models", "routes", "utils", "tests"],
                    "config_files": ["requirements.txt", "config.py"]
                }
            else:
                return {
                    "main_files": ["main.py"],
                    "directories": ["src", "tests", "docs"],
                    "config_files": ["requirements.txt", "setup.py"]
                }
        
        elif language in ["javascript", "typescript"]:
            return {
                "main_files": ["index.js", "app.js"],
                "directories": ["src", "tests", "docs"],
                "config_files": ["package.json", "tsconfig.json" if language == "typescript" else None]
            }
        
        return {
            "main_files": ["main.py"],
            "directories": ["src", "tests"],
            "config_files": []
        }
    
    def _identify_components(self, features: List[str]) -> List[Dict[str, Any]]:
        """Identify required components."""
        components = []
        
        for feature in features:
            if feature == "authentication":
                components.append({
                    "name": "AuthManager",
                    "type": "class",
                    "responsibilities": ["user_authentication", "token_management"]
                })
            elif feature == "database":
                components.append({
                    "name": "DatabaseManager", 
                    "type": "class",
                    "responsibilities": ["data_persistence", "query_execution"]
                })
            elif feature == "validation":
                components.append({
                    "name": "Validator",
                    "type": "class",
                    "responsibilities": ["input_validation", "data_verification"]
                })
        
        # Componente principal siempre presente
        components.append({
            "name": "MainApplication",
            "type": "class",
            "responsibilities": ["application_logic", "component_coordination"]
        })
        
        return components
    
    def _identify_dependencies(self, features: List[str], language: str) -> List[str]:
        """Identify required dependencies."""
        dependencies = []
        
        if language == "python":
            if "database" in features:
                dependencies.extend(["sqlalchemy", "psycopg2"])
            if "api" in str(features):
                dependencies.extend(["fastapi", "uvicorn"])
            if "testing" in features:
                dependencies.append("pytest")
        
        elif language in ["javascript", "typescript"]:
            if "api" in str(features):
                dependencies.extend(["express", "cors"])
            if "testing" in features:
                dependencies.extend(["jest", "supertest"])
            if language == "typescript":
                dependencies.extend(["typescript", "@types/node"])
        
        return dependencies
    
    def _generate_code(self, architecture: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate main code."""
        
        language = requirements.get("language", "python")
        components = architecture.get("components", [])
        
        generated_code = {
            "main_code": self._generate_main_code(language, components),
            "functions": self._generate_functions(components, language),
            "classes": self._generate_classes(components, language),
            "imports": self._generate_imports(architecture.get("dependencies", []), language)
        }
        
        return generated_code
    
    def _generate_main_code(self, language: str, components: List[Dict[str, Any]]) -> str:
        """Generate main code."""
        
        if language == "python":
            main_code = '''#!/usr/bin/env python3
"""
Main application module.
Generated by CapibaraGPT Coding Agent.
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    logger.info("Starting application...")
    
    try:
        # Initialize application components
        app = MainApplication()
        app.initialize()
        
        # Run application
        app.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    logger.info("Application completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
'''
        
        elif language in ["javascript", "typescript"]:
            main_code = '''// Main application module
// Generated by CapibaraGPT Coding Agent

const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Routes
app.get('/', (req, res) => {
    res.json({ message: 'Application is running' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

module.exports = app;
'''
        
        else:
            main_code = "# Generated code placeholder"
        
        return main_code
    
    def _generate_functions(self, components: List[Dict[str, Any]], language: str) -> List[Dict[str, str]]:
        """Generate helper functions."""
        functions = []
        
        if language == "python":
            functions.append({
                "name": "validate_input",
                "code": '''def validate_input(data: Dict[str, Any]) -> bool:
    """Validate input data."""
    if not data:
        return False
    
    # Add specific validation logic here
    return True'''
            })
            
            functions.append({
                "name": "process_data",
                "code": '''def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process input data and return results."""
    if not validate_input(data):
        raise ValueError("Invalid input data")
    
    # Add processing logic here
    result = {"status": "processed", "data": data}
    return result'''
            })
        
        return functions
    
    def _generate_classes(self, components: List[Dict[str, Any]], language: str) -> List[Dict[str, str]]:
        """Generate main classes."""
        classes = []
        
        for component in components:
            if component.get("type") == "class":
                class_code = self._generate_class_code(component, language)
                classes.append({
                    "name": component["name"],
                    "code": class_code
                })
        
        return classes
    
    def _generate_class_code(self, component: Dict[str, Any], language: str) -> str:
        """Generate code for a specific class."""
        
        name = component.get("name", "GeneratedClass")
        responsibilities = component.get("responsibilities", [])
        
        if language == "python":
            methods = []
            for responsibility in responsibilities:
                method_name = responsibility.replace("_", "_")
                methods.append(f'''    def {method_name}(self) -> bool:
        """Handle {responsibility.replace('_', ' ')}."""
        # Implementation needed
        return True''')
            
            class_code = f'''class {name}:
    """Generated class for {name}."""
    
    def __init__(self):
        """Initialize {name}."""
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the component."""
        self.initialized = True
    
{chr(10).join(methods) if methods else "    pass"}'''
            
            return class_code
        
        return f"// {name} class placeholder"
    
    def _generate_imports(self, dependencies: List[str], language: str) -> List[str]:
        """Generate import statements."""
        imports = []

        if language == "python":
            # Standard imports
            imports.extend([
                "import logging",
                "import sys",
                "from typing import Dict, Any, Optional, List"
            ])

            # Dependency imports
            for dep in dependencies:
                if dep == "fastapi":
                    imports.append("from fastapi import FastAPI, HTTPException")
                elif dep == "sqlalchemy":
                    imports.append("from sqlalchemy import create_engine")
        
        elif language in ["javascript", "typescript"]:
            for dep in dependencies:
                imports.append(f"const {dep} = require('{dep}');")
        
        return imports
    
    def _generate_tests(self, generated_code: Dict[str, Any], requirements: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate test cases."""
        tests = []
        language = requirements.get("language", "python")
        
        if language == "python":
            tests.append({
                "name": "test_main_functionality",
                "code": '''import pytest
from main import MainApplication

def test_main_application_init():
    """Test main application initialization."""
    app = MainApplication()
    assert app is not None
    
def test_main_application_initialize():
    """Test application initialization."""
    app = MainApplication()
    app.initialize()
    assert app.initialized is True'''
            })
        
        return tests
    
    def _generate_documentation(self, generated_code: Dict[str, Any], architecture: Dict[str, Any]) -> str:
        """Generate code documentation."""

        doc_sections = []

        # Title and description
        doc_sections.append("# Generated Code Documentation")
        doc_sections.append("\nThis code was generated by CapibaraGPT Coding Agent.\n")

        # Architecture
        doc_sections.append("## Architecture")
        components = architecture.get("components", [])
        if components:
            doc_sections.append("### Components:")
            for comp in components:
                doc_sections.append(f"- **{comp['name']}**: {', '.join(comp.get('responsibilities', []))}")

        # Functions
        functions = generated_code.get("functions", [])
        if functions:
            doc_sections.append("\n## Functions")
            for func in functions:
                doc_sections.append(f"### {func['name']}")
                doc_sections.append("Generated utility function.")

        # Classes
        classes = generated_code.get("classes", [])
        if classes:
            doc_sections.append("\n## Classes")
            for cls in classes:
                doc_sections.append(f"### {cls['name']}")
                doc_sections.append("Generated class component.")

        # Usage
        doc_sections.append("\n## Usage")
        doc_sections.append("```python")
        doc_sections.append("# Example usage")
        doc_sections.append("python main.py")
        doc_sections.append("```")
        
        return "\n".join(doc_sections)
    
    def _calculate_code_quality(self, generated_code: Dict[str, Any]) -> Dict[str, float]:
        """Calculate code quality metrics."""

        main_code = generated_code.get("main_code", "")
        functions = generated_code.get("functions", [])
        classes = generated_code.get("classes", [])

        # Basic metrics
        total_lines = len(main_code.split("\n"))
        comment_lines = len([line for line in main_code.split("\n") if line.strip().startswith("#")])
        
        return {
            "complexity_score": 0.8,  # Simulado
            "documentation_coverage": min(1.0, comment_lines / max(1, total_lines) * 4),
            "modularity_score": min(1.0, (len(functions) + len(classes)) / 5),
            "overall_quality": 0.85
        }


# Export all advanced behaviors
__all__ = [
    "ResearchBehavior",
    "CodingBehavior"
]
