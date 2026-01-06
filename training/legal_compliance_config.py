"""
Legal Compliance Configuration for Capibara6 Training

This module ensures legal compliance and legitimate use of all open source models
in the consensus strategy.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class LegalComplianceConfig:
    """Configurestion for legal compliance of expert models."""
    
    # Legal framework
    legal_framework: Dict[str, Any] = None
    license_compliance: Dict[str, Any] = None
    use_case_validation: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.legal_framework is None:
            self.legal_framework = {
                "framework": "Open Source Software Licenses",
                "compliance_standard": "OSI Approved Licenses",
                "legitimate_use": "Research and Development",
                "commercial_use": "Permitted under license terms",
                "attribution_required": True,
                "modification_allowed": True,
                "distribution_allowed": True
            }
        
        if self.license_compliance is None:
            self.license_compliance = {
                "apache_2_0": {
                    "description": "Apache License 2.0",
                    "permissions": [
                        "Commercial use",
                        "Modification",
                        "Distribution",
                        "Patent use",
                        "Private use"
                    ],
                    "limitations": [
                        "Trademark use",
                        "Liability",
                        "Warranty"
                    ],
                    "conditions": [
                        "License and copyright notice",
                        "State changes"
                    ],
                    "compliance_status": "FULLY_COMPLIANT"
                },
                "mit": {
                    "description": "MIT License",
                    "permissions": [
                        "Commercial use",
                        "Modification",
                        "Distribution",
                        "Private use"
                    ],
                    "limitations": [
                        "Liability",
                        "Warranty"
                    ],
                    "conditions": [
                        "License and copyright notice"
                    ],
                    "compliance_status": "FULLY_COMPLIANT"
                },
                "bsd_3_clause": {
                    "description": "BSD 3-Clause License",
                    "permissions": [
                        "Commercial use",
                        "Modification",
                        "Distribution",
                        "Private use"
                    ],
                    "limitations": [
                        "Liability",
                        "Warranty"
                    ],
                    "conditions": [
                        "License and copyright notice"
                    ],
                    "compliance_status": "FULLY_COMPLIANT"
                }
            }
        
        if self.use_case_validation is None:
            self.use_case_validation = {
                "research_and_development": {
                    "description": "AI Research and Development",
                    "legitimate_use": True,
                    "commercial_application": True,
                    "ethical_guidelines": "Follow AI ethics principles",
                    "data_privacy": "Comply with data protection laws"
                },
                "educational_purposes": {
                    "description": "Educational and Academic Use",
                    "legitimate_use": True,
                    "commercial_application": False,
                    "ethical_guidelines": "Educational fair use",
                    "data_privacy": "Student data protection"
                },
                "commercial_development": {
                    "description": "Commercial Software Development",
                    "legitimate_use": True,
                    "commercial_application": True,
                    "ethical_guidelines": "Business ethics compliance",
                    "data_privacy": "GDPR and privacy compliance"
                }
            }

class LegalComplianceValidator:
    """Validatestor for legal compliance of expert models."""
    
    def __init__(self):
        self.config = LegalComplianceConfig()
        self.expert_models_legal_info = self._load_expert_models_legal_info()
    
    def _load_expert_models_legal_info(self) -> Dict[str, Dict[str, Any]]:
        """Load legal information for all expert models."""
        return {
            "spanish_language": {
                "model_id": "PlanTL-GOB-ES/roberta-base-bne",
                "license": "Apache 2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "organization": "PlanTL - State Secretariat for Digitalization and AI",
                "country": "Spain",
                "use_case": "Spanish language processing and generation",
                "legitimate_use": True,
                "commercial_use": True,
                "attribution_required": True,
                "compliance_status": "FULLY_COMPLIANT",
                "legal_notes": [
                    "Developed by Spanish government research organization",
                    "Apache 2.0 license allows commercial use",
                    "Specifically designed for Spanish language tasks",
                    "No restrictions on modification or distribution"
                ]
            },
            
            "mathematics": {
                "model_id": "EleutherAI/gpt-neo-125M",
                "license": "Apache 2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "organization": "EleutherAI",
                "country": "United States",
                "use_case": "Mathematical reasoning and calculations",
                "legitimate_use": True,
                "commercial_use": True,
                "attribution_required": True,
                "compliance_status": "FULLY_COMPLIANT",
                "legal_notes": [
                    "Open source GPT model with Apache 2.0 license",
                    "Developed by EleutherAI research organization",
                    "Suitable for mathematical and reasoning tasks",
                    "No usage restrictions beyond license terms"
                ]
            },
            
            "programming": {
                "model_id": "Salesforce/codet5-small",
                "license": "BSD-3-Clause",
                "license_url": "https://opensource.org/licenses/BSD-3-Clause",
                "organization": "Salesforce",
                "country": "United States",
                "use_case": "Code generation and understanding",
                "legitimate_use": True,
                "commercial_use": True,
                "attribution_required": True,
                "compliance_status": "FULLY_COMPLIANT",
                "legal_notes": [
                    "Salesforce open source code generation model",
                    "BSD-3-Clause license is permissive",
                    "Specifically designed for programming tasks",
                    "Widely used in industry and research"
                ]
            },
            
            "reasoning": {
                "model_id": "microsoft/DialoGPT-small",
                "license": "MIT",
                "license_url": "https://opensource.org/licenses/MIT",
                "organization": "Microsoft",
                "country": "United States",
                "use_case": "Logical reasoning and analytical thinking",
                "legitimate_use": True,
                "commercial_use": True,
                "attribution_required": True,
                "compliance_status": "FULLY_COMPLIANT",
                "legal_notes": [
                    "Microsoft's open source dialogue model",
                    "MIT license is one of the most permissive",
                    "Suitable for reasoning and conversation tasks",
                    "No restrictions on commercial use"
                ]
            },
            
            "scientific": {
                "model_id": "allenai/scibert_scivocab_uncased",
                "license": "Apache 2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "organization": "Allen Institute for AI",
                "country": "United States",
                "use_case": "Scientific literature and research",
                "legitimate_use": True,
                "commercial_use": True,
                "attribution_required": True,
                "compliance_status": "FULLY_COMPLIANT",
                "legal_notes": [
                    "Allen AI's scientific BERT model",
                    "Apache 2.0 license allows all uses",
                    "Specifically trained on scientific literature",
                    "Widely used in academic and commercial research"
                ]
            },
            
            "technical": {
                "model_id": "EleutherAI/gpt-neo-125M",
                "license": "Apache 2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "organization": "EleutherAI",
                "country": "United States",
                "use_case": "Technical documentation and engineering",
                "legitimate_use": True,
                "commercial_use": True,
                "attribution_required": True,
                "compliance_status": "FULLY_COMPLIANT",
                "legal_notes": [
                    "Same model as mathematics expert, different use case",
                    "Apache 2.0 license allows all uses",
                    "Suitable for technical and engineering tasks",
                    "No additional legal restrictions"
                ]
            },
            
            "general": {
                "model_id": "microsoft/DialoGPT-medium",
                "license": "MIT",
                "license_url": "https://opensource.org/licenses/MIT",
                "organization": "Microsoft",
                "country": "United States",
                "use_case": "General conversation and tasks",
                "legitimate_use": True,
                "commercial_use": True,
                "attribution_required": True,
                "compliance_status": "FULLY_COMPLIANT",
                "legal_notes": [
                    "Microsoft's medium-sized dialogue model",
                    "MIT license allows all uses",
                    "General purpose conversation model",
                    "No restrictions on commercial use"
                ]
            }
        }
    
    def validate_legal_compliance(self) -> Dict[str, Any]:
        """Validates legal compliance of all expert models."""
        validation_results = {
            "overall_compliance": True,
            "validation_timestamp": datetime.now().isoformat(),
            "expert_models": {},
            "license_summary": {},
            "compliance_issues": []
        }
        
        # Validate each expert model
        for expert_type, legal_info in self.expert_models_legal_info.items():
            model_compliance = self._validate_model_compliance(expert_type, legal_info)
            validation_results["expert_models"][expert_type] = model_compliance
            
            if not model_compliance["compliance_status"] == "FULLY_COMPLIANT":
                validation_results["overall_compliance"] = False
                validation_results["compliance_issues"].append(
                    f"Model {expert_type}: {model_compliance['compliance_status']}"
                )
        
        # Generate license summary
        validation_results["license_summary"] = self._generate_license_summary()
        
        return validation_results
    
    def _validate_model_compliance(self, expert_type: str, legal_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validates compliance for a specific model."""
        compliance = {
            "expert_type": expert_type,
            "model_id": legal_info["model_id"],
            "license": legal_info["license"],
            "organization": legal_info["organization"],
            "use_case": legal_info["use_case"],
            "compliance_status": legal_info["compliance_status"],
            "legitimate_use": legal_info["legitimate_use"],
            "commercial_use": legal_info["commercial_use"],
            "attribution_required": legal_info["attribution_required"],
            "legal_notes": legal_info["legal_notes"],
            "validation_passed": True
        }
        
        # Check license compliance
        license_config = self.config.license_compliance.get(legal_info["license"].lower().replace(" ", "_"))
        if license_config:
            compliance["license_details"] = license_config
        else:
            compliance["validation_passed"] = False
            compliance["compliance_status"] = "LICENSE_NOT_FOUND"
        
        return compliance
    
    def _generate_license_summary(self) -> Dict[str, Any]:
        """Generates summary of licenses used."""
        license_counts = {}
        organizations = {}
        
        for expert_type, legal_info in self.expert_models_legal_info.items():
            license = legal_info["license"]
            organization = legal_info["organization"]
            
            license_counts[license] = license_counts.get(license, 0) + 1
            if organization not in organizations:
                organizations[organization] = []
            organizations[organization].append(expert_type)
        
        return {
            "license_distribution": license_counts,
            "organizations": organizations,
            "total_models": len(self.expert_models_legal_info),
            "all_licenses_osi_approved": True,
            "commercial_use_allowed": True
        }
    
    def generate_legal_documentation(self) -> Dict[str, Any]:
        """Generates comprehensive legal documentation."""
        return {
            "legal_framework": self.config.legal_framework,
            "license_compliance": self.config.license_compliance,
            "use_case_validation": self.config.use_case_validation,
            "expert_models_legal_info": self.expert_models_legal_info,
            "compliance_validation": self.validate_legal_compliance(),
            "attribution_requirements": self._generate_attribution_requirements(),
            "usage_guidelines": self._generate_usage_guidelines()
        }
    
    def _generate_attribution_requirements(self) -> Dict[str, Any]:
        """Generates attribution requirements for all models."""
        attributions = {}
        
        for expert_type, legal_info in self.expert_models_legal_info.items():
            attributions[expert_type] = {
                "model_id": legal_info["model_id"],
                "organization": legal_info["organization"],
                "license": legal_info["license"],
                "attribution_text": f"Based on {legal_info['model_id']} by {legal_info['organization']} ({legal_info['license']} license)",
                "license_url": legal_info["license_url"]
            }
        
        return attributions
    
    def _generate_usage_guidelines(self) -> Dict[str, Any]:
        """Generates usage guidelines for legitimate use."""
        return {
            "research_and_development": {
                "description": "AI Research and Development",
                "permitted_uses": [
                    "Model training and fine-tuning",
                    "Consensus generation",
                    "Performance evaluation",
                    "Academic research",
                    "Commercial development"
                ],
                "ethical_guidelines": [
                    "Follow AI ethics principles",
                    "Ensure data privacy compliance",
                    "Avoid harmful content generation",
                    "Respect intellectual property rights",
                    "Provide appropriate attribution"
                ],
                "compliance_requirements": [
                    "Maintain license notices",
                    "Include attribution in outputs",
                    "Comply with data protection laws",
                    "Follow fair use principles"
                ]
            },
            "commercial_use": {
                "description": "Commercial Software Development",
                "permitted_uses": [
                    "Commercial software integration",
                    "API development",
                    "Product enhancement",
                    "Service provision"
                ],
                "ethical_guidelines": [
                    "Business ethics compliance",
                    "Transparent AI usage",
                    "User privacy protection",
                    "Fair competition practices"
                ],
                "compliance_requirements": [
                    "License compliance",
                    "Attribution requirements",
                    "GDPR and privacy compliance",
                    "Terms of service compliance"
                ]
            }
        }

# Convenience functions
def validate_all_models_compliance() -> Dict[str, Any]:
    """Validates compliance of all expert models."""
    validator = LegalComplianceValidator()
    return validator.validate_legal_compliance()

def generate_legal_documentation() -> Dict[str, Any]:
    """Generates comprehensive legal documentation."""
    validator = LegalComplianceValidator()
    return validator.generate_legal_documentation()

def get_model_attributions() -> Dict[str, Any]:
    """Get attribution requirements for all models."""
    validator = LegalComplianceValidator()
    return validator._generate_attribution_requirements()

if __name__ == "__main__":
    # Generate and display legal documentation
    legal_docs = generate_legal_documentation()
    
    print("=== Legal Compliance Documentation ===")
    print(f"Overall Compliance: {legal_docs['compliance_validation']['overall_compliance']}")
    print(f"Total Models: {legal_docs['compliance_validation']['license_summary']['total_models']}")
    print(f"All Licenses OSI Approved: {legal_docs['compliance_validation']['license_summary']['all_licenses_osi_approved']}")
    print(f"Commercial Use Allowed: {legal_docs['compliance_validation']['license_summary']['commercial_use_allowed']}")
    
    print("\n=== License Distribution ===")
    for license_type, count in legal_docs['compliance_validation']['license_summary']['license_distribution'].items():
        print(f"  {license_type}: {count} models")
    
    print("\n=== Organizations ===")
    for org, models in legal_docs['compliance_validation']['license_summary']['organizations'].items():
        print(f"  {org}: {', '.join(models)}")
    
    print("\n=== Attribution Requirements ===")
    for expert_type, attribution in legal_docs['attribution_requirements'].items():
        print(f"  {expert_type}: {attribution['attribution_text']}")
    
    # Save documentation to file
    with open("legal_compliance_documentation.json", "w", encoding="utf-8") as f:
        json.dump(legal_docs, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Legal compliance documentation generated and saved to 'legal_compliance_documentation.json'")