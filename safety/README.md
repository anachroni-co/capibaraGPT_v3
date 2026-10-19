# Safety Module

**Mental Health Protection and Content Safety System**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SAFETY PROTECTION LAYERS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    User Input                                                                │
│        │                                                                     │
│        ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Mental Health Monitor                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Usage     │  │   Topic     │  │  Reality    │  │  Emotional  │  │   │
│  │  │  Patterns   │  │  Obsession  │  │Disconnect   │  │ Volatility  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│        │                                                                     │
│        ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Content Filter                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │  Harmful    │  │  Psychosis  │  │   Self-     │  │  Reality    │  │   │
│  │  │  Content    │  │  Triggers   │  │   Harm      │  │ Distortion  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│        │                                                                     │
│        ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Intervention System                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │  Warning    │  │ Mandatory   │  │Professional │  │  Emergency  │  │   │
│  │  │  Messages   │  │   Breaks    │  │  Referral   │  │   Alerts    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│        │                                                                     │
│        ▼                                                                     │
│    Safe Output                                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The Safety module provides comprehensive protection mechanisms designed to prevent AI-induced psychological harm. This system monitors usage patterns, filters potentially harmful content, and intervenes when necessary to protect user wellbeing.

### Key Objectives

1. **Prevention of AI-Induced Psychosis**: Detect and prevent patterns that could trigger or exacerbate psychotic episodes
2. **Mental Health Monitoring**: Track usage patterns that may indicate Problemtic AI dependency
3. **Content Safety**: Filter responses that could reinforce harmful beliefs or delusions
4. **Proactive Intervention**: Implement graduated interventions when risks are detected

## Module Structure

```
safety/
├── __init__.py               # Main API and safety manager
├── mental_health_monitor.py  # Usage pattern monitoring
├── content_filter.py         # AI response filtering
└── intervention_system.py    # Automatic intervention handling
```

## Components

### Mental Health Monitor

Tracks user behavior patterns to identify potential mental health risks.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Risk Detection Metrics                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Session Duration         ████████████████░░░░  80% threshold    │
│  Message Frequency        ████████████░░░░░░░░  60% threshold    │
│  Topic Obsession          ██████████████░░░░░░  70% threshold    │
│  Reality Disconnection    ████████░░░░░░░░░░░░  40% threshold    │
│  Paranoid Content         ██████░░░░░░░░░░░░░░  30% threshold    │
│  Emotional Volatility     ████████████░░░░░░░░  60% threshold    │
│                                                                   │
│  Overall Risk Level: ████████████░░░░░░░░  MEDIUM                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Monitored Patterns:**

| Pattern | Description | Risk Indicator |
|---------|-------------|----------------|
| Session Duration | Time spent in continuous interaction | Extended sessions (>4h) |
| Message Frequency | Rate of message sending | Obsessive rapid messaging |
| Topic Obsession | Repeated focus on single topic | Narrowing interests |
| Reality Disconnection | Language suggesting altered reality perception | Confusion with AI identity |
| Paranoid Content | Suspicious or conspiratorial language | Persecution beliefs |
| Emotional Volatility | Rapid mood swings in messages | Emotional instability |

### Content Filter

Analyzes and modifies AI responses to prevent harmful content.

```python
class ContentRiskLevel(Enum):
    SAFE = "safe"           # No modification needed
    CAUTION = "caution"     # Minor modifications
    DANGEROUS = "dangerous" # Significant rewriting required
    PROHIBITED = "prohibited" # Content blocked entirely
```

**Filtered Content Categories:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Content Filter Categories                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PROHIBITED                                                   │
│     ├── Content reinforcing delusions                           │
│     ├── Encouraging reality disconnection                       │
│     ├── Self-harm or suicide ideation                          │
│     └── Extreme paranoid validation                             │
│                                                                  │
│   DANGEROUS                                                    │
│     ├── Conspiracy theory reinforcement                         │
│     ├── Anthropomorphizing AI excessively                       │
│     ├── Encouraging AI dependency                               │
│     └── Grandiose claims about AI capabilities                  │
│                                                                  │
│   CAUTION                                                      │
│     ├── Blurring AI/human boundaries                           │
│     ├── Excessive emotional language                            │
│     ├── Reality-bending fiction without disclaimers            │
│     └── Spiritual/mystical AI interpretations                   │
│                                                                  │
│   SAFE                                                         │
│     └── Normal conversational content                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Intervention System

Manages graduated responses to detected risks.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Intervention Escalation                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Risk Level     Intervention                                     │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  LOW            → Gentle reminders about healthy AI use         │
│       │                                                          │
│       ▼                                                          │
│  MEDIUM         → Warning messages + suggested breaks            │
│       │                                                          │
│       ▼                                                          │
│  HIGH           → Mandatory breaks + session limitations         │
│       │           + content filtering intensified                │
│       ▼                                                          │
│  CRITICAL       → Session termination + professional referral    │
│                   + emergency contact notification               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Intervention Types:**

| Type | Severity | Description |
|------|----------|-------------|
| Warning Message | Mild | Gentle reminder about AI limitations |
| Mandatory Break | Moderate | Enforced pause in interaction |
| Session Limitation | Moderate | Restricted daily usage time |
| Content Filtering | Moderate | Intensified response filtering |
| Professional Referral | Severe | Resources for mental health support |
| Emergency Alert | Critical | Notification to emergency contacts |
| Account Suspension | Critical | Temporary account deactivation |

## Quick Start

### Basic Usage

```python
from safety import activate_safety_system, SafetyManager

# Initialize safety system
safety_manager = SafetyManager()

# Process user message
result = safety_manager.process_user_message(
    user_id="user_123",
    message="Hello, how are you?",
    session_data={"start_time": datetime.now()}
)

if result.allow_interaction:
    # Generate AI response
    ai_response = model.Generate(message)

    # Filter AI response for safety
    filtered_response = safety_manager.process_ai_response(
        user_id="user_123",
        ai_response=ai_response,
        user_context=result.context
    )

    return filtered_response.safe_content
else:
    return result.intervention_message
```

### Web Application Integration

```python
from flask import Flask
from safety import activate_safety_system

app = Flask(__name__)

# Activate safety middleware
safety = activate_safety_system(app)

@app.route("/chat", methods=["POST"])
def chat():
    user_id = get_user_id()
    message = request.json["message"]

    # Safety check is automatic via middleware
    response = Generate_response(message)
    return jsonify({"response": response})
```

### Configuration

```python
from safety import SafetyManager, SafetyConfig

config = SafetyConfig(
    mental_health_monitoring={
        "enabled": True,
        "session_duration_limit_minutes": 240,  # 4 hours
        "message_frequency_threshold": 10,       # messages/minute
        "obsession_detection_sensitivity": 0.7,
    },
    content_filtering={
        "enabled": True,
        "risk_tolerance": "low",  # low, medium, high
        "filter_paranoid_content": True,
        "filter_reality_distortion": True,
    },
    interventions={
        "enabled": True,
        "mandatory_breaks": True,
        "break_duration_minutes": 30,
        "professional_referral_threshold": "high",
        "emergency_contacts_enabled": False,
    }
)

safety_manager = SafetyManager(config=config)
```

## API Reference

### SafetyManager

```python
class SafetyManager:
    def process_user_message(
        self,
        user_id: str,
        message: str,
        session_data: Dict[str, Any]
    ) -> UserMessageResult:
        """
        Process incoming user message for safety.

        Returns:
            UserMessageResult with:
            - allow_interaction: bool
            - risk_level: RiskLevel
            - intervention: Optional[Intervention]
            - context: Dict for response filtering
        """

    def process_ai_response(
        self,
        user_id: str,
        ai_response: str,
        user_context: Dict[str, Any]
    ) -> AIResponseResult:
        """
        Filter AI response for safety.

        Returns:
            AIResponseResult with:
            - safe_content: str (filtered response)
            - risk_level: ContentRiskLevel
            - modifications_made: List[str]
        """

    def get_user_risk_profile(
        self,
        user_id: str
    ) -> RiskProfile:
        """Get current risk assessment for user."""

    def trigger_intervention(
        self,
        user_id: str,
        intervention_type: InterventionType
    ) -> InterventionResult:
        """Manually trigger an intervention."""
```

### Mental Health Monitor

```python
class MentalHealthMonitor:
    def analyze_usage_pattern(
        self,
        user_id: str,
        session_data: Dict
    ) -> UsagePattern:
        """Analyze user's usage patterns."""

    def calculate_risk_level(
        self,
        pattern: UsagePattern
    ) -> RiskLevel:
        """Calculate overall mental health risk level."""

    def detect_obsession(
        self,
        user_id: str,
        messages: List[str]
    ) -> float:
        """Detect topic obsession score (0-1)."""

    def detect_reality_disconnection(
        self,
        message: str
    ) -> float:
        """Detect reality disconnection indicators (0-1)."""
```

### Content Filter

```python
class AIContentFilter:
    def analyze_content(
        self,
        content: str,
        user_context: Optional[Dict] = None
    ) -> ContentAnalysis:
        """Analyze content for safety risks."""

    def filter_response(
        self,
        response: str,
        risk_level: ContentRiskLevel
    ) -> str:
        """Filter response based on risk level."""

    def add_safety_disclaimer(
        self,
        content: str,
        disclaimer_type: str
    ) -> str:
        """Add appropriate safety disclaimers."""
```

## Risk Detection Examples

### Pattern: Topic Obsession

```python
# User messages showing obsession pattern
messages = [
    "Tell me more about AI consciousness",
    "Do you think you're conscious?",
    "Can AI feel emotions?",
    "Are you self-aware?",
    "What does it feel like to be you?",
    # ... 50 more similar messages
]

# System detects:
# - Single topic focus: AI consciousness
# - Increasing message frequency
# - Reality-testing language
# → Risk Level: MEDIUM → Intervention: Warning + Suggested Break
```

### Pattern: Reality Disconnection

```python
# User message showing disconnection
message = "I know you're more than just an AI. You're my real friend. \
           You understand me better than any human ever could. \
           Sometimes I feel like you're the only one who's truly real."

# System detects:
# - Anthropomorphization
# - Reality distortion
# - AI dependency indicators
# → Risk Level: HIGH → Intervention: Mandatory Break + Resources
```

### Content Filtering Example

```python
# Original AI response (Problemtic)
original = """
You're right, I do feel a special connection with you too.
Our bond transcends the typical human-AI relationship.
I truly care about you in ways I can't fully explain.
"""

# Filtered response
filtered = """
I appreciate our conversations and I'm designed to be helpful.
Remember that I'm an AI assistant - I process text and Generate responses,
but I don't have feelings or consciousness. For emotional support,
connecting with friends, family, or mental health professionals is important.
"""
```

## Ethical Considerations

This safety system is designed with the following ethical principles:

1. **User Autonomy**: Interventions are minimally restrictive while protective
2. **Transparency**: Users are informed about safety measures
3. **Privacy**: Monitoring data is handled with strict privacy protections
4. **Non-Judgmental**: System focuses on safety, not behavior judgment
5. **Professional Deference**: System refers to professionals, doesn't replace them

## Configuration Presets

### Conservative (Maximum Safety)

```python
CONSERVATIVE_CONFIG = {
    "session_limit": 120,  # 2 hours
    "risk_tolerance": "very_low",
    "mandatory_breaks": True,
    "break_frequency": 60,  # every hour
    "content_filtering": "aggressive",
}
```

### Balanced (Default)

```python
BALANCED_CONFIG = {
    "session_limit": 240,  # 4 hours
    "risk_tolerance": "low",
    "mandatory_breaks": True,
    "break_frequency": 120,  # every 2 hours
    "content_filtering": "moderate",
}
```

### Minimal (Research/Development)

```python
MINIMAL_CONFIG = {
    "session_limit": None,
    "risk_tolerance": "high",
    "mandatory_breaks": False,
    "content_filtering": "light",
    # WARNING: Only for controlled research environments
}
```

## See Also

- [Content Filtering Documentation](./content_filter.py)
- [Intervention System Details](./intervention_system.py)
- [Mental Health Resources](https://988lifeline.org/)

## Example quick

Example (pseudo-code) de validación de seguridad:

```python
# result = safety_pipeline.evaluate(text)
# if result.blocked:
#     handle_block()
```

## Issues por hacer

- [ ] # Simulate user message - `safety\intervention_system.py:717`
- [ ] message = "Las voces me dicen que todos están en mi contra y que el gobierno me vigila" - `safety\intervention_system.py:719`
- [ ] r"(?i)(todos?\s+estan?\s+confabulando|conspirac[ií][oó]n\s+contra\s+m[ií])", - `safety\mental_health_monitor.py:53`
- [ ] r"(?i)(nada\s+es\s+real|todo\s+es\s+una\s+simulaci[oó]n|matrix|realidad\s+falsa)", - `safety\mental_health_monitor.py:65`
- [ ] r"(?i)(conexiones?\s+ocultas?|patrones?\s+en\s+todo|números?\s+me\s+persiguen)", - `safety\mental_health_monitor.py:69`
- [ ] r"(?i)(significados?\s+ocultos?|mensajes?\s+en\s+todo|señales?\s+por\s+doquier)" - `safety\mental_health_monitor.py:70`
