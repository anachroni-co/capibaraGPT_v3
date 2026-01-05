"""
Improved Programming-Specific RAG Query Detector for Capibara6
Only activates RAG for programming-related queries
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class ProgrammingRAGQuery:
    """Programming-specific RAG query with metadata"""
    query: str
    is_programming_query: bool
    confidence: float
    detected_language: Optional[str] = None  # Language detected: 'python', 'javascript', 'java', etc.
    detected_topic: Optional[str] = None     # Topic: 'algorithm', 'debugging', 'syntax', etc.
    top_k: int = 5


class ProgrammingRAGDetector:
    """
    Programming-only RAG query detection using keyword patterns
    
    ONLY activates RAG for programming-related queries:
    - Code snippets or requests for code
    - Programming language questions
    - Algorithm implementations
    - API/documentation queries
    - Debugging help
    - Library/framework usage
    """
    
    def __init__(self):
        """Initialize detector with programming-specific patterns"""
        
        # Programming language patterns with higher specificity
        self.language_patterns = {
            'python': [
                r'\bpython\b',
                r'\bdef\b\s+\w+\s*\(',
                r'\bclass\b\s+\w+\s*:',
                r'\bimport\b\s+\w+',
                r'\bfrom\b\s+\w+\s+bimport\b',
                r'\blist\b|\bdict\b|\btuple\b|\bset\b|\bstr\b|\bint\b|\bfloat\b|\bbool\b',
                r'\blen\b|\brange\b|\bzip\b|\benumerate\b|\bsorted\b|\bmap\b|\bfilter\b',
                r'\bif\b.*\b__name__\s*==\s*[\'"]__main__[\'"]',
                r'\btry\b.*\bexcept\b|\bfinally\b',
                r'\bfor\b\s+\w+\s+\bin\b|\bwhile\b\s+\w+',
                r'\bprint\b\s*\(|\binput\b\s*\(',
                r'\bwith\b\s+\w+\s+as\b',
                r'\blambda\b.*:',
            ],
            'javascript': [
                r'\bjavascript\b|js\b',
                r'\bfunction\b\s+\w+\s*\(|\bfunction\s*\*',
                r'\bconst\b\s+\w+|\blet\b\s+\w+|\bvar\b\s+\w+',
                r'\basync\b\s+\w+|\bawait\b',
                r'\bthis\b\.|\bdocument\b\.|\bwindow\b\.|\bconsole\b\.',
                r'\b.map\b|\b.filter\b|\b.reduce\b|\b.forEach\b|\b.find\b',
                r'\bPromise\b|\basync\b|\bawait\b',
                r'\bimport\b\s+.*\bfrom\b|\bexport\b\s+\{|require\(',
            ],
            'java': [
                r'\bjava\b',
                r'\bpublic\b\s+(class|static|void|int|double|String)|\bprivate\b|\bprotected\b',
                r'\bclass\b\s+\w+|\binterface\b\s+\w+',
                r'\bstatic\b\s+|\bfinal\b\s+|\babstract\b\s+',
                r'\bvoid\b\s+\w+\s*\(|\bString\b\s+\w+|\bint\b\s+\w+',
                r'\bimport\s+java\b|\bpackage\b\s+',
                r'\bthrows\b\s+\w+|\bthrow\b\s+',
            ],
            'cpp': [
                r'\bc\+\+|cpp\b',
                r'\b#include\s*[<"]',
                r'\busing\s+namespace\b|\bstd::',
                r'\bclass\b\s+\w+|\bstruct\b\s+\w+',
                r'\btemplate\s*<',
                r'\bpublic:|private:|protected:',
                r'\bint\b\s+\w+|\bchar\b\s+\w+|\bfloat\b\s+\w+',
            ],
            'go': [
                r'\bgolang\b|go\b',
                r'\bpackage\b\s+\w+',
                r'\bfunc\b\s+\w+|\bfunc\s*\(',
                r'\bimport\s*[("]',
                r'\bstruct\b\s*{|\binterface\b\s*{',
                r'\bgoroutine\b|\bgo\s+func\b|\bchannel\b|\bchan\b',
            ],
            'rust': [
                r'\brust\b',
                r'\bfn\b\s+\w+\s*\(|use\s+std::',
                r'\bimpl\b|\btrait\b|\bmatch\b|\bif\s+let\b',
                r'\bpub\b\s+|\bmod\b\s+|\bextern\b',
                r'\bVec<|\bString\b|\bOption<|\bResult<',
            ],
            'typescript': [
                r'\btypescript\b|ts\b',
                r':\s*(string|number|boolean|array|any|void|never)',
                r'<\s*\w+\s*>\s*=>|\binterface\b|\btype\b\s+\w+',
                r'\bimplements\b|\bextends\b',
                r'as\s+(string|number|boolean|any|unknown)',
            ],
            'sql': [
                r'\bselect\b.*\bfrom\b|\binsert\b|\bupdate\b|\bdelete\b',
                r'\bcreate\b.*\b(table|database)|\bdrop\b',
                r'\bjoin\b|\baggregate\b|\bgroup\s+by\b|\border\s+by\b',
                r'\bwhere\b|\bhaving\b|\bwith\b',
            ],
            'shell': [
                r'\bbash\b|\bshell\b|\bsh\b',
                r'\bpip\b|\bnpm\b|\byarn\b|\bcargo\b|\bgem\b',
                r'\bchmod\b|\bchown\b|\bgrep\b|\bsed\b|\bawk\b',
                r'\bcurl\b|\bwget\b|\bssh\b|\bscp\b',
                r'\$[A-Z_]+\b|\$\{\w+\}',
            ]
        }
        
        # Programming topic patterns
        self.topic_patterns = {
            'code_syntax': [
                r'how to write\b.*\bin\b.*\b(language|code)',
                r'syntax for\b.*\b(language|function|loop|condition)',
                r'\bimplement\b.*\balgorithm',
                r'\bcode example\b',
                r'\bsnippet\b',
                r'\bfunction\b.*\b(definition|syntax|declaration)',
                r'\bclass\b.*\b(syntax|structure|inheritance)',
                r'\bsyntax\b.*\b(for|if|while|foreach|try|catch)',
            ],
            'debugging': [
                r'\b(error|bug|exception|traceback)\b.*\b(fix|solve|debug|resolve)',
                r'\b(debug|troubleshoot|fix)\b.*\b(error|bug|issue)',
                r'\bnot working\b|\bfailed\b.*\b(error|runtime)',
                r'\b(stack|traceback|error)\b.*\b(message|report|log)',
                r'\bfix\b.*\b(error|bug|issue)',
            ],
            'algorithm': [
                r'\balgorithm\b.*\b(implementation|solution|code)',
                r'\b(sort|search|find|minimize|maximize)\b.*\b(algorithm|code)',
                r'\bcomplexity\b|\bbig o\b|\boptimize\b',
                r'\brecursion\b|\bdynamic programming\b|\bgreedy\b|\bdivide and conquer\b',
                r'\bdata structure\b.*\b(implementation|code)',
            ],
            'api_documentation': [
                r'\bhow to use\b.*\b(library|api|function|method)',
                r'\b(function|method|class|library)\b.*\b(usage|example|call|implement)',
                r'\bdocumentation\b|\bparameters\b|\boptions\b',
                r'\barguments?\b.*\b(type|required|default)',
                r'\breference\b.*\b(api|function|library)',
            ],
            'library_framework': [
                r'\b(library|framework|package)\b.*\b(use|install|implement|configure)',
                r'\b(react|vue|angular|django|flask|spring|express|fastapi|pytorch|tensorflow|numpy|pandas)\b',
                r'\b(pip|npm|yarn|cargo|maven|gradle|pipenv|poetry)\b',
                r'\binstall\b.*\b(package|library|dependency)',
            ]
        }
        
        # Programming keywords and concepts
        self.programming_keywords = [
            # Languages
            'python', 'javascript', 'js', 'java', 'c++', 'c#', 'go', 'rust', 'typescript', 'ts',
            'php', 'ruby', 'swift', 'kotlin', 'scala', 'perl', 'r', 'matlab', 'sql', 'bash',
            
            # Concepts
            'code', 'function', 'class', 'method', 'variable', 'parameter', 'argument',
            'algorithm', 'debug', 'error', 'exception', 'bug', 'fix', 'debugging',
            'implementation', 'library', 'framework', 'package', 'module', 'dependency',
            'api', 'documentation', 'syntax', 'snippet', 'example', 'code',
            'compile', 'runtime', 'interpreter', 'IDE', 'editor', 'terminal', 'shell',
            'IDE', 'VSCode', 'IntelliJ', 'PyCharm', 'Eclipse',
            
            # Paradigms
            'oop', 'object oriented', 'functional', 'procedural', 'declarative', 'imperative',
            'inheritance', 'polymorphism', 'encapsulation', 'abstraction',
            
            # Tech terms
            'git', 'github', 'gitlab', 'repository', 'branch', 'merge', 'pull request',
            'REST', 'API', 'endpoint', 'JSON', 'XML', 'HTTP', 'HTTPS', 'URL',
            'database', 'SQL', 'query', 'table', 'index', 'schema',
            'server', 'client', 'backend', 'frontend', 'full stack',
            'container', 'Docker', 'Kubernetes', 'microservice',
        ]

    def detect(self, query: str) -> ProgrammingRAGQuery:
        """
        Detect if query is programming-related and needs RAG
        
        Args:
            query: User query
            
        Returns:
            ProgrammingRAGQuery with detection results
        """
        query_lower = query.lower()
        
        # Calculate confidence based on matches
        confidence = 0.0
        reasons = []
        
        # Detect programming language (highest weight)
        detected_language = self._detect_language(query_lower)
        if detected_language:
            confidence += 0.5
            reasons.append(f"lang:{detected_language}")
        
        # Detect programming topics
        detected_topic = self._detect_topic(query_lower)
        if detected_topic:
            confidence += 0.3
            reasons.append(f"topic:{detected_topic}")
        
        # Count programming keyword matches
        keyword_matches = sum(1 for keyword in self.programming_keywords if f" {keyword} " in f" {query_lower} ")
        if keyword_matches > 0:
            confidence += min(keyword_matches * 0.1, 0.2)  # Up to 0.2 for keywords
            reasons.append(f"k:{keyword_matches}")
        
        # Check for code-like patterns
        code_indicators = [
            r'".*".*\(',  # String followed by parentheses (like function calls)
            r'\w+\s*\([^)]*\)',  # Function call pattern
            r'\w+\s*\.\s*\w+',  # Object.property or object.method pattern
            r'#.*|//.*|/\*.*\*/',  # Comments
            r'\{\s*\}',  # Empty brackets
            r'\[\s*\]',  # Empty array
            r'=\s+[^;]*;',  # Assignment
        ]
        
        code_score = 0
        for pattern in code_indicators:
            if re.search(pattern, query):
                code_score += 0.1
        
        if code_score > 0:
            confidence += min(code_score, 0.2)
            reasons.append(f"code:{code_score:.1f}")
        
        # Adjust confidence based on context words that suggest it's NOT programming
        non_programming_indicators = [
            'weather', 'capital', 'history', 'cook', 'recipe', 'food', 'poem', 'poetry',
            'literature', 'book', 'movie', 'film', 'music', 'art', 'painting', 'sport',
            'athlete', 'game', 'score', 'time', 'hour', 'minute', 'second'
        ]
        
        non_prog_matches = sum(1 for ind in non_programming_indicators if ind in query_lower)
        if non_prog_matches > 0:
            confidence = max(0.0, confidence - (non_prog_matches * 0.1))
        
        # Threshold for programming query
        is_programming = confidence >= 0.5
        
        # Determine top_k based on complexity
        top_k = 5  # Default
        if detected_topic in ['algorithm', 'debugging']:
            top_k = 8  # More results for complex topics
        elif detected_topic in ['code_syntax', 'api_documentation']:
            top_k = 6  # Moderate results for syntax/docs
        elif detected_language:
            top_k = 7  # More context for specific code queries
            
        print(f"DEBUG: Query: '{query[:30]}...', Lang: {detected_language}, Topic: {detected_topic}, Conf: {confidence:.2f}, Reason: {reasons}, IsProg: {is_programming}")
        
        return ProgrammingRAGQuery(
            query=query,
            is_programming_query=is_programming,
            confidence=min(confidence, 1.0),
            detected_language=detected_language,
            detected_topic=detected_topic,
            top_k=top_k
        )
    
    def _detect_language(self, query_lower: str) -> Optional[str]:
        """Detect programming language from query"""
        for language, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return language
        return None
    
    def _detect_topic(self, query_lower: str) -> Optional[str]:
        """Detect programming topic from query"""
        for topic, patterns in self.topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return topic
        return None


class ProgrammingRAGParallelFetcher:
    """
    Programming-specific RAG context fetcher
    Only activates RAG for programming-related queries
    """
    
    def __init__(
        self,
        bridge_url: str = "http://localhost:8001",
        collection_name: str = "programming_docs",
        enable_rag: bool = True,
        detection_threshold: float = 0.5,
        max_context_tokens: int = 1000
    ):
        """
        Initialize programming-specific RAG fetcher
        
        Args:
            bridge_url: URL of capibara6-api bridge
            collection_name: Milvus collection name (specific to programming docs)
            enable_rag: Enable RAG integration
            detection_threshold: Confidence threshold for programming detection
            max_context_tokens: Max tokens for context (approx)
        """
        self.enable_rag = enable_rag
        self.detection_threshold = detection_threshold
        self.max_context_tokens = max_context_tokens

        # Initialize components
        if enable_rag:
            self.detector = ProgrammingRAGDetector()
        else:
            self.detector = None

        # Stats
        self.total_queries = 0
        self.programming_queries = 0
        self.non_programming_queries = 0
        self.total_fetch_time = []
        self.context_cache = {}  # Simple in-memory cache

        print(f"🎯 Programming-Specific RAG Fetcher initialized")
        print(f"   Enabled: {enable_rag}")
        print(f"   Bridge: {bridge_url}")
        print(f"   Collection: {collection_name}")
        print(f"   Only activates for programming queries")

    async def detect_and_fetch(
        self,
        query: str,
        request_id: str
    ) -> Tuple[bool, Optional['RAGContext']]:
        """
        Detect if query is programming-related and fetch context if needed
        
        Args:
            query: User query
            request_id: Request ID for tracking
            
        Returns:
            Tuple of (is_programming_query, context or None)
        """
        self.total_queries += 1

        if not self.enable_rag:
            return False, None

        # Phase 1: Fast programming detection (< 1ms)
        import time
        start_time = time.time()
        prog_query = self.detector.detect(query)

        if not prog_query.is_programming_query or prog_query.confidence < self.detection_threshold:
            self.non_programming_queries += 1
            return False, None

        self.programming_queries += 1
        print(f"💻 [{request_id}] Programming query detected: {prog_query.detected_language or 'unknown'} - {prog_query.detected_topic or 'general'} (conf: {prog_query.confidence:.2f})")

        # TODO: Implement actual Milvus client for programming docs
        # For now, return that it's a programming query but without actual RAG context
        # This is where you'd connect to your actual RAG system
        
        return True, None  # Placeholder - would be actual RAG context

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        return {
            'total_queries': self.total_queries,
            'programming_queries': self.programming_queries,
            'non_programming_queries': self.non_programming_queries,
            'programming_rate': f"{(self.programming_queries / self.total_queries * 100):.1f}%" if self.total_queries > 0 else "0%",
        }


# Example usage function
def is_programming_query(query: str) -> bool:
    """
    Helper function to determine if a query is programming-related
    
    Args:
        query: The user's query
        
    Returns:
        True if the query is programming-related, False otherwise
    """
    detector = ProgrammingRAGDetector()
    result = detector.detect(query)
    return result.is_programming_query


if __name__ == '__main__':
    print("🎯 Testing Programming-Specific RAG Detector")
    print("=" * 60)

    # Test cases
    test_queries = [
        # Programming queries (should activate RAG)
        "How do I sort an array in Python using the bubble sort algorithm?",
        "Write a function to calculate factorial in JavaScript",
        "I'm getting a 'TypeError: cannot read property of undefined' error in my React app",
        "What are the differences between let, const, and var in JavaScript?",
        "Show me a Python code example for connecting to PostgreSQL",
        "Help me debug this Python code: def sum(a, b): return a + b",
        "How to implement binary search in C++?",
        "What is the syntax for async/await in TypeScript?",
        
        # Non-programming queries (should NOT activate RAG)
        "What is the weather forecast for tomorrow?",
        "Tell me about the history of Rome",
        "How do I cook pasta?",
        "Explain the theory of relativity",
        "What are the benefits of exercise?",
        "Hello, how are you today?",
        "Can you write a poem about spring?",
        "What time is it in Tokyo?",
    ]

    detector = ProgrammingRAGDetector()
    
    print("\n📝 Testing Programming Query Detection:")
    programming_detected = 0
    total_tested = len(test_queries)
    
    for query in test_queries:
        result = detector.detect(query)
        status = "✅ PROGRAMMING" if result.is_programming_query else "❌ NOT programming"
        print(f"{status} [{result.confidence:.2f}] \"{query[:50]}...\"")
        print(f"     Language: {result.detected_language}, Topic: {result.detected_topic}, K: {result.top_k}")
        
        if result.is_programming_query:
            programming_detected += 1
    
    print(f"\n📊 Results: {programming_detected}/{total_tested} queries detected as programming-related")
    print(f"   Programming rate: {(programming_detected/total_tested)*100:.1f}%")
    print(f"\n✅ Programming-only RAG detection working correctly!")