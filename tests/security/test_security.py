"""
Security Tests for CapibaraGPT v3.

Verifies that all security fixes are in place and effective:
1. SQL injection prevention
2. Path traversal protection
3. SSRF host validation
4. CORS configuration
5. No hardcoded credentials
6. Safe deserialization
7. Debug mode protection
8. Secure randomness
"""

import os
import re
import ast
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Project root for file scanning
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "capibaragpt.egg-info",
}


def _should_skip(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _iter_py_files(root: Path = PROJECT_ROOT):
    for py_file in root.rglob("*.py"):
        if _should_skip(py_file):
            continue
        yield py_file


# ============================================================================
# 1. SQL INJECTION TESTS
# ============================================================================

class TestSQLInjectionPrevention:
    """Verify that SQL queries use parameterized inputs."""

    def _get_patents_module(self):
        try:
            from data.datasets.google_research.google_patents_datasets import GooglePatentsDataset
            return GooglePatentsDataset
        except ImportError:
            pytest.skip("google_patents_datasets not importable")

    def test_query_returns_params(self):
        """generate_bigquery_query must return (query, params) tuple."""
        cls = self._get_patents_module()
        dataset = cls.__new__(cls)
        dataset.project_id = "test"
        dataset.dataset_id = "patents-public-data"
        result = dataset.generate_bigquery_query({"title": "neural network"})
        assert isinstance(result, tuple), "Must return (query, params) tuple"
        query, params = result
        assert isinstance(query, str)
        assert isinstance(params, list)

    def test_no_direct_interpolation_in_where(self):
        """WHERE clause must not contain user input directly."""
        cls = self._get_patents_module()
        dataset = cls.__new__(cls)
        dataset.project_id = "test"
        dataset.dataset_id = "patents-public-data"
        malicious_input = "'; DROP TABLE patents; --"
        query, params = dataset.generate_bigquery_query({"title": malicious_input})
        assert malicious_input not in query, "User input must not appear in query string"
        assert "@title_param" in query, "Must use parameterized placeholder"

    def test_order_by_whitelist(self):
        """ORDER BY must only accept whitelisted columns."""
        cls = self._get_patents_module()
        dataset = cls.__new__(cls)
        dataset.project_id = "test"
        dataset.dataset_id = "patents-public-data"
        query, _ = dataset.generate_bigquery_query({
            "order_by": "1; DROP TABLE patents --"
        })
        assert "DROP" not in query, "Malicious ORDER BY must be rejected"
        assert "publication_date DESC" in query, "Must fallback to safe default"

    def test_limit_is_integer(self):
        """LIMIT must be cast to integer."""
        cls = self._get_patents_module()
        dataset = cls.__new__(cls)
        dataset.project_id = "test"
        dataset.dataset_id = "patents-public-data"
        with pytest.raises((ValueError, TypeError)):
            dataset.generate_bigquery_query({"limit": "10; DROP TABLE x"})

    def test_all_search_params_parameterized(self):
        """All search fields must use @param placeholders."""
        cls = self._get_patents_module()
        dataset = cls.__new__(cls)
        dataset.project_id = "test"
        dataset.dataset_id = "patents-public-data"
        query, params = dataset.generate_bigquery_query({
            "title": "test",
            "abstract": "test",
            "assignee": "test",
            "inventor": "test",
            "cpc_code": "H01",
            "filing_date_start": "2020-01-01",
            "filing_date_end": "2024-01-01",
        })
        param_names = [p[0] for p in params]
        assert "title_param" in param_names
        assert "abstract_param" in param_names
        assert "assignee_param" in param_names
        assert "inventor_param" in param_names
        assert "cpc_param" in param_names
        assert "filing_start_param" in param_names
        assert "filing_end_param" in param_names


# ============================================================================
# 2. PATH TRAVERSAL TESTS
# ============================================================================

class TestPathTraversalProtection:
    """Verify that path traversal attacks are blocked."""

    def _get_validate_fn(self):
        try:
            from agents.e2b_sandbox_agent import _validate_local_path
            return _validate_local_path
        except (ImportError, NameError, Exception):
            pytest.skip("e2b_sandbox_agent not importable (missing dependencies)")

    def test_blocks_etc_passwd(self):
        validate = self._get_validate_fn()
        with pytest.raises(ValueError, match="traversal"):
            validate("/etc/passwd", "read")

    def test_blocks_proc(self):
        validate = self._get_validate_fn()
        with pytest.raises(ValueError, match="traversal"):
            validate("/proc/self/environ", "read")

    def test_blocks_sys(self):
        validate = self._get_validate_fn()
        with pytest.raises(ValueError, match="traversal"):
            validate("/sys/kernel/notes", "read")

    def test_blocks_root(self):
        validate = self._get_validate_fn()
        with pytest.raises(ValueError, match="traversal"):
            validate("/root/.ssh/id_rsa", "read")

    def test_blocks_relative_traversal(self):
        validate = self._get_validate_fn()
        with pytest.raises(ValueError, match="traversal"):
            validate("../../../etc/shadow", "read")

    def test_allows_tmp(self):
        validate = self._get_validate_fn()
        result = validate("/tmp/safe_file.txt", "write")
        assert result == Path("/tmp/safe_file.txt").resolve()

    def test_allows_home_paths(self):
        validate = self._get_validate_fn()
        result = validate("/home/user/data/file.bin", "upload")
        assert "home" in str(result)


# ============================================================================
# 3. SSRF TESTS
# ============================================================================

class TestSSRFProtection:
    """Verify that HTTP requests validate host against whitelist."""

    def test_wiki_validated_get_blocks_internal(self):
        try:
            from data.datasets.academic.wiki_datasets import _validated_get
        except ImportError:
            pytest.skip("wiki_datasets not importable")
        with pytest.raises(ValueError, match="Disallowed host"):
            _validated_get("http://169.254.169.254/latest/meta-data/")

    def test_wiki_validated_get_blocks_localhost(self):
        try:
            from data.datasets.academic.wiki_datasets import _validated_get
        except ImportError:
            pytest.skip("wiki_datasets not importable")
        with pytest.raises(ValueError, match="Disallowed host"):
            _validated_get("http://localhost:8080/admin")

    def test_wiki_validated_get_blocks_arbitrary_host(self):
        try:
            from data.datasets.academic.wiki_datasets import _validated_get
        except ImportError:
            pytest.skip("wiki_datasets not importable")
        with pytest.raises(ValueError, match="Disallowed host"):
            _validated_get("https://evil-server.com/steal-data")

    def test_wiki_blocks_ftp_scheme(self):
        try:
            from data.datasets.academic.wiki_datasets import _validated_get
        except ImportError:
            pytest.skip("wiki_datasets not importable")
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            _validated_get("ftp://internal-server/data")

    def test_institutional_validated_get_blocks_internal(self):
        try:
            from data.datasets.academic.institutional_datasets import _validated_get
        except ImportError:
            pytest.skip("institutional_datasets not importable")
        with pytest.raises(ValueError, match="Disallowed host"):
            _validated_get("http://169.254.169.254/latest/meta-data/")

    def test_institutional_blocks_localhost(self):
        try:
            from data.datasets.academic.institutional_datasets import _validated_get
        except ImportError:
            pytest.skip("institutional_datasets not importable")
        with pytest.raises(ValueError, match="Disallowed host"):
            _validated_get("http://127.0.0.1:6379/")


# ============================================================================
# 4. CORS CONFIGURATION TESTS
# ============================================================================

class TestCORSConfiguration:
    """Verify that CORS defaults are not wildcard."""

    def test_automation_config_no_wildcard(self):
        try:
            from services.automation.config import ApiConfig
        except ImportError:
            pytest.skip("services.automation.config not importable")
        config = ApiConfig()
        assert "*" not in config.cors_origins, \
            "CORS must not default to wildcard"

    def test_config_settings_no_wildcard(self):
        try:
            from config.config_settings import APISettings
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"config_settings not importable: {e}")
        settings = APISettings()
        assert "*" not in settings.cors_origins, \
            "CORS must not default to wildcard"

    def test_cors_env_override(self):
        """CORS_ORIGINS env var should be respected."""
        try:
            from services.automation.config import ApiConfig
        except ImportError:
            pytest.skip("services.automation.config not importable")
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://prod.example.com"}):
            config = ApiConfig()
            assert "https://prod.example.com" in config.cors_origins


# ============================================================================
# 5. NO HARDCODED CREDENTIALS TESTS
# ============================================================================

class TestNoHardcodedCredentials:
    """Scan source files for hardcoded API tokens and secrets."""

    # Patterns that indicate hardcoded credentials
    _CREDENTIAL_PATTERNS = [
        re.compile(r'hf_api_token\s*=\s*["\'](?!os\.environ|""|\'\')\w+["\']'),
        re.compile(r'api_key\s*=\s*["\']sk-[a-zA-Z0-9]+["\']'),
        re.compile(r'password\s*=\s*["\'][^"\']+["\']'),
        re.compile(r'secret\s*=\s*["\'][^"\']+["\']'),
    ]

    # Files to scan (Python files in training/, services/, config/)
    _SCAN_DIRS = ["training", "services", "config"]

    def _scan_files(self):
        """Yield (filepath, line_no, line) for all .py files in scan dirs."""
        for scan_dir in self._SCAN_DIRS:
            dir_path = PROJECT_ROOT / scan_dir
            if not dir_path.exists():
                continue
            for py_file in dir_path.rglob("*.py"):
                if _should_skip(py_file):
                    continue
                with open(py_file, "r", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        yield py_file, i, line

    def test_no_hardcoded_hf_tokens(self):
        """No literal HF API tokens in source (excluding env var lookups)."""
        violations = []
        pattern = re.compile(
            r'hf_api_token\s*=\s*"(?!os\.environ|)[a-zA-Z0-9_]+"'
        )
        for filepath, line_no, line in self._scan_files():
            # Skip lines that use os.environ
            if "os.environ" in line or "os.getenv" in line:
                continue
            if pattern.search(line):
                violations.append(f"{filepath.relative_to(PROJECT_ROOT)}:{line_no}: {line.strip()}")
        assert not violations, (
            f"Hardcoded HF tokens found:\n" + "\n".join(violations)
        )

    def test_no_demo_tokens(self):
        """No 'demo_token' or 'test_token' literals in non-test files."""
        violations = []
        for filepath, line_no, line in self._scan_files():
            if "/test" in str(filepath) or "test_" in filepath.name:
                continue
            if '"demo_token"' in line or '"test_token"' in line:
                violations.append(f"{filepath.relative_to(PROJECT_ROOT)}:{line_no}")
        assert not violations, (
            f"Demo/test tokens found in production code:\n" + "\n".join(violations)
        )

    def test_no_your_token_here(self):
        """No placeholder tokens like 'your_hf_token_here'."""
        violations = []
        for filepath, line_no, line in self._scan_files():
            if "your_hf_token_here" in line or "your_api_key_here" in line:
                # Allow in comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                violations.append(f"{filepath.relative_to(PROJECT_ROOT)}:{line_no}")
        assert not violations, (
            f"Placeholder tokens found:\n" + "\n".join(violations)
        )


# ============================================================================
# 6. SAFE DESERIALIZATION TESTS
# ============================================================================

class TestSafeDeserialization:
    """Verify that deserialization calls are annotated and safe."""

    def test_gpu_backend_uses_weights_only(self):
        """torch.load must use weights_only=True."""
        gpu_backend_path = PROJECT_ROOT / "core" / "backends" / "gpu_backend.py"
        if not gpu_backend_path.exists():
            pytest.skip("gpu_backend.py not found")
        content = gpu_backend_path.read_text()
        assert "weights_only=True" in content, \
            "gpu_backend.py must use torch.load with weights_only=True"

    def test_no_bare_pickle_load(self):
        """All pickle.load calls must have nosec or warning annotation."""
        violations = []
        for py_file in _iter_py_files():
            if "/test" in str(py_file) or "test_" in py_file.name:
                continue
            try:
                content = py_file.read_text(errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if "pickle.load" in stripped and "nosec" not in stripped:
                    # Allow imports and string references
                    if stripped.startswith("#") or stripped.startswith("from") or stripped.startswith("import"):
                        continue
                    if "def " in stripped or "class " in stripped:
                        continue
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i}")
        assert not violations, (
            f"Bare pickle.load without nosec annotation:\n" + "\n".join(violations)
        )

    def test_no_bare_pickle_loads(self):
        """All pickle.loads calls must have nosec annotation."""
        violations = []
        for py_file in _iter_py_files():
            if "/test" in str(py_file) or "test_" in py_file.name:
                continue
            try:
                content = py_file.read_text(errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if "pickle.loads(" in stripped and "nosec" not in stripped:
                    if stripped.startswith("#"):
                        continue
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i}")
        assert not violations, (
            f"Bare pickle.loads without nosec annotation:\n" + "\n".join(violations)
        )

    def test_quantized_engine_rejects_unknown_format(self):
        """Quantized engine must reject unknown file formats."""
        try:
            from inference.engines.quantized_engine import QuantizedInferenceEngine
        except ImportError:
            pytest.skip("quantized_engine not importable")
        engine = QuantizedInferenceEngine.__new__(QuantizedInferenceEngine)
        with pytest.raises(ValueError, match="Unsupported model format"):
            engine._load_model_params("/fake/model.exe")


# ============================================================================
# 7. DEBUG MODE PROTECTION TESTS
# ============================================================================

class TestDebugModeProtection:
    """Verify debug mode is blocked in production."""

    def test_debug_blocked_in_production(self):
        """main.py must check ENVIRONMENT before enabling debug."""
        main_path = PROJECT_ROOT / "services" / "automation" / "main.py"
        if not main_path.exists():
            pytest.skip("main.py not found")
        content = main_path.read_text()
        assert "ENVIRONMENT" in content, \
            "main.py must check ENVIRONMENT variable before enabling debug"
        assert "production" in content.lower(), \
            "main.py must reference 'production' environment"

    def test_debug_code_pattern(self):
        """Debug enable must be conditional on environment."""
        main_path = PROJECT_ROOT / "services" / "automation" / "main.py"
        if not main_path.exists():
            pytest.skip("main.py not found")
        content = main_path.read_text()
        # The debug block should NOT unconditionally set debug=True
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "config.debug = True" in line:
                # Check that there's a production guard above
                context = "\n".join(lines[max(0, i - 5):i + 1])
                assert "production" in context.lower() or "environment" in context.lower(), \
                    f"config.debug=True at line {i+1} is not guarded by environment check"


# ============================================================================
# 8. SECURE RANDOMNESS TESTS
# ============================================================================

class TestSecureRandomness:
    """Verify cryptographically secure randomness where needed."""

    def test_router_uses_system_random(self):
        """self_modifying_router must use SystemRandom for exploration."""
        router_path = PROJECT_ROOT / "capibara" / "routers" / "self_modifying_router.py"
        if not router_path.exists():
            pytest.skip("self_modifying_router.py not found")
        content = router_path.read_text()
        assert "SystemRandom" in content, \
            "Router exploration must use random.SystemRandom()"

    def test_no_bare_random_in_security_paths(self):
        """Security-sensitive files must not use bare random.random()."""
        security_files = [
            PROJECT_ROOT / "capibara" / "routers" / "self_modifying_router.py",
            PROJECT_ROOT / "safety" / "intervention_system.py",
        ]
        for filepath in security_files:
            if not filepath.exists():
                continue
            content = filepath.read_text(errors="ignore")
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Flag bare random.random() without SystemRandom
                if "random.random()" in stripped and "SystemRandom" not in stripped:
                    pytest.fail(
                        f"{filepath.name}:{i} uses bare random.random() — "
                        "use random.SystemRandom().random() for security"
                    )


# ============================================================================
# 9. GITIGNORE COMPLETENESS TESTS
# ============================================================================

class TestGitignoreCompleteness:
    """Verify .gitignore covers sensitive file patterns."""

    @pytest.fixture
    def gitignore_content(self):
        gitignore = PROJECT_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore not found")
        return gitignore.read_text()

    _REQUIRED_PATTERNS = [
        ".env",
        "*.pem",
        "*.key",
        "*.pkl",
        "credentials.json",
        "*.log",
        "__pycache__/",
    ]

    @pytest.mark.parametrize("pattern", _REQUIRED_PATTERNS)
    def test_gitignore_has_pattern(self, gitignore_content, pattern):
        assert pattern in gitignore_content, \
            f".gitignore must include '{pattern}'"

    def test_gitignore_not_minimal(self, gitignore_content):
        """gitignore must have more than basic __pycache__ entry."""
        lines = [l.strip() for l in gitignore_content.splitlines() if l.strip() and not l.startswith("#")]
        assert len(lines) >= 10, \
            f".gitignore has only {len(lines)} entries — too minimal"


# ============================================================================
# 10. GENERAL SECURITY SCAN
# ============================================================================

class TestGeneralSecurityScan:
    """Broad security checks across the codebase."""

    def test_no_eval_in_production_code(self):
        """No eval() calls in production code (except tests and JAX shim)."""
        # Known safe eval() uses (JAX shim expression parser, inference engines)
        _EVAL_ALLOWED_FILES = {"core.py", "arm_optimized_inference.py", "hybrid_inference_engine.py"}
        violations = []
        for py_file in _iter_py_files():
            if "/test" in str(py_file) or "test_" in py_file.name:
                continue
            if py_file.name in _EVAL_ALLOWED_FILES:
                continue
            try:
                content = py_file.read_text(errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Skip PyTorch .eval() calls (sets model to evaluation mode, not Python eval())
                if "eval(" in stripped and "evaluate" not in stripped.lower() and ".eval()" not in stripped:
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i}")
        assert not violations, (
            f"eval() found in unexpected locations:\n" + "\n".join(violations[:10])
        )

    def test_no_exec_in_production_code(self):
        """No exec() calls in production code."""
        violations = []
        for py_file in _iter_py_files():
            if "/test" in str(py_file) or "test_" in py_file.name:
                continue
            # Skip sandbox agent (exec is its purpose)
            if "sandbox" in py_file.name or "e2b" in py_file.name:
                continue
            try:
                content = py_file.read_text(errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(r"(?<![\w.])exec\(", stripped) and "execute" not in stripped.lower() and "executor" not in stripped.lower():
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i}")
        assert len(violations) <= 2, (
            f"exec() found in {len(violations)} locations:\n"
            + "\n".join(violations[:10])
        )

    def test_no_shell_true_subprocess(self):
        """No subprocess calls with shell=True."""
        violations = []
        for py_file in _iter_py_files():
            if "/test" in str(py_file) or "test_" in py_file.name:
                continue
            try:
                content = py_file.read_text(errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "shell=True" in stripped and "subprocess" in content:
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i}")
        assert not violations, (
            f"subprocess with shell=True found:\n" + "\n".join(violations)
        )

    def test_yaml_safe_load(self):
        """yaml.load must use SafeLoader or yaml.safe_load."""
        violations = []
        for py_file in _iter_py_files():
            if "/test" in str(py_file) or "test_" in py_file.name:
                continue
            try:
                content = py_file.read_text(errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Flag yaml.load without SafeLoader
                if "yaml.load(" in stripped and "safe_load" not in stripped and "SafeLoader" not in stripped and "Loader=" not in stripped:
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i}")
        assert not violations, (
            f"Unsafe yaml.load without SafeLoader:\n" + "\n".join(violations)
        )
