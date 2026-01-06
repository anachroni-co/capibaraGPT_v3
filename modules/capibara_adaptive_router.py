"""
Capibara Adaptive Router con HRMService y experto especial por consenso

- HRMService: procesamiento simple de señal PPG/RR para extraer métricas (HR, HRV)
- CapibaraAdaptiveRouter: selección de experto basada en señales HRM
- Integración con HuggingFaceConsensusConfig para experto especial (medical)
"""

import os
import sys
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import toml  # type: ignore
except Exception:  # pragma: no cover
    toml = None  # type: ignore

try:
    from capibara.training.huggingface_consensus_config import (
        HuggingFaceConsensusConfig,
    )
except Exception:
    HuggingFaceConsensusConfig = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class HRMConfig:
    enabled: bool = True
    sampling_rate_hz: int = 100
    window_seconds: int = 10
    expected_hr_min: int = 40
    expected_hr_max: int = 180
    anomaly_rmssd_threshold_ms: float = 80.0
    anomaly_hr_jump_bpm: float = 30.0


class HRMService:
    """Servicio simple para extraer métricas de una señal HRM (PPG o RR).

    Entrada esperada:
    - Señal PPG (onda) en np.ndarray (float) o
    - Serie de intervalos RR en ms (si rr_intervals_ms=True)
    """

    def __init__(self, config: Optional[HRMConfig] = None):
        self.config = config or HRMConfig()

    def _detect_rr_intervals_from_ppg(self, signal: np.ndarray) -> np.ndarray:
        """Estimación muy basic of intervalos RR a partir de PPG por autocorrelación.
        Retorna RR en ms aprox.
        """
        if signal.ndim != 1:
            signal = signal.reshape(-1)
        signal = signal.astype(np.float32)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        corr = np.correlate(signal, signal, mode="full")
        corr = corr[corr.size // 2 :]
        # Ignorar el primer pico (lag 0). Buscar pico principal en [0.3s, 2.0s]
        min_lag = int(0.3 * self.config.sampling_rate_hz)
        max_lag = int(2.0 * self.config.sampling_rate_hz)
        if max_lag <= min_lag:
            max_lag = min_lag + 1
        roi = corr[min_lag:max_lag]
        if roi.size == 0:
            return np.array([], dtype=np.float32)
        peak_lag = np.argmax(roi) + min_lag
        # Convertir periodo a RR (ms)
        rr_ms = 1000.0 * float(peak_lag) / float(self.config.sampling_rate_hz)
        # Construir una secuencia de RR constante como estimación
        num_beats = max(1, int(self.config.window_seconds * 60.0 / max(1.0, 60000.0 / rr_ms)))
        return np.full((num_beats,), rr_ms, dtype=np.float32)

    def _compute_rmssd_ms(self, rr_ms: np.ndarray) -> float:
        if rr_ms.size < 2:
            return 0.0
        diff = np.diff(rr_ms)
        return float(np.sqrt(np.mean(diff * diff)))

    def _compute_hr_bpm(self, rr_ms: np.ndarray) -> float:
        if rr_ms.size == 0:
            return 0.0
        mean_rr_ms = float(np.mean(rr_ms))
        return 60000.0 / max(1e-6, mean_rr_ms)

    def analyze(
        self,
        data: np.ndarray,
        rr_intervals_ms: bool = False,
    ) -> Dict[str, Any]:
        """Returns basic HRM metrics from PPG o RR.

        Returns: {hr_bpm, rmssd_ms, anomaly_score, anomalies: {hr_out_of_range, hr_jump, high_rmssd}}
        """
        if data is None:
            return {"hr_bpm": 0.0, "rmssd_ms": 0.0, "anomaly_score": 0.0, "anomalies": {}}

        rr_ms = data.astype(np.float32)
        if not rr_intervals_ms:
            rr_ms = self._detect_rr_intervals_from_ppg(rr_ms)

        hr_bpm = self._compute_hr_bpm(rr_ms)
        rmssd_ms = self._compute_rmssd_ms(rr_ms)

        anomalies = {
            "hr_out_of_range": not (self.config.expected_hr_min <= hr_bpm <= self.config.expected_hr_max),
            "high_rmssd": rmssd_ms >= self.config.anomaly_rmssd_threshold_ms,
            # Note: hr_jump requires history; here we approximate with 0
            "hr_jump": False,
        }
        anomaly_score = sum(float(v) for v in anomalies.values()) / 3.0

        return {
            "hr_bpm": hr_bpm,
            "rmssd_ms": rmssd_ms,
            "anomaly_score": anomaly_score,
            "anomalies": anomalies,
        }


class CapibaraAdaptiveRouter:
    """Router adaptativo que utiliza señales HRM para seleccionar experto especial."""

    def __init__(self, config_path: Optional[str] = None):
        self.router_config = self._load_router_config(config_path)
        self.hrm_service = HRMService(self._extract_hrm_config(self.router_config))
        self.consensus_config = self._load_consensus_config()

    def _load_router_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path and os.path.exists(config_path) and toml:
            try:
                return toml.load(config_path)
            except Exception as e:  # pragma: no cover
                logger.warning(f"No se pudo cargar TOML {config_path}: {e}")
        # Fallback a archivo por defecto si existe
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "configs_toml",
            "production",
            "unified_router.toml",
        )
        if toml and os.path.exists(default_path):
            try:
                return toml.load(default_path)
            except Exception as e:  # pragma: no cover
                logger.warning(f"No se pudo cargar TOML por defecto: {e}")
        return {}

    def _extract_hrm_config(self, cfg: Dict[str, Any]) -> HRMConfig:
        hrm_dict = (cfg.get("hrm", {}) if isinstance(cfg, dict) else {}) or {}
        return HRMConfig(
            enabled=bool(hrm_dict.get("enabled", True)),
            sampling_rate_hz=int(hrm_dict.get("sampling_rate_hz", 100)),
            window_seconds=int(hrm_dict.get("window_seconds", 10)),
            expected_hr_min=int(hrm_dict.get("expected_hr_min", 40)),
            expected_hr_max=int(hrm_dict.get("expected_hr_max", 180)),
            anomaly_rmssd_threshold_ms=float(hrm_dict.get("anomaly_rmssd_threshold_ms", 80.0)),
            anomaly_hr_jump_bpm=float(hrm_dict.get("anomaly_hr_jump_bpm", 30.0)),
        )

    def _load_consensus_config(self):
        if HuggingFaceConsensusConfig is None:
            return None
        try:
            return HuggingFaceConsensusConfig()
        except Exception:  # pragma: no cover
            return None

    def route(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Returns routing decision considering HRM como experto especial.

        Contexto puede incluir:
        - ppg_signal: np.ndarray de señal PPG
        - rr_intervals_ms: np.ndarray de RR en ms
        """
        context = context or {}

        specialist = None
        hrm_metrics: Optional[Dict[str, Any]] = None

        if self.hrm_service and self.hrm_service.config.enabled:
            if "ppg_signal" in context and isinstance(context["ppg_signal"], np.ndarray):
                hrm_metrics = self.hrm_service.analyze(context["ppg_signal"], rr_intervals_ms=False)
                specialist = "hrm"
            elif "rr_intervals_ms" in context and isinstance(context["rr_intervals_ms"], np.ndarray):
                hrm_metrics = self.hrm_service.analyze(context["rr_intervals_ms"], rr_intervals_ms=True)
                specialist = "hrm"

        # Simple expert selection based on HRM
        selected_expert = "general"
        confidence = 0.6
        if hrm_metrics is not None:
            selected_expert = "medical"
            confidence = 0.8 if hrm_metrics["anomaly_score"] >= 0.34 else 0.7

        consensus_weights: Dict[str, float] = {}
        if self.consensus_config is not None:
            # Aumentar peso de experto médico si HRM presente
            expert_models = self.consensus_config.expert_models
            if selected_expert in expert_models:
                base_weight = float(expert_models[selected_expert].get("weight", 1.0))
                consensus_weights[selected_expert] = base_weight * (1.3 if specialist == "hrm" else 1.0)
            # Añadir razonamiento como apoyo
            if "reasoning" in expert_models:
                consensus_weights["reasoning"] = float(expert_models["reasoning"].get("weight", 1.0))

        return {
            "expert": selected_expert,
            "specialist": specialist,
            "confidence": confidence,
            "hrm_metrics": hrm_metrics,
            "consensus_weights": consensus_weights,
        }


def main() -> bool:
    logger.info("CapibaraAdaptiveRouter arrancando")
    router = CapibaraAdaptiveRouter()
    # Minimal example with synthetic signal
    t = np.linspace(0, 5.0, 500)
    ppg = 0.6 * np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 2.4 * t)
    decision = router.route("analyze patient status", {"ppg_signal": ppg.astype(np.float32)})
    logger.info(f"Routing decision: {decision}")
    return True


if __name__ == "__main__":
    main()
