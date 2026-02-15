"""
Model Drift Detection for Genesis Platform

Detects concept drift, data drift, and performance degradation.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score
from prometheus_client import Gauge, Counter
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# Prometheus metrics
drift_score = Gauge(
    "genesis_drift_score",
    "Model drift score",
    ["model_name", "drift_type"]
)

drift_detected = Counter(
    "genesis_drift_detected_total",
    "Total drift detections",
    ["model_name", "drift_type"]
)


@dataclass
class DriftWindow:
    """Sliding window for drift detection"""
    predictions: deque = field(default_factory=lambda: deque(maxlen=1000))
    ground_truths: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidences: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    feature_distributions: Dict[str, deque] = field(default_factory=dict)


class DriftDetector:
    """
    Detects model drift using multiple methods.

    Drift Types:
    - Concept Drift: Change in P(Y|X) - relationship between features and target
    - Data Drift: Change in P(X) - distribution of input features
    - Performance Drift: Degradation in model metrics over time
    """

    def __init__(
        self,
        model_name: str,
        window_size: int = 1000,
        baseline_window_size: int = 5000,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.1  # 10% drop triggers alert
    ):
        """
        Initialize drift detector.

        Args:
            model_name: Model name to monitor
            window_size: Recent window size for drift comparison
            baseline_window_size: Baseline window size (training data distribution)
            drift_threshold: Statistical significance threshold (p-value)
            performance_threshold: Performance drop threshold (0.1 = 10%)
        """
        self.model_name = model_name
        self.window_size = window_size
        self.baseline_window_size = baseline_window_size
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold

        # Windows
        self.baseline = DriftWindow()
        self.current = DriftWindow()

        # Metrics tracking
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.recent_metrics: List[Dict[str, float]] = []

        logger.info(
            f"Drift detector initialized for {model_name} "
            f"(window={window_size}, threshold={drift_threshold})"
        )

    def add_baseline_sample(
        self,
        prediction: Any,
        ground_truth: Any,
        confidence: float,
        features: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add sample to baseline distribution (training data).

        Args:
            prediction: Model prediction
            ground_truth: True label
            confidence: Prediction confidence
            features: Optional feature values
        """
        self.baseline.predictions.append(prediction)
        self.baseline.ground_truths.append(ground_truth)
        self.baseline.confidences.append(confidence)
        self.baseline.timestamps.append(datetime.now())

        if features:
            for feature_name, value in features.items():
                if feature_name not in self.baseline.feature_distributions:
                    self.baseline.feature_distributions[feature_name] = deque(maxlen=self.baseline_window_size)
                self.baseline.feature_distributions[feature_name].append(value)

    def add_current_sample(
        self,
        prediction: Any,
        ground_truth: Any,
        confidence: float,
        features: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add sample to current window (production data).

        Args:
            prediction: Model prediction
            ground_truth: True label
            confidence: Prediction confidence
            features: Optional feature values
        """
        self.current.predictions.append(prediction)
        self.current.ground_truths.append(ground_truth)
        self.current.confidences.append(confidence)
        self.current.timestamps.append(datetime.now())

        if features:
            for feature_name, value in features.items():
                if feature_name not in self.current.feature_distributions:
                    self.current.feature_distributions[feature_name] = deque(maxlen=self.window_size)
                self.current.feature_distributions[feature_name].append(value)

    def calculate_baseline_metrics(self) -> Dict[str, float]:
        """
        Calculate baseline performance metrics.

        Returns:
            Dictionary of metrics
        """
        if len(self.baseline.predictions) < 100:
            logger.warning(f"Insufficient baseline samples: {len(self.baseline.predictions)}")
            return {}

        predictions = np.array(list(self.baseline.predictions))
        ground_truths = np.array(list(self.baseline.ground_truths))
        confidences = np.array(list(self.baseline.confidences))

        metrics = {
            "accuracy": accuracy_score(ground_truths, predictions),
            "precision": precision_score(ground_truths, predictions, average='weighted', zero_division=0),
            "recall": recall_score(ground_truths, predictions, average='weighted', zero_division=0),
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences))
        }

        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics calculated: accuracy={metrics['accuracy']:.3f}")

        return metrics

    def detect_performance_drift(self) -> Dict[str, Any]:
        """
        Detect performance degradation.

        Returns:
            Detection results
        """
        if not self.baseline_metrics or len(self.current.predictions) < 100:
            return {
                "drift_detected": False,
                "reason": "insufficient_data"
            }

        # Calculate current metrics
        predictions = np.array(list(self.current.predictions))
        ground_truths = np.array(list(self.current.ground_truths))
        confidences = np.array(list(self.current.confidences))

        current_metrics = {
            "accuracy": accuracy_score(ground_truths, predictions),
            "precision": precision_score(ground_truths, predictions, average='weighted', zero_division=0),
            "recall": recall_score(ground_truths, predictions, average='weighted', zero_division=0),
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences))
        }

        # Compare with baseline
        accuracy_drop = self.baseline_metrics["accuracy"] - current_metrics["accuracy"]
        precision_drop = self.baseline_metrics["precision"] - current_metrics["precision"]
        recall_drop = self.baseline_metrics["recall"] - current_metrics["recall"]

        drift_detected = (
            accuracy_drop > self.performance_threshold or
            precision_drop > self.performance_threshold or
            recall_drop > self.performance_threshold
        )

        if drift_detected:
            drift_detected_counter = drift_detected.labels(
                model_name=self.model_name,
                drift_type="performance"
            )
            drift_detected_counter.inc()

            logger.warning(
                f"Performance drift detected for {self.model_name}: "
                f"accuracy_drop={accuracy_drop:.3f}, "
                f"precision_drop={precision_drop:.3f}, "
                f"recall_drop={recall_drop:.3f}"
            )

        # Update Prometheus
        drift_score.labels(
            model_name=self.model_name,
            drift_type="performance"
        ).set(accuracy_drop)

        return {
            "drift_detected": drift_detected,
            "drift_type": "performance",
            "baseline_metrics": self.baseline_metrics,
            "current_metrics": current_metrics,
            "drops": {
                "accuracy": float(accuracy_drop),
                "precision": float(precision_drop),
                "recall": float(recall_drop)
            },
            "threshold": self.performance_threshold
        }

    def detect_data_drift(self) -> Dict[str, Any]:
        """
        Detect data drift using Kolmogorov-Smirnov test.

        Returns:
            Detection results
        """
        if not self.baseline.feature_distributions or not self.current.feature_distributions:
            return {
                "drift_detected": False,
                "reason": "no_feature_distributions"
            }

        drift_results = {}
        overall_drift = False

        for feature_name in self.baseline.feature_distributions.keys():
            if feature_name not in self.current.feature_distributions:
                continue

            baseline_dist = np.array(list(self.baseline.feature_distributions[feature_name]))
            current_dist = np.array(list(self.current.feature_distributions[feature_name]))

            if len(baseline_dist) < 30 or len(current_dist) < 30:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(baseline_dist, current_dist)

            feature_drift = p_value < self.drift_threshold

            drift_results[feature_name] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": feature_drift
            }

            if feature_drift:
                overall_drift = True
                logger.warning(
                    f"Data drift detected in feature '{feature_name}': "
                    f"KS={ks_stat:.3f}, p={p_value:.4f}"
                )

        if overall_drift:
            drift_detected_counter = drift_detected.labels(
                model_name=self.model_name,
                drift_type="data"
            )
            drift_detected_counter.inc()

        # Update Prometheus with max KS statistic
        if drift_results:
            max_ks = max(r["ks_statistic"] for r in drift_results.values())
            drift_score.labels(
                model_name=self.model_name,
                drift_type="data"
            ).set(max_ks)

        return {
            "drift_detected": overall_drift,
            "drift_type": "data",
            "features": drift_results,
            "threshold": self.drift_threshold
        }

    def detect_concept_drift(self) -> Dict[str, Any]:
        """
        Detect concept drift using confidence distribution shift.

        Returns:
            Detection results
        """
        if len(self.baseline.confidences) < 100 or len(self.current.confidences) < 100:
            return {
                "drift_detected": False,
                "reason": "insufficient_data"
            }

        baseline_conf = np.array(list(self.baseline.confidences))
        current_conf = np.array(list(self.current.confidences))

        # KS test on confidence distributions
        ks_stat, p_value = stats.ks_2samp(baseline_conf, current_conf)

        drift_detected = p_value < self.drift_threshold

        if drift_detected:
            drift_detected_counter = drift_detected.labels(
                model_name=self.model_name,
                drift_type="concept"
            )
            drift_detected_counter.inc()

            logger.warning(
                f"Concept drift detected for {self.model_name}: "
                f"confidence distribution changed (KS={ks_stat:.3f}, p={p_value:.4f})"
            )

        # Update Prometheus
        drift_score.labels(
            model_name=self.model_name,
            drift_type="concept"
        ).set(ks_stat)

        return {
            "drift_detected": drift_detected,
            "drift_type": "concept",
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "baseline_confidence": {
                "mean": float(np.mean(baseline_conf)),
                "std": float(np.std(baseline_conf))
            },
            "current_confidence": {
                "mean": float(np.mean(current_conf)),
                "std": float(np.std(current_conf))
            },
            "threshold": self.drift_threshold
        }

    def detect_all_drifts(self) -> Dict[str, Any]:
        """
        Run all drift detection methods.

        Returns:
            Combined detection results
        """
        results = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "performance_drift": self.detect_performance_drift(),
            "data_drift": self.detect_data_drift(),
            "concept_drift": self.detect_concept_drift()
        }

        # Overall drift flag
        results["drift_detected"] = (
            results["performance_drift"].get("drift_detected", False) or
            results["data_drift"].get("drift_detected", False) or
            results["concept_drift"].get("drift_detected", False)
        )

        if results["drift_detected"]:
            logger.warning(
                f"Drift detected for {self.model_name}! "
                f"Performance: {results['performance_drift']['drift_detected']}, "
                f"Data: {results['data_drift']['drift_detected']}, "
                f"Concept: {results['concept_drift']['drift_detected']}"
            )

        return results

    def reset_current_window(self) -> None:
        """Reset current window after drift handling"""
        self.current = DriftWindow()
        logger.info(f"Reset current window for {self.model_name}")
