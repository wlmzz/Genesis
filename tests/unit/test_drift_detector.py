"""
Unit tests for drift detection (mlops/drift_detection/drift_detector.py)
"""

import pytest
import numpy as np
from mlops.drift_detection.drift_detector import DriftDetector


class TestDriftDetector:
    """Test DriftDetector class"""

    @pytest.fixture
    def detector(self):
        """Create drift detector instance"""
        return DriftDetector(
            model_name="test-model",
            window_size=100,
            drift_threshold=0.05,
            performance_threshold=0.1
        )

    def test_initialization(self, detector):
        """Test DriftDetector initialization"""
        assert detector.model_name == "test-model"
        assert detector.window_size == 100
        assert detector.drift_threshold == 0.05
        assert detector.performance_threshold == 0.1

    def test_add_baseline_sample(self, detector):
        """Test adding baseline samples"""
        detector.add_baseline_sample(
            prediction=1,
            ground_truth=1,
            confidence=0.95
        )

        assert len(detector.baseline.predictions) == 1
        assert len(detector.baseline.ground_truths) == 1
        assert len(detector.baseline.confidences) == 1

    def test_add_current_sample(self, detector):
        """Test adding current samples"""
        detector.add_current_sample(
            prediction=1,
            ground_truth=1,
            confidence=0.90,
            features={"bbox_size": 100}
        )

        assert len(detector.current.predictions) == 1
        assert "bbox_size" in detector.current.feature_distributions

    def test_calculate_baseline_metrics(self, detector):
        """Test calculating baseline metrics"""
        # Add samples
        for i in range(200):
            pred = 1 if i < 180 else 0  # 90% accuracy
            detector.add_baseline_sample(
                prediction=pred,
                ground_truth=1,
                confidence=0.95
            )

        metrics = detector.calculate_baseline_metrics()

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0.85 < metrics["accuracy"] < 0.95

    def test_detect_performance_drift_no_drift(self, detector):
        """Test performance drift detection - no drift"""
        # Baseline: 90% accuracy
        for i in range(200):
            pred = 1 if i < 180 else 0
            detector.add_baseline_sample(pred, 1, 0.95)

        detector.calculate_baseline_metrics()

        # Current: 88% accuracy (within threshold)
        for i in range(200):
            pred = 1 if i < 176 else 0
            detector.add_current_sample(pred, 1, 0.93)

        result = detector.detect_performance_drift()

        assert result["drift_detected"] is False

    def test_detect_performance_drift_with_drift(self, detector):
        """Test performance drift detection - drift detected"""
        # Baseline: 90% accuracy
        for i in range(200):
            pred = 1 if i < 180 else 0
            detector.add_baseline_sample(pred, 1, 0.95)

        detector.calculate_baseline_metrics()

        # Current: 75% accuracy (>10% drop)
        for i in range(200):
            pred = 1 if i < 150 else 0
            detector.add_current_sample(pred, 1, 0.80)

        result = detector.detect_performance_drift()

        assert result["drift_detected"] is True
        assert result["drops"]["accuracy"] > 0.10

    def test_detect_data_drift(self, detector):
        """Test data drift detection using KS test"""
        # Baseline distribution
        for i in range(200):
            detector.add_baseline_sample(
                prediction=1,
                ground_truth=1,
                confidence=0.95,
                features={"bbox_size": float(np.random.normal(100, 10))}
            )

        # Current distribution (shifted)
        for i in range(200):
            detector.add_current_sample(
                prediction=1,
                ground_truth=1,
                confidence=0.95,
                features={"bbox_size": float(np.random.normal(150, 10))}  # Mean shifted
            )

        result = detector.detect_data_drift()

        assert "features" in result
        if "bbox_size" in result["features"]:
            # Should detect drift due to distribution shift
            assert result["features"]["bbox_size"]["ks_statistic"] > 0

    def test_detect_concept_drift(self, detector):
        """Test concept drift detection"""
        # Baseline: high confidence
        for i in range(200):
            detector.add_baseline_sample(1, 1, float(np.random.normal(0.95, 0.02)))

        # Current: lower confidence (concept drift)
        for i in range(200):
            detector.add_current_sample(1, 1, float(np.random.normal(0.75, 0.05)))

        result = detector.detect_concept_drift()

        # Should detect confidence distribution shift
        assert "ks_statistic" in result
        assert result["ks_statistic"] > 0

    def test_detect_all_drifts(self, detector):
        """Test detecting all drift types"""
        # Add baseline
        for i in range(200):
            detector.add_baseline_sample(1, 1, 0.95, features={"size": 100.0})

        detector.calculate_baseline_metrics()

        # Add current (with drift)
        for i in range(200):
            pred = 1 if i < 150 else 0  # Performance drift
            detector.add_current_sample(pred, 1, 0.75, features={"size": 150.0})

        results = detector.detect_all_drifts()

        assert "performance_drift" in results
        assert "data_drift" in results
        assert "concept_drift" in results
        assert "drift_detected" in results
