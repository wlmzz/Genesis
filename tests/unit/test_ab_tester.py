"""
Unit tests for A/B testing framework (mlops/ab_testing/ab_tester.py)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
from mlops.ab_testing.ab_tester import ABTester, Variant, ABExperiment


class TestABTester:
    """Test ABTester class"""

    @pytest.fixture
    def control_predictor(self):
        """Mock control model predictor"""
        return lambda x: "control_prediction"

    @pytest.fixture
    def treatment_predictor(self):
        """Mock treatment model predictor"""
        return lambda x: "treatment_prediction"

    def test_create_experiment(self, control_predictor, treatment_predictor):
        """Test creating A/B experiment"""
        tester = ABTester()

        experiment = tester.create_experiment(
            experiment_id="test-exp",
            model_name="test-model",
            control_version="1",
            control_predictor=control_predictor,
            treatment_version="2",
            treatment_predictor=treatment_predictor,
            traffic_split=0.3,
            duration_hours=24
        )

        assert experiment.experiment_id == "test-exp"
        assert experiment.model_name == "test-model"
        assert len(experiment.variants) == 2
        assert experiment.variants[0].traffic_percentage == 0.7  # Control: 70%
        assert experiment.variants[1].traffic_percentage == 0.3  # Treatment: 30%

    def test_select_variant(self, control_predictor, treatment_predictor):
        """Test variant selection (traffic splitting)"""
        tester = ABTester()

        experiment = tester.create_experiment(
            experiment_id="test-exp",
            model_name="test-model",
            control_version="1",
            control_predictor=control_predictor,
            treatment_version="2",
            treatment_predictor=treatment_predictor,
            traffic_split=0.5
        )

        # Run many selections and check distribution
        control_count = 0
        treatment_count = 0

        for _ in range(1000):
            variant = tester.select_variant("test-exp")
            if variant.name == "control":
                control_count += 1
            else:
                treatment_count += 1

        # Should be roughly 50/50 split (with tolerance)
        assert 400 < control_count < 600
        assert 400 < treatment_count < 600

    @pytest.mark.asyncio
    async def test_predict(self, control_predictor, treatment_predictor):
        """Test making prediction with variant selection"""
        tester = ABTester()

        experiment = tester.create_experiment(
            experiment_id="test-exp",
            model_name="test-model",
            control_version="1",
            control_predictor=control_predictor,
            treatment_version="2",
            treatment_predictor=treatment_predictor,
            traffic_split=0.5
        )

        result = await tester.predict(
            experiment_id="test-exp",
            input_data="test_input"
        )

        assert "prediction" in result
        assert "variant" in result
        assert result["variant"] in ["control", "treatment"]

    def test_get_experiment_stats(self, control_predictor, treatment_predictor):
        """Test getting experiment statistics"""
        tester = ABTester()

        experiment = tester.create_experiment(
            experiment_id="test-exp",
            model_name="test-model",
            control_version="1",
            control_predictor=control_predictor,
            treatment_version="2",
            treatment_predictor=treatment_predictor
        )

        # Add some metrics
        tester._record_metric("test-exp", "control", "accuracy", 0.85)
        tester._record_metric("test-exp", "control", "accuracy", 0.87)
        tester._record_metric("test-exp", "treatment", "accuracy", 0.90)
        tester._record_metric("test-exp", "treatment", "accuracy", 0.92)

        stats = tester.get_experiment_stats("test-exp")

        assert stats["experiment_id"] == "test-exp"
        assert len(stats["variants"]) == 2

        # Check control metrics
        control_stats = stats["variants"][0]
        assert "metrics" in control_stats
        assert "accuracy" in control_stats["metrics"]
        assert control_stats["metrics"]["accuracy"]["count"] == 2

    def test_statistical_test(self, control_predictor, treatment_predictor):
        """Test statistical significance testing"""
        tester = ABTester()

        experiment = tester.create_experiment(
            experiment_id="test-exp",
            model_name="test-model",
            control_version="1",
            control_predictor=control_predictor,
            treatment_version="2",
            treatment_predictor=treatment_predictor,
            min_samples=10
        )

        # Add enough samples for statistical test
        for _ in range(100):
            tester._record_metric("test-exp", "control", "accuracy", 0.85)
            tester._record_metric("test-exp", "treatment", "accuracy", 0.92)

        stats = tester.get_experiment_stats("test-exp")
        test_result = stats.get("statistical_test", {})

        assert test_result.get("status") == "completed"
        assert "p_value" in test_result
        assert "winner" in test_result

    def test_conclude_experiment(self, control_predictor, treatment_predictor):
        """Test concluding experiment"""
        tester = ABTester()

        experiment = tester.create_experiment(
            experiment_id="test-exp",
            model_name="test-model",
            control_version="1",
            control_predictor=control_predictor,
            treatment_version="2",
            treatment_predictor=treatment_predictor,
            min_samples=10
        )

        # Add metrics
        for _ in range(100):
            tester._record_metric("test-exp", "control", "accuracy", 0.80)
            tester._record_metric("test-exp", "treatment", "accuracy", 0.90)

        conclusion = tester.conclude_experiment("test-exp")

        assert "winner" in conclusion
        assert conclusion["status"] == "completed"
