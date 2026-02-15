"""
A/B Testing Framework for Model Comparison

Enables safe deployment of new model versions with statistical significance testing.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import numpy as np
from scipy import stats
from prometheus_client import Counter, Histogram, Gauge
import json

logger = logging.getLogger(__name__)


# Prometheus metrics
ab_test_requests = Counter(
    "genesis_ab_test_requests_total",
    "Total A/B test requests",
    ["experiment_id", "variant"]
)

ab_test_latency = Histogram(
    "genesis_ab_test_latency_seconds",
    "A/B test prediction latency",
    ["experiment_id", "variant"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

ab_test_metric = Gauge(
    "genesis_ab_test_metric",
    "A/B test metric value",
    ["experiment_id", "variant", "metric_name"]
)


@dataclass
class Variant:
    """Model variant in A/B test"""
    name: str
    model_version: str
    traffic_percentage: float  # 0.0 to 1.0
    predictor: Callable  # Function that makes predictions
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    request_count: int = 0
    error_count: int = 0


@dataclass
class ABExperiment:
    """A/B Testing Experiment"""
    experiment_id: str
    model_name: str
    variants: List[Variant]
    start_time: datetime
    duration_hours: int
    min_samples: int = 100  # Minimum samples before statistical test
    confidence_level: float = 0.95  # Statistical significance threshold
    primary_metric: str = "accuracy"  # Metric to optimize
    status: str = "running"  # running, paused, completed
    winner: Optional[str] = None


class ABTester:
    """
    A/B Testing framework for ML models.

    Features:
    - Traffic splitting
    - Statistical significance testing
    - Automatic winner selection
    - Metrics tracking
    - Canary deployments
    """

    def __init__(self):
        self.experiments: Dict[str, ABExperiment] = {}
        logger.info("A/B Tester initialized")

    def create_experiment(
        self,
        experiment_id: str,
        model_name: str,
        control_version: str,
        control_predictor: Callable,
        treatment_version: str,
        treatment_predictor: Callable,
        traffic_split: float = 0.5,  # % of traffic to treatment (0.0 - 1.0)
        duration_hours: int = 24,
        min_samples: int = 100,
        primary_metric: str = "accuracy"
    ) -> ABExperiment:
        """
        Create a new A/B test experiment.

        Args:
            experiment_id: Unique experiment identifier
            model_name: Model name being tested
            control_version: Current production model version
            control_predictor: Function for control model predictions
            treatment_version: New model version to test
            treatment_predictor: Function for treatment model predictions
            traffic_split: Percentage of traffic to treatment (0.0-1.0)
            duration_hours: Experiment duration in hours
            min_samples: Minimum samples before statistical test
            primary_metric: Metric to optimize

        Returns:
            ABExperiment object
        """
        control = Variant(
            name="control",
            model_version=control_version,
            traffic_percentage=1.0 - traffic_split,
            predictor=control_predictor
        )

        treatment = Variant(
            name="treatment",
            model_version=treatment_version,
            traffic_percentage=traffic_split,
            predictor=treatment_predictor
        )

        experiment = ABExperiment(
            experiment_id=experiment_id,
            model_name=model_name,
            variants=[control, treatment],
            start_time=datetime.now(),
            duration_hours=duration_hours,
            min_samples=min_samples,
            confidence_level=0.95,
            primary_metric=primary_metric
        )

        self.experiments[experiment_id] = experiment

        logger.info(
            f"Created A/B experiment {experiment_id}: "
            f"{control_version} (control) vs {treatment_version} (treatment), "
            f"traffic split: {(1-traffic_split)*100:.0f}% / {traffic_split*100:.0f}%"
        )

        return experiment

    def select_variant(self, experiment_id: str) -> Variant:
        """
        Select variant based on traffic split.

        Args:
            experiment_id: Experiment ID

        Returns:
            Selected variant
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Random selection based on traffic percentage
        rand = random.random()
        cumulative = 0.0

        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if rand <= cumulative:
                return variant

        # Fallback to control
        return experiment.variants[0]

    async def predict(
        self,
        experiment_id: str,
        input_data: Any,
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using selected variant.

        Args:
            experiment_id: Experiment ID
            input_data: Input for prediction
            ground_truth: Optional ground truth for metric calculation

        Returns:
            Dictionary with prediction and metadata
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment or experiment.status != "running":
            raise ValueError(f"Experiment {experiment_id} not running")

        # Select variant
        variant = self.select_variant(experiment_id)

        # Track request
        ab_test_requests.labels(
            experiment_id=experiment_id,
            variant=variant.name
        ).inc()

        # Make prediction with timing
        start_time = datetime.now()
        try:
            prediction = variant.predictor(input_data)
            latency = (datetime.now() - start_time).total_seconds()

            # Track latency
            ab_test_latency.labels(
                experiment_id=experiment_id,
                variant=variant.name
            ).observe(latency)

            variant.request_count += 1

            # Calculate metrics if ground truth provided
            if ground_truth is not None:
                accuracy = self._calculate_accuracy(prediction, ground_truth)
                self._record_metric(experiment_id, variant.name, "accuracy", accuracy)

            return {
                "prediction": prediction,
                "variant": variant.name,
                "model_version": variant.model_version,
                "latency": latency,
                "experiment_id": experiment_id
            }

        except Exception as e:
            logger.error(f"Prediction failed for variant {variant.name}: {e}")
            variant.error_count += 1
            raise

    def _calculate_accuracy(self, prediction: Any, ground_truth: Any) -> float:
        """Calculate prediction accuracy (simplified)"""
        # Implement based on your prediction format
        # This is a placeholder
        return 1.0 if prediction == ground_truth else 0.0

    def _record_metric(
        self,
        experiment_id: str,
        variant_name: str,
        metric_name: str,
        value: float
    ) -> None:
        """Record metric for variant"""
        experiment = self.experiments[experiment_id]

        for variant in experiment.variants:
            if variant.name == variant_name:
                if metric_name not in variant.metrics:
                    variant.metrics[metric_name] = []
                variant.metrics[metric_name].append(value)

                # Update Prometheus gauge
                ab_test_metric.labels(
                    experiment_id=experiment_id,
                    variant=variant_name,
                    metric_name=metric_name
                ).set(value)
                break

    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment statistics.

        Args:
            experiment_id: Experiment ID

        Returns:
            Statistics dictionary
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        stats = {
            "experiment_id": experiment_id,
            "model_name": experiment.model_name,
            "status": experiment.status,
            "start_time": experiment.start_time.isoformat(),
            "duration_hours": experiment.duration_hours,
            "elapsed_hours": (datetime.now() - experiment.start_time).total_seconds() / 3600,
            "variants": []
        }

        for variant in experiment.variants:
            variant_stats = {
                "name": variant.name,
                "model_version": variant.model_version,
                "traffic_percentage": variant.traffic_percentage,
                "request_count": variant.request_count,
                "error_count": variant.error_count,
                "error_rate": variant.error_count / max(variant.request_count, 1),
                "metrics": {}
            }

            # Calculate metric statistics
            for metric_name, values in variant.metrics.items():
                if values:
                    variant_stats["metrics"][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    }

            stats["variants"].append(variant_stats)

        # Statistical significance test
        if len(experiment.variants) == 2:
            stats["statistical_test"] = self._perform_statistical_test(experiment)

        return stats

    def _perform_statistical_test(self, experiment: ABExperiment) -> Dict[str, Any]:
        """
        Perform statistical significance test (t-test).

        Args:
            experiment: ABExperiment object

        Returns:
            Test results
        """
        control = experiment.variants[0]
        treatment = experiment.variants[1]

        metric = experiment.primary_metric

        if metric not in control.metrics or metric not in treatment.metrics:
            return {"status": "insufficient_data"}

        control_values = control.metrics[metric]
        treatment_values = treatment.metrics[metric]

        if len(control_values) < experiment.min_samples or len(treatment_values) < experiment.min_samples:
            return {
                "status": "insufficient_samples",
                "control_samples": len(control_values),
                "treatment_samples": len(treatment_values),
                "required_samples": experiment.min_samples
            }

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)

        # Calculate effect size (Cohen's d)
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        pooled_std = np.sqrt((np.std(control_values)**2 + np.std(treatment_values)**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        # Determine if significant
        is_significant = p_value < (1 - experiment.confidence_level)

        # Determine winner
        winner = None
        if is_significant:
            winner = "treatment" if treatment_mean > control_mean else "control"

        result = {
            "status": "completed",
            "metric": metric,
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "improvement": float((treatment_mean - control_mean) / control_mean * 100),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "is_significant": is_significant,
            "confidence_level": experiment.confidence_level,
            "winner": winner
        }

        logger.info(
            f"Statistical test for {experiment.experiment_id}: "
            f"winner={winner}, p_value={p_value:.4f}, improvement={result['improvement']:.2f}%"
        )

        return result

    def conclude_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Conclude experiment and determine winner.

        Args:
            experiment_id: Experiment ID

        Returns:
            Conclusion results
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        stats = self.get_experiment_stats(experiment_id)
        test_result = stats.get("statistical_test", {})

        if test_result.get("is_significant"):
            experiment.winner = test_result["winner"]
            experiment.status = "completed"

            logger.info(
                f"Experiment {experiment_id} concluded: "
                f"Winner = {experiment.winner}, "
                f"Improvement = {test_result.get('improvement', 0):.2f}%"
            )
        else:
            experiment.status = "completed"
            experiment.winner = "control"  # Default to control if no significant difference

            logger.info(
                f"Experiment {experiment_id} concluded: "
                f"No significant difference found, defaulting to control"
            )

        return {
            "experiment_id": experiment_id,
            "winner": experiment.winner,
            "status": experiment.status,
            "stats": stats
        }

    def pause_experiment(self, experiment_id: str) -> None:
        """Pause running experiment"""
        experiment = self.experiments.get(experiment_id)
        if experiment:
            experiment.status = "paused"
            logger.info(f"Paused experiment {experiment_id}")

    def resume_experiment(self, experiment_id: str) -> None:
        """Resume paused experiment"""
        experiment = self.experiments.get(experiment_id)
        if experiment:
            experiment.status = "running"
            logger.info(f"Resumed experiment {experiment_id}")
