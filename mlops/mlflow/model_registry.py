"""
MLflow Model Registry for Genesis Platform

Manages model versions, staging, and production deployment.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model registry for all Genesis ML models.

    Features:
    - Model versioning
    - Stage management (Staging, Production, Archived)
    - Model metadata tracking
    - Automatic model promotion
    - Rollback capabilities
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        artifact_location: str = "./mlops/mlflow/artifacts",
        experiment_name: str = "genesis-models"
    ):
        """
        Initialize MLflow model registry.

        Args:
            tracking_uri: MLflow tracking server URI
            artifact_location: Local or S3 path for artifacts
            experiment_name: MLflow experiment name
        """
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.experiment_name = experiment_name

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)

        # Initialize client
        self.client = MlflowClient(tracking_uri)

        logger.info(f"Model registry initialized: {tracking_uri} (experiment: {experiment_name})")

    def register_yolo_model(
        self,
        model_path: str,
        model_name: str = "yolo-person-detection",
        description: str = "YOLOv8 person detection model",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register YOLO model to MLflow.

        Args:
            model_path: Path to YOLO weights (.pt file)
            model_name: Registered model name
            description: Model description
            metadata: Additional metadata (mAP, training config, etc.)

        Returns:
            Model version string
        """
        with mlflow.start_run(run_name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            # Log model file
            mlflow.log_artifact(model_path, artifact_path="model")

            # Log metadata
            if metadata:
                mlflow.log_params(metadata)

            # Log model info
            mlflow.log_param("model_type", "yolov8")
            mlflow.log_param("model_task", "person-detection")
            mlflow.log_param("input_size", "640x640")
            mlflow.log_param("registered_at", datetime.now().isoformat())

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"

            try:
                # Create registered model
                self.client.create_registered_model(
                    model_name,
                    description=description
                )
            except mlflow.exceptions.MlflowException:
                # Model already exists
                pass

            # Create model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run.info.run_id
            )

            logger.info(f"Registered {model_name} version {model_version.version}")

            return model_version.version

    def register_face_recognition_model(
        self,
        model_name: str = "facenet-embeddings",
        description: str = "FaceNet embedding model",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register face recognition model.

        Args:
            model_name: Registered model name
            description: Model description
            metadata: Model metadata

        Returns:
            Model version string
        """
        with mlflow.start_run(run_name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            # Log metadata
            if metadata:
                mlflow.log_params(metadata)

            mlflow.log_param("model_type", "facenet")
            mlflow.log_param("embedding_size", 512)
            mlflow.log_param("registered_at", datetime.now().isoformat())

            # Register model
            model_uri = f"runs:/{run.info.run_id}"

            try:
                self.client.create_registered_model(
                    model_name,
                    description=description
                )
            except mlflow.exceptions.MlflowException:
                pass

            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run.info.run_id
            )

            logger.info(f"Registered {model_name} version {model_version.version}")

            return model_version.version

    def promote_to_staging(self, model_name: str, version: str) -> None:
        """
        Promote model version to Staging stage.

        Args:
            model_name: Registered model name
            version: Model version to promote
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        logger.info(f"Promoted {model_name} v{version} to Staging")

    def promote_to_production(self, model_name: str, version: str, archive_existing: bool = True) -> None:
        """
        Promote model version to Production stage.

        Args:
            model_name: Registered model name
            version: Model version to promote
            archive_existing: Whether to archive existing production models
        """
        if archive_existing:
            # Archive existing production models
            existing_prod = self.client.get_latest_versions(model_name, stages=["Production"])
            for model_version in existing_prod:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Archived"
                )
                logger.info(f"Archived {model_name} v{model_version.version}")

        # Promote new version to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        logger.info(f"Promoted {model_name} v{version} to Production")

    def rollback_to_version(self, model_name: str, version: str) -> None:
        """
        Rollback to a specific model version.

        Args:
            model_name: Registered model name
            version: Version to rollback to
        """
        self.promote_to_production(model_name, version, archive_existing=True)
        logger.warning(f"Rolled back {model_name} to version {version}")

    def get_production_model(self, model_name: str) -> Optional[Any]:
        """
        Get current production model version.

        Args:
            model_name: Registered model name

        Returns:
            Model version object or None
        """
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0]
        return None

    def get_staging_model(self, model_name: str) -> Optional[Any]:
        """
        Get current staging model version.

        Args:
            model_name: Registered model name

        Returns:
            Model version object or None
        """
        versions = self.client.get_latest_versions(model_name, stages=["Staging"])
        if versions:
            return versions[0]
        return None

    def list_model_versions(self, model_name: str) -> List[Any]:
        """
        List all versions of a model.

        Args:
            model_name: Registered model name

        Returns:
            List of model versions
        """
        try:
            return self.client.search_model_versions(f"name='{model_name}'")
        except mlflow.exceptions.MlflowException:
            logger.warning(f"Model {model_name} not found")
            return []

    def log_model_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log metrics for a specific model version.

        Args:
            model_name: Registered model name
            version: Model version
            metrics: Dictionary of metrics (e.g., {"mAP": 0.85, "precision": 0.92})
        """
        model_version = self.client.get_model_version(model_name, version)
        run_id = model_version.run_id

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)

        logger.info(f"Logged metrics for {model_name} v{version}: {metrics}")

    def add_model_tag(
        self,
        model_name: str,
        version: str,
        key: str,
        value: str
    ) -> None:
        """
        Add tag to model version.

        Args:
            model_name: Registered model name
            version: Model version
            key: Tag key
            value: Tag value
        """
        self.client.set_model_version_tag(model_name, version, key, value)
        logger.info(f"Added tag {key}={value} to {model_name} v{version}")

    def compare_models(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            model_name: Registered model name
            version1: First version
            version2: Second version

        Returns:
            Comparison dictionary
        """
        mv1 = self.client.get_model_version(model_name, version1)
        mv2 = self.client.get_model_version(model_name, version2)

        # Get metrics from runs
        run1 = self.client.get_run(mv1.run_id)
        run2 = self.client.get_run(mv2.run_id)

        comparison = {
            "version1": {
                "version": version1,
                "stage": mv1.current_stage,
                "created": mv1.creation_timestamp,
                "metrics": run1.data.metrics,
                "params": run1.data.params
            },
            "version2": {
                "version": version2,
                "stage": mv2.current_stage,
                "created": mv2.creation_timestamp,
                "metrics": run2.data.metrics,
                "params": run2.data.params
            }
        }

        return comparison
