"""
Unit tests for MLflow model registry (mlops/mlflow/model_registry.py)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from mlops.mlflow.model_registry import ModelRegistry


class TestModelRegistry:
    """Test ModelRegistry class"""

    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow dependencies"""
        with patch('mlops.mlflow.model_registry.mlflow') as mock_mlflow, \
             patch('mlops.mlflow.model_registry.MlflowClient') as mock_client:

            mock_mlflow.set_tracking_uri = MagicMock()
            mock_mlflow.create_experiment = MagicMock(return_value="exp_id")
            mock_mlflow.get_experiment_by_name = MagicMock()
            mock_mlflow.set_experiment = MagicMock()
            mock_mlflow.start_run = MagicMock()

            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            yield {
                'mlflow': mock_mlflow,
                'client': mock_client_instance
            }

    def test_initialization(self, mock_mlflow):
        """Test ModelRegistry initialization"""
        registry = ModelRegistry(
            tracking_uri="http://localhost:5000",
            experiment_name="test-experiment"
        )

        assert registry.tracking_uri == "http://localhost:5000"
        assert registry.experiment_name == "test-experiment"

    def test_register_yolo_model(self, mock_mlflow, tmp_path):
        """Test YOLO model registration"""
        # Create dummy model file
        model_path = tmp_path / "yolov8n.pt"
        model_path.write_text("dummy model")

        mock_mlflow['client'].create_registered_model = MagicMock()
        mock_mlflow['client'].create_model_version = MagicMock(
            return_value=MagicMock(version="1")
        )

        with patch('mlops.mlflow.model_registry.mlflow.start_run'):
            registry = ModelRegistry()
            version = registry.register_yolo_model(
                model_path=str(model_path),
                model_name="yolo-test",
                metadata={"mAP": 0.85}
            )

            assert version == "1"

    def test_promote_to_staging(self, mock_mlflow):
        """Test promoting model to staging"""
        mock_mlflow['client'].transition_model_version_stage = MagicMock()

        registry = ModelRegistry()
        registry.promote_to_staging("yolo-test", "1")

        mock_mlflow['client'].transition_model_version_stage.assert_called_once_with(
            name="yolo-test",
            version="1",
            stage="Staging"
        )

    def test_promote_to_production(self, mock_mlflow):
        """Test promoting model to production"""
        mock_mlflow['client'].transition_model_version_stage = MagicMock()
        mock_mlflow['client'].get_latest_versions = MagicMock(return_value=[])

        registry = ModelRegistry()
        registry.promote_to_production("yolo-test", "2", archive_existing=True)

        # Should transition to Production
        calls = mock_mlflow['client'].transition_model_version_stage.call_args_list
        assert any("Production" in str(call) for call in calls)

    def test_rollback_to_version(self, mock_mlflow):
        """Test rolling back to previous version"""
        mock_mlflow['client'].transition_model_version_stage = MagicMock()
        mock_mlflow['client'].get_latest_versions = MagicMock(return_value=[])

        registry = ModelRegistry()
        registry.rollback_to_version("yolo-test", "1")

        # Should promote version 1 to Production
        assert mock_mlflow['client'].transition_model_version_stage.called
