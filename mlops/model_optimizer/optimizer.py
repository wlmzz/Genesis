"""
Model Optimization for Genesis Platform

Converts and optimizes models for production deployment (ONNX, TensorRT).
"""

import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch
import numpy as np
import time

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Optimize models for inference performance.

    Supports:
    - ONNX export and optimization
    - TensorRT conversion (NVIDIA GPUs)
    - Quantization (INT8, FP16)
    - Model pruning
    """

    def __init__(self):
        self.onnx_available = self._check_onnx()
        self.tensorrt_available = self._check_tensorrt()

        logger.info(
            f"Model optimizer initialized "
            f"(ONNX: {self.onnx_available}, TensorRT: {self.tensorrt_available})"
        )

    def _check_onnx(self) -> bool:
        """Check if ONNX is available"""
        try:
            import onnx
            import onnxruntime
            return True
        except ImportError:
            logger.warning("ONNX not available. Install: pip install onnx onnxruntime")
            return False

    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            logger.warning("TensorRT not available. Install NVIDIA TensorRT.")
            return False

    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict] = None
    ) -> bool:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (e.g., (1, 3, 640, 640))
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Optional dynamic axes for variable input sizes

        Returns:
            Success status
        """
        if not self.onnx_available:
            logger.error("ONNX not available")
            return False

        try:
            import torch.onnx

            # Create dummy input
            dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)

            # Export to ONNX
            model.eval()
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes or {}
                )

            logger.info(f"Exported model to ONNX: {output_path}")

            # Verify
            self.verify_onnx_model(output_path)

            return True

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    def verify_onnx_model(self, model_path: str) -> bool:
        """
        Verify ONNX model validity.

        Args:
            model_path: Path to ONNX model

        Returns:
            Valid status
        """
        if not self.onnx_available:
            return False

        try:
            import onnx

            model = onnx.load(model_path)
            onnx.checker.check_model(model)

            logger.info(f"ONNX model verified: {model_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False

    def optimize_onnx(
        self,
        input_path: str,
        output_path: str,
        optimization_level: str = "all"  # "basic", "extended", "all"
    ) -> bool:
        """
        Optimize ONNX model for inference.

        Args:
            input_path: Input ONNX model path
            output_path: Output optimized model path
            optimization_level: Optimization level

        Returns:
            Success status
        """
        if not self.onnx_available:
            logger.error("ONNX not available")
            return False

        try:
            import onnxruntime as ort

            # Set optimization level
            opt_levels = {
                "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            }

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = opt_levels.get(
                optimization_level,
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.optimized_model_filepath = output_path

            # Create session (triggers optimization)
            ort.InferenceSession(input_path, sess_options)

            logger.info(f"Optimized ONNX model: {output_path} (level: {optimization_level})")
            return True

        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return False

    def quantize_onnx(
        self,
        input_path: str,
        output_path: str,
        quantization_mode: str = "dynamic"  # "dynamic" or "static"
    ) -> bool:
        """
        Quantize ONNX model to INT8.

        Args:
            input_path: Input ONNX model path
            output_path: Output quantized model path
            quantization_mode: "dynamic" (post-training) or "static" (calibration required)

        Returns:
            Success status
        """
        if not self.onnx_available:
            logger.error("ONNX not available")
            return False

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            if quantization_mode == "dynamic":
                quantize_dynamic(
                    input_path,
                    output_path,
                    weight_type=QuantType.QInt8
                )

                logger.info(f"Quantized ONNX model (INT8): {output_path}")
                return True
            else:
                logger.warning("Static quantization requires calibration data")
                return False

        except Exception as e:
            logger.error(f"ONNX quantization failed: {e}")
            return False

    def convert_to_tensorrt(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",  # "fp32", "fp16", "int8"
        max_batch_size: int = 1,
        workspace_size_mb: int = 2048
    ) -> bool:
        """
        Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Input ONNX model path
            output_path: Output TensorRT engine path
            precision: Precision mode ("fp32", "fp16", "int8")
            max_batch_size: Maximum batch size
            workspace_size_mb: Workspace memory in MB

        Returns:
            Success status
        """
        if not self.tensorrt_available:
            logger.error("TensorRT not available")
            return False

        try:
            import tensorrt as trt

            logger.info(f"Converting to TensorRT ({precision})...")

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Create builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False

            # Create builder config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_mb * 1024 * 1024)

            # Set precision
            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                # INT8 requires calibration - simplified here
                logger.warning("INT8 requires calibration. Using default.")

            # Build engine
            engine = builder.build_serialized_network(network, config)

            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False

            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine)

            logger.info(f"TensorRT engine created: {output_path}")
            return True

        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False

    def benchmark_model(
        self,
        model_path: str,
        model_type: str,  # "pytorch", "onnx", "tensorrt"
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark model inference performance.

        Args:
            model_path: Model file path
            model_type: Model type ("pytorch", "onnx", "tensorrt")
            input_shape: Input tensor shape
            num_iterations: Number of benchmark iterations
            warmup_iterations: Warmup iterations

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking {model_type} model: {model_path}")

        if model_type == "onnx" and self.onnx_available:
            return self._benchmark_onnx(model_path, input_shape, num_iterations, warmup_iterations)
        elif model_type == "pytorch":
            return self._benchmark_pytorch(model_path, input_shape, num_iterations, warmup_iterations)
        elif model_type == "tensorrt" and self.tensorrt_available:
            return self._benchmark_tensorrt(model_path, input_shape, num_iterations, warmup_iterations)
        else:
            logger.error(f"Unsupported model type or runtime not available: {model_type}")
            return {}

    def _benchmark_onnx(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_iterations: int,
        warmup_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark ONNX model"""
        import onnxruntime as ort

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(warmup_iterations):
            session.run(None, {input_name: dummy_input})

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            latencies.append(time.perf_counter() - start)

        return {
            "model_type": "onnx",
            "mean_latency_ms": np.mean(latencies) * 1000,
            "std_latency_ms": np.std(latencies) * 1000,
            "min_latency_ms": np.min(latencies) * 1000,
            "max_latency_ms": np.max(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "throughput_fps": 1.0 / np.mean(latencies)
        }

    def _benchmark_pytorch(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_iterations: int,
        warmup_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark PyTorch model"""
        model = torch.load(model_path)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        dummy_input = torch.randn(*input_shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                model(dummy_input)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

        return {
            "model_type": "pytorch",
            "device": str(device),
            "mean_latency_ms": np.mean(latencies) * 1000,
            "std_latency_ms": np.std(latencies) * 1000,
            "min_latency_ms": np.min(latencies) * 1000,
            "max_latency_ms": np.max(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "throughput_fps": 1.0 / np.mean(latencies)
        }

    def _benchmark_tensorrt(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_iterations: int,
        warmup_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark TensorRT engine"""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(model_path, 'rb') as f:
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate buffers
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = input_size  # Simplified
        input_buffer = cuda.mem_alloc(input_size)
        output_buffer = cuda.mem_alloc(output_size)

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(warmup_iterations):
            cuda.memcpy_htod(input_buffer, dummy_input)
            context.execute_v2([int(input_buffer), int(output_buffer)])

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            cuda.memcpy_htod(input_buffer, dummy_input)
            context.execute_v2([int(input_buffer), int(output_buffer)])
            cuda.Context.synchronize()
            latencies.append(time.perf_counter() - start)

        return {
            "model_type": "tensorrt",
            "mean_latency_ms": np.mean(latencies) * 1000,
            "std_latency_ms": np.std(latencies) * 1000,
            "min_latency_ms": np.min(latencies) * 1000,
            "max_latency_ms": np.max(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "throughput_fps": 1.0 / np.mean(latencies)
        }
