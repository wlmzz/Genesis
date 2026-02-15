"""
Health check HTTP server for worker services
Exposes /health, /ready, and /metrics endpoints
"""
import asyncio
import logging
import json
from aiohttp import web
from typing import Callable, Dict, Any, Optional
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)


class HealthServer:
    """
    HTTP server for health checks and metrics
    Runs alongside worker in separate asyncio task
    """

    def __init__(
        self,
        port: int = 8080,
        worker_name: str = "genesis-worker",
        get_status: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        """
        Args:
            port: HTTP server port
            worker_name: Name of the worker service
            get_status: Optional callback to get worker status
        """
        self.port = port
        self.worker_name = worker_name
        self.get_status = get_status or (lambda: {"status": "unknown"})

        self.app = web.Application()
        self.runner = None
        self.site = None

        # Add routes
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_get("/ready", self.ready_handler)
        self.app.router.add_get("/status", self.status_handler)
        self.app.router.add_get("/metrics", self.metrics_handler)

        # Health state
        self.is_healthy = True
        self.is_ready = False

    async def health_handler(self, request: web.Request) -> web.Response:
        """
        Health check endpoint - is the service alive?
        Returns 200 if healthy, 503 if unhealthy
        """
        if self.is_healthy:
            return web.Response(
                text=json.dumps({
                    "status": "healthy",
                    "service": self.worker_name
                }),
                content_type="application/json",
                status=200
            )
        else:
            return web.Response(
                text=json.dumps({
                    "status": "unhealthy",
                    "service": self.worker_name
                }),
                content_type="application/json",
                status=503
            )

    async def ready_handler(self, request: web.Request) -> web.Response:
        """
        Readiness check endpoint - is the service ready to accept traffic?
        Returns 200 if ready, 503 if not ready
        """
        if self.is_ready:
            return web.Response(
                text=json.dumps({
                    "status": "ready",
                    "service": self.worker_name
                }),
                content_type="application/json",
                status=200
            )
        else:
            return web.Response(
                text=json.dumps({
                    "status": "not_ready",
                    "service": self.worker_name
                }),
                content_type="application/json",
                status=503
            )

    async def status_handler(self, request: web.Request) -> web.Response:
        """
        Detailed status endpoint - worker-specific status information
        """
        try:
            status = self.get_status()
            status["service"] = self.worker_name
            status["healthy"] = self.is_healthy
            status["ready"] = self.is_ready

            return web.Response(
                text=json.dumps(status, indent=2),
                content_type="application/json",
                status=200
            )
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.Response(
                text=json.dumps({
                    "error": str(e),
                    "service": self.worker_name
                }),
                content_type="application/json",
                status=500
            )

    async def metrics_handler(self, request: web.Request) -> web.Response:
        """
        Prometheus metrics endpoint
        """
        try:
            metrics = generate_latest()
            return web.Response(
                body=metrics,
                content_type=CONTENT_TYPE_LATEST
            )
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(
                text=f"Error: {str(e)}",
                status=500
            )

    async def start(self):
        """Start the health check server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()

        logger.info(f"Health server started on port {self.port}")
        logger.info(f"  Health:  http://0.0.0.0:{self.port}/health")
        logger.info(f"  Ready:   http://0.0.0.0:{self.port}/ready")
        logger.info(f"  Status:  http://0.0.0.0:{self.port}/status")
        logger.info(f"  Metrics: http://0.0.0.0:{self.port}/metrics")

    async def stop(self):
        """Stop the health check server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Health server stopped")

    def set_healthy(self, healthy: bool):
        """Update health status"""
        self.is_healthy = healthy
        if not healthy:
            logger.warning(f"Worker {self.worker_name} marked as unhealthy")

    def set_ready(self, ready: bool):
        """Update readiness status"""
        self.is_ready = ready
        if ready:
            logger.info(f"Worker {self.worker_name} is ready")
