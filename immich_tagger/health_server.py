"""HTTP health and metrics endpoints."""

import asyncio
import time
from datetime import datetime, timezone

from aiohttp import web

from .build_info import get_build_info
from .config import settings
from .logging import get_logger
from .models import HealthStatus


class HealthServer:
    """Health server that does not mutate active processing state."""

    def __init__(self, processor):
        self.processor = processor
        self.logger = get_logger("health_server")
        self.app = web.Application()
        self.connection_cache = {}
        self.cache_duration = 43200
        self.setup_routes()

    def setup_routes(self):
        """Register HTTP routes."""
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_get("/metrics", self.metrics_handler)
        self.app.router.add_get("/", self.root_handler)

    def _test_connection_cached(self, library_index: int, api_key: str) -> bool:
        """Test one library connection with a 12-hour result cache."""
        current_time = time.time()
        cache_key = f"library_{library_index}"
        cached = self.connection_cache.get(cache_key)
        if cached and current_time - cached[0] < self.cache_duration:
            return cached[1]

        try:
            response = self.processor.immich_client._make_request_silent(
                method="GET",
                endpoint="/api/tags",
                api_key=api_key,
            )
            result = response.status_code == 200
        except Exception:
            result = False

        self.connection_cache[cache_key] = (current_time, result)
        return result

    def _clear_connection_cache(self):
        """Force fresh connectivity checks."""
        self.connection_cache.clear()
        self.logger.debug("Connection cache cleared")

    def _collect_health_response(self):
        """Collect health data without changing the active API key/library."""
        library_statuses = {}
        overall_healthy = True
        library_configs = self.processor.immich_client.library_configs

        for index, library_config in enumerate(library_configs):
            library_name = library_config["name"]
            api_key = library_config["api_key"]
            try:
                connection_ok = self._test_connection_cached(index, api_key)
                user_info = (
                    self.processor.immich_client.get_user_info_for_api_key(api_key)
                )
                library_statuses[library_name] = {
                    "status": "healthy" if connection_ok else "unhealthy",
                    "user": {
                        "name": user_info["name"],
                        "email": user_info["email"],
                    },
                    "metrics": self.processor.library_metrics.get(
                        library_name,
                        {},
                    ),
                }
                if not connection_ok:
                    overall_healthy = False
            except Exception as e:
                library_statuses[library_name] = {
                    "status": "error",
                    "error": str(e),
                    "metrics": {},
                }
                overall_healthy = False
                self.connection_cache.pop(f"library_{index}", None)

        build_info = get_build_info()
        run_status = self.processor.get_run_status()
        health_status = HealthStatus(
            status="healthy" if overall_healthy else "unhealthy",
            version=build_info["version"],
            build=build_info,
            run_status=run_status,
            metrics={
                "libraries": library_statuses,
                "global": self.processor.get_metrics(),
                "total_libraries": len(library_configs),
            },
        )
        return overall_healthy, health_status.model_dump()

    async def health_handler(self, request):
        """Serve health checks while model inference runs separately."""
        try:
            overall_healthy, payload = await asyncio.to_thread(
                self._collect_health_response
            )
            return web.json_response(
                payload,
                status=200 if overall_healthy else 503,
            )
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._clear_connection_cache()
            return web.json_response(
                {"status": "unhealthy", "error": str(e)},
                status=503,
            )

    async def metrics_handler(self, request):
        """Return processing and host metrics."""
        try:
            import psutil

            metrics = self.processor.get_metrics()
            metrics["build"] = get_build_info()
            metrics.update(
                {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                }
            )
            return web.json_response(metrics)
        except Exception as e:
            self.logger.error(f"Metrics retrieval failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def root_handler(self, request):
        """Return basic service information."""
        build_info = get_build_info()
        return web.json_response(
            {
                "service": "Immich Auto-Tagger",
                "version": build_info["version"],
                "build": build_info,
                "endpoints": {
                    "/health": "Health check endpoint",
                    "/metrics": "Processing metrics",
                    "/": "Service information",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def start(self):
        """Start the HTTP listener."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", settings.health_port)
        await site.start()
        self.logger.info(
            f"Health server started on 0.0.0.0:{settings.health_port}"
        )
        return runner

    async def stop(self, runner):
        """Stop the HTTP listener."""
        await runner.cleanup()
        self.logger.info("Health server stopped")


async def run_health_server(processor):
    """Run the health server until its task is cancelled."""
    server = HealthServer(processor)
    runner = await server.start()
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await server.stop(runner)
