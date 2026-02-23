"""Vast.ai spot provider — deploy vLLM on Vast.ai interruptible instances via REST API."""

from __future__ import annotations

import logging
import os
import time

import requests

from tuna.models import DeployRequest, DeploymentResult, PreflightCheck, PreflightResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register

logger = logging.getLogger(__name__)

_API_BASE = "https://console.vast.ai/api/v0"

# Maps tuna GPU short names to Vast.ai gpu_name search strings.
_VASTAI_GPU_MAP: dict[str, str] = {
    "RTX4090": "RTX 4090",
    "A6000": "RTX A6000",
    "A5000": "RTX A5000",
    "A4000": "RTX A4000",
    "A100_80GB": "A100_SXM4",
    "A100_40GB": "A100_PCIE",
    "H100": "H100",
    "H200": "H200",
    "L40": "L40",
    "L40S": "L40S",
    "L4": "L4",
    "A40": "A40",
    "A10": "A10",
    "A10G": "A10G",
    "T4": "T4",
}


def _headers() -> dict[str, str]:
    """Return auth headers for the Vast.ai REST API."""
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VASTAI_API_KEY environment variable is not set. "
            "Get your API key from https://cloud.vast.ai/account/"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


class VastAISpotProvider(InferenceProvider):
    """Deploy a vLLM server on Vast.ai interruptible (spot) instances.

    Uses the Vast.ai REST API to search for the cheapest available offer
    matching the requested GPU, then creates an instance running vLLM.
    """

    def name(self) -> str:
        return "vastai-spot"

    def preflight(self, request: DeployRequest) -> PreflightResult:
        result = PreflightResult(provider=self.name())

        api_key = os.environ.get("VASTAI_API_KEY")
        if not api_key:
            result.checks.append(PreflightCheck(
                name="api_key",
                passed=False,
                message="VASTAI_API_KEY environment variable is not set",
                fix_command="export VASTAI_API_KEY=<your-key>  # https://cloud.vast.ai/account/",
            ))
            return result

        result.checks.append(PreflightCheck(
            name="api_key",
            passed=True,
            message="VASTAI_API_KEY is set",
        ))

        # Validate the key works
        try:
            resp = requests.get(
                f"{_API_BASE}/users/current",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 401 or resp.status_code == 403:
                result.checks.append(PreflightCheck(
                    name="api_key_valid",
                    passed=False,
                    message=f"VASTAI_API_KEY is invalid ({resp.status_code})",
                    fix_command="export VASTAI_API_KEY=<your-key>",
                ))
            else:
                resp.raise_for_status()
                result.checks.append(PreflightCheck(
                    name="api_key_valid",
                    passed=True,
                    message="VASTAI_API_KEY is valid",
                ))
        except requests.exceptions.ConnectionError:
            result.checks.append(PreflightCheck(
                name="api_key_valid",
                passed=False,
                message="Could not reach Vast.ai API (connection error)",
            ))
        except Exception as e:
            result.checks.append(PreflightCheck(
                name="api_key_valid",
                passed=False,
                message=f"Vast.ai API check failed: {e}",
            ))

        # Check GPU availability
        gpu_name = _VASTAI_GPU_MAP.get(request.gpu)
        if not gpu_name:
            result.checks.append(PreflightCheck(
                name="gpu_supported",
                passed=False,
                message=f"GPU {request.gpu!r} not mapped for Vast.ai. Supported: {sorted(_VASTAI_GPU_MAP.keys())}",
            ))
        else:
            result.checks.append(PreflightCheck(
                name="gpu_supported",
                passed=True,
                message=f"GPU {request.gpu} mapped to Vast.ai '{gpu_name}'",
            ))

        return result

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        service_name = f"{request.service_name}-spot"

        gpu_name = _VASTAI_GPU_MAP.get(request.gpu)
        if not gpu_name:
            raise ValueError(
                f"Unknown GPU type for Vast.ai: {request.gpu!r}. "
                f"Supported: {sorted(_VASTAI_GPU_MAP.keys())}"
            )

        hf_token = os.environ.get("HF_TOKEN", "")

        env = {
            "HF_TOKEN": hf_token,
        }

        # On-create script: install vLLM and run the server
        onstart_cmd = (
            f"pip install vllm=={request.vllm_version} && {vllm_cmd}"
        )

        metadata = {
            "service_name": service_name,
            "gpu_name": gpu_name,
            "gpu_count": str(request.gpu_count),
            "docker_image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel",
            "onstart_cmd": onstart_cmd,
            "port": "8001",
            "disk_gb": "100",
        }

        return ProviderPlan(
            provider=self.name(),
            rendered_script="",
            env=env,
            metadata=metadata,
        )

    def deploy(self, plan: ProviderPlan) -> DeploymentResult:
        service_name = plan.metadata["service_name"]

        try:
            headers = _headers()
        except RuntimeError as e:
            return DeploymentResult(
                provider=self.name(),
                error=str(e),
                metadata={"service_name": service_name},
            )

        # Step 1: Search for offers matching our GPU requirements
        gpu_name = plan.metadata["gpu_name"]
        gpu_count = int(plan.metadata["gpu_count"])
        disk_gb = int(plan.metadata["disk_gb"])

        offer_id = self._find_best_offer(headers, gpu_name, gpu_count, disk_gb)
        if not offer_id:
            return DeploymentResult(
                provider=self.name(),
                error=f"No Vast.ai offers found for GPU '{gpu_name}' x{gpu_count}",
                metadata={"service_name": service_name},
            )

        # Step 2: Create instance from offer
        env_vars = {k: v for k, v in plan.env.items() if v}

        instance_payload = {
            "client_id": "me",
            "image": plan.metadata["docker_image"],
            "disk": disk_gb,
            "onstart": plan.metadata["onstart_cmd"],
            "env": env_vars,
            "label": service_name,
        }

        logger.info("Creating Vast.ai instance from offer %s for %s", offer_id, service_name)
        try:
            resp = requests.put(
                f"{_API_BASE}/asks/{offer_id}/",
                headers=headers,
                json=instance_payload,
                timeout=60,
            )
            resp.raise_for_status()
            instance_data = resp.json()

            if not instance_data.get("success"):
                error_msg = instance_data.get("msg", "Unknown error")
                return DeploymentResult(
                    provider=self.name(),
                    error=f"Instance creation failed: {error_msg}",
                    metadata={"service_name": service_name},
                )

            instance_id = str(instance_data.get("new_contract"))
        except Exception as e:
            logger.error("Vast.ai instance creation failed: %s", e)
            return DeploymentResult(
                provider=self.name(),
                error=f"Instance creation failed: {e}",
                metadata={"service_name": service_name},
            )

        logger.info("Vast.ai instance %s created, waiting for ready...", instance_id)

        # Step 3: Wait for the instance to start and get the endpoint
        endpoint_url = self._wait_for_instance_ready(
            headers, instance_id, plan.metadata["port"]
        )

        if endpoint_url:
            logger.info("Vast.ai instance %s ready at %s", instance_id, endpoint_url)
            return DeploymentResult(
                provider=self.name(),
                endpoint_url=endpoint_url,
                health_url=f"{endpoint_url}/health",
                metadata={
                    "service_name": service_name,
                    "instance_id": instance_id,
                },
            )

        return DeploymentResult(
            provider=self.name(),
            error="Instance created but endpoint not yet available (still provisioning)",
            metadata={
                "service_name": service_name,
                "instance_id": instance_id,
            },
        )

    def _find_best_offer(
        self,
        headers: dict,
        gpu_name: str,
        gpu_count: int,
        disk_gb: int,
    ) -> str | None:
        """Search Vast.ai for the cheapest interruptible offer matching our requirements."""
        # Build search query for Vast.ai bundles/search endpoint
        query_params = {
            "q": {
                "gpu_name": {"eq": gpu_name},
                "num_gpus": {"gte": gpu_count},
                "disk_space": {"gte": disk_gb},
                "rentable": {"eq": True},
                "rented": {"eq": False},
                "type": "interruptible",
                "order": [["dph_total", "asc"]],
            }
        }

        try:
            resp = requests.post(
                f"{_API_BASE}/bundles/",
                headers=headers,
                json=query_params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            offers = data.get("offers", [])

            if not offers:
                logger.warning("No Vast.ai offers found for %s x%d", gpu_name, gpu_count)
                return None

            # Return the cheapest offer
            best = offers[0]
            logger.info(
                "Found Vast.ai offer %s: %s x%d at $%.3f/hr",
                best["id"], gpu_name, gpu_count, best.get("dph_total", 0),
            )
            return str(best["id"])

        except Exception as e:
            logger.error("Vast.ai offer search failed: %s", e)
            return None

    def _wait_for_instance_ready(
        self, headers: dict, instance_id: str, port: str, timeout: int = 600
    ) -> str | None:
        """Poll Vast.ai API until the instance is running and has an IP."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{_API_BASE}/instances/{instance_id}/",
                    headers=headers,
                    timeout=15,
                )
                resp.raise_for_status()
                instance = resp.json()

                actual_status = instance.get("actual_status")
                public_ip = instance.get("public_ipaddr")
                ports = instance.get("ports", {})

                if actual_status == "running" and public_ip and ports:
                    # Find the mapped port for our target port
                    port_key = f"{port}/tcp"
                    mapped = ports.get(port_key)
                    if mapped:
                        host_port = mapped[0].get("HostPort", port)
                        endpoint = f"http://{public_ip}:{host_port}"
                        return endpoint

                if actual_status in ("exited", "error"):
                    logger.error("Instance %s entered terminal state: %s", instance_id, actual_status)
                    return None

            except Exception as e:
                logger.debug("Poll for instance %s failed: %s", instance_id, e)

            time.sleep(15)

        logger.warning("Timed out waiting for instance %s to become ready", instance_id)
        return None

    def destroy(self, result: DeploymentResult) -> None:
        instance_id = result.metadata.get("instance_id")
        if not instance_id:
            logger.warning("No instance_id in metadata, cannot destroy Vast.ai instance")
            return

        try:
            headers = _headers()
        except RuntimeError as e:
            logger.error("Cannot destroy Vast.ai instance: %s", e)
            return

        logger.info("Destroying Vast.ai instance %s", instance_id)
        try:
            resp = requests.delete(
                f"{_API_BASE}/instances/{instance_id}/",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            logger.info("Vast.ai instance %s destroyed", instance_id)
        except Exception as e:
            logger.warning("Failed to destroy Vast.ai instance %s: %s", instance_id, e)

    def status(self, service_name: str) -> dict:
        spot_service = f"{service_name}-spot"

        try:
            headers = _headers()
        except RuntimeError:
            return {"provider": self.name(), "status": "unknown", "error": "VASTAI_API_KEY not set"}

        # List instances and find ours by label
        try:
            resp = requests.get(
                f"{_API_BASE}/instances/",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            instances = data.get("instances", [])
        except Exception as e:
            return {"provider": self.name(), "status": "unknown", "error": str(e)}

        for inst in instances:
            if inst.get("label") == spot_service:
                actual_status = inst.get("actual_status", "unknown")
                result = {
                    "provider": self.name(),
                    "service_name": spot_service,
                    "instance_id": str(inst.get("id")),
                    "status": actual_status,
                }
                public_ip = inst.get("public_ipaddr")
                ports = inst.get("ports", {})
                if public_ip and ports:
                    port_key = "8001/tcp"
                    mapped = ports.get(port_key)
                    if mapped:
                        host_port = mapped[0].get("HostPort", "8001")
                        result["endpoint"] = f"http://{public_ip}:{host_port}"
                return result

        return {
            "provider": self.name(),
            "service_name": spot_service,
            "status": "not found",
        }


register("vastai-spot", VastAISpotProvider)
