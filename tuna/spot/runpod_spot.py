"""RunPod spot provider — deploy vLLM on RunPod spot GPU pods via REST API."""

from __future__ import annotations

import logging
import os
import time

import requests

from tuna.catalog import provider_gpu_id, provider_gpu_map
from tuna.models import DeployRequest, DeploymentResult, PreflightCheck, PreflightResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.providers.registry import register

logger = logging.getLogger(__name__)

_API_BASE = "https://rest.runpod.io/v1"


def _headers() -> dict[str, str]:
    """Return auth headers for the RunPod REST API."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY environment variable is not set. "
            "Get your API key from https://www.runpod.io/console/user/settings"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


class RunPodSpotProvider(InferenceProvider):
    """Deploy a vLLM server on RunPod spot GPU pods via REST API.

    Uses RunPod's GPU pod API with spot (bid-based) instances to run
    vLLM as a persistent server, providing an OpenAI-compatible endpoint.
    """

    def name(self) -> str:
        return "runpod-spot"

    def preflight(self, request: DeployRequest) -> PreflightResult:
        result = PreflightResult(provider=self.name())

        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            result.checks.append(PreflightCheck(
                name="api_key",
                passed=False,
                message="RUNPOD_API_KEY environment variable is not set",
                fix_command="export RUNPOD_API_KEY=<your-key>  # https://www.runpod.io/console/user/settings",
            ))
            return result

        result.checks.append(PreflightCheck(
            name="api_key",
            passed=True,
            message="RUNPOD_API_KEY is set",
        ))

        # Validate the key works
        try:
            resp = requests.get(
                f"{_API_BASE}/pods",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 401:
                result.checks.append(PreflightCheck(
                    name="api_key_valid",
                    passed=False,
                    message="RUNPOD_API_KEY is invalid (401 Unauthorized)",
                    fix_command="export RUNPOD_API_KEY=<your-key>",
                ))
            else:
                resp.raise_for_status()
                result.checks.append(PreflightCheck(
                    name="api_key_valid",
                    passed=True,
                    message="RUNPOD_API_KEY is valid",
                ))
        except requests.exceptions.ConnectionError:
            result.checks.append(PreflightCheck(
                name="api_key_valid",
                passed=False,
                message="Could not reach RunPod API (connection error)",
            ))
        except Exception as e:
            result.checks.append(PreflightCheck(
                name="api_key_valid",
                passed=False,
                message=f"RunPod API check failed: {e}",
            ))

        return result

    def plan(self, request: DeployRequest, vllm_cmd: str) -> ProviderPlan:
        service_name = f"{request.service_name}-spot"

        try:
            gpu_type_id = provider_gpu_id(request.gpu, "runpod")
        except KeyError:
            raise ValueError(
                f"Unknown GPU type for RunPod: {request.gpu!r}. "
                f"Supported: {sorted(provider_gpu_map('runpod').keys())}"
            )

        hf_token = os.environ.get("HF_TOKEN", "")

        # Build environment variables for vLLM
        env = {
            "HF_TOKEN": hf_token,
        }

        # Docker command: install vLLM and run the server
        docker_cmd = (
            f"pip install vllm=={request.vllm_version} && {vllm_cmd}"
        )

        metadata = {
            "service_name": service_name,
            "gpu_type_id": gpu_type_id,
            "gpu_count": str(request.gpu_count),
            "docker_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "docker_cmd": docker_cmd,
            "port": "8001",
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

        # Build environment list for RunPod pod API
        env_list = {k: v for k, v in plan.env.items() if v}

        pod_payload = {
            "name": service_name,
            "imageName": plan.metadata["docker_image"],
            "gpuTypeId": plan.metadata["gpu_type_id"],
            "gpuCount": int(plan.metadata["gpu_count"]),
            "cloudType": "SPOT",
            "dockerArgs": plan.metadata["docker_cmd"],
            "containerDiskInGb": 50,
            "volumeInGb": 100,
            "ports": f"{plan.metadata['port']}/http",
            "env": env_list,
        }

        logger.info("Creating RunPod spot pod: %s", service_name)
        try:
            resp = requests.post(
                f"{_API_BASE}/pods",
                headers=headers,
                json=pod_payload,
                timeout=60,
            )
            resp.raise_for_status()
            pod_data = resp.json()
            pod_id = pod_data["id"]
        except Exception as e:
            logger.error("RunPod spot pod creation failed: %s", e)
            return DeploymentResult(
                provider=self.name(),
                error=f"Pod creation failed: {e}",
                metadata={"service_name": service_name},
            )

        logger.info("RunPod spot pod %s created (id=%s), waiting for ready...", service_name, pod_id)

        # Poll for the pod to become RUNNING and get its endpoint
        endpoint_url = self._wait_for_pod_ready(headers, pod_id, plan.metadata["port"])

        if endpoint_url:
            logger.info("RunPod spot pod %s ready at %s", service_name, endpoint_url)
            return DeploymentResult(
                provider=self.name(),
                endpoint_url=endpoint_url,
                health_url=f"{endpoint_url}/health",
                metadata={
                    "service_name": service_name,
                    "pod_id": pod_id,
                },
            )

        # Pod created but not ready yet — return with info for later discovery
        return DeploymentResult(
            provider=self.name(),
            error="Pod created but endpoint not yet available (still provisioning)",
            metadata={
                "service_name": service_name,
                "pod_id": pod_id,
            },
        )

    def _wait_for_pod_ready(
        self, headers: dict, pod_id: str, port: str, timeout: int = 600
    ) -> str | None:
        """Poll RunPod API until the pod is RUNNING and has a public IP."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{_API_BASE}/pods/{pod_id}",
                    headers=headers,
                    timeout=15,
                )
                resp.raise_for_status()
                pod = resp.json()

                status = pod.get("desiredStatus", "")
                runtime = pod.get("runtime")

                if status == "RUNNING" and runtime:
                    # RunPod provides proxy endpoint via pod_id
                    pod_ip = runtime.get("gpus", [{}])[0].get("publicIp") if runtime.get("gpus") else None
                    # Use the RunPod proxy URL format
                    proxy_url = f"https://{pod_id}-{port}.proxy.runpod.net"
                    return proxy_url

                if status in ("EXITED", "TERMINATED"):
                    logger.error("Pod %s entered terminal status: %s", pod_id, status)
                    return None

            except Exception as e:
                logger.debug("Poll for pod %s failed: %s", pod_id, e)

            time.sleep(15)

        logger.warning("Timed out waiting for pod %s to become ready", pod_id)
        return None

    def destroy(self, result: DeploymentResult) -> None:
        pod_id = result.metadata.get("pod_id")
        if not pod_id:
            logger.warning("No pod_id in metadata, cannot destroy RunPod spot pod")
            return

        try:
            headers = _headers()
        except RuntimeError as e:
            logger.error("Cannot destroy RunPod pod: %s", e)
            return

        logger.info("Terminating RunPod spot pod %s", pod_id)
        try:
            resp = requests.delete(
                f"{_API_BASE}/pods/{pod_id}",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            logger.info("RunPod spot pod %s terminated", pod_id)
        except Exception as e:
            logger.warning("Failed to terminate RunPod pod %s: %s", pod_id, e)

    def status(self, service_name: str) -> dict:
        spot_service = f"{service_name}-spot"

        try:
            headers = _headers()
        except RuntimeError:
            return {"provider": self.name(), "status": "unknown", "error": "RUNPOD_API_KEY not set"}

        # List pods and find ours by name
        try:
            resp = requests.get(
                f"{_API_BASE}/pods",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            pods = resp.json()
        except Exception as e:
            return {"provider": self.name(), "status": "unknown", "error": str(e)}

        for pod in pods:
            if pod.get("name") == spot_service:
                pod_status = pod.get("desiredStatus", "unknown")
                result = {
                    "provider": self.name(),
                    "service_name": spot_service,
                    "pod_id": pod.get("id"),
                    "status": pod_status.lower(),
                }
                runtime = pod.get("runtime")
                if runtime and pod.get("id"):
                    result["endpoint"] = f"https://{pod['id']}-8001.proxy.runpod.net"
                return result

        return {
            "provider": self.name(),
            "service_name": spot_service,
            "status": "not found",
        }


register("runpod-spot", RunPodSpotProvider)
