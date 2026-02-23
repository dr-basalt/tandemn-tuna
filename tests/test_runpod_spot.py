"""Tests for tuna.spot.runpod_spot — plan/deploy/destroy, no real API calls."""

from unittest.mock import MagicMock, patch

import pytest

from tuna.models import DeployRequest, DeploymentResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.scaling import ScalingPolicy, ServerlessScaling, SpotScaling
from tuna.spot.runpod_spot import RunPodSpotProvider


class TestRunPodSpotProvider:
    def test_is_inference_provider(self):
        assert issubclass(RunPodSpotProvider, InferenceProvider)
        assert isinstance(RunPodSpotProvider(), InferenceProvider)

    def test_name(self):
        provider = RunPodSpotProvider()
        assert provider.name() == "runpod-spot"


class TestRunPodSpotPreflight:
    def setup_method(self):
        self.provider = RunPodSpotProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="L40S",
            service_name="test-svc",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_preflight_no_api_key(self):
        result = self.provider.preflight(self.request)
        assert not result.ok
        assert any("RUNPOD_API_KEY" in c.message for c in result.checks)

    @patch("tuna.spot.runpod_spot.requests.get")
    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    def test_preflight_valid_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = self.provider.preflight(self.request)
        assert result.ok

    @patch("tuna.spot.runpod_spot.requests.get")
    @patch.dict("os.environ", {"RUNPOD_API_KEY": "bad-key"})
    def test_preflight_invalid_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = self.provider.preflight(self.request)
        assert not result.ok
        assert any("invalid" in c.message.lower() for c in result.checks)


class TestRunPodSpotPlan:
    def setup_method(self):
        self.provider = RunPodSpotProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="A100_80GB",
            service_name="test-svc",
        )
        self.vllm_cmd = (
            "vllm serve Qwen/Qwen3-0.6B "
            "--host 0.0.0.0 --port 8001 --max-model-len 4096 "
            "--served-model-name Qwen/Qwen3-0.6B --tensor-parallel-size 1 "
            "--disable-log-requests --uvicorn-log-level info --enforce-eager"
        )

    def test_plan_provider(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.provider == "runpod-spot"

    def test_plan_service_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["service_name"] == "test-svc-spot"

    def test_plan_gpu_type(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_type_id"] == "NVIDIA A100-SXM4-80GB"

    def test_plan_gpu_count(self):
        self.request.gpu_count = 4
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_count"] == "4"

    def test_plan_docker_cmd_contains_vllm(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "vllm serve" in plan.metadata["docker_cmd"]
        assert "Qwen/Qwen3-0.6B" in plan.metadata["docker_cmd"]

    def test_plan_unknown_gpu_raises(self):
        self.request.gpu = "UNKNOWN_GPU"
        with pytest.raises(ValueError, match="Unknown GPU type for RunPod"):
            self.provider.plan(self.request, self.vllm_cmd)


class TestRunPodSpotDeploy:
    def setup_method(self):
        self.provider = RunPodSpotProvider()

    @patch("tuna.spot.runpod_spot.RunPodSpotProvider._wait_for_pod_ready",
           return_value="https://abc123-8001.proxy.runpod.net")
    @patch("tuna.spot.runpod_spot.requests.post")
    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    def test_deploy_success(self, mock_post, mock_wait):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"id": "abc123"}
        mock_post.return_value = mock_resp

        plan = ProviderPlan(
            provider="runpod-spot",
            rendered_script="",
            env={"HF_TOKEN": ""},
            metadata={
                "service_name": "test-svc-spot",
                "gpu_type_id": "NVIDIA A100-SXM4-80GB",
                "gpu_count": "1",
                "docker_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                "docker_cmd": "pip install vllm && vllm serve ...",
                "port": "8001",
            },
        )
        result = self.provider.deploy(plan)

        assert result.error is None
        assert result.endpoint_url == "https://abc123-8001.proxy.runpod.net"
        assert result.metadata["pod_id"] == "abc123"

    @patch("tuna.spot.runpod_spot.requests.post", side_effect=RuntimeError("API down"))
    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    def test_deploy_api_failure(self, mock_post):
        plan = ProviderPlan(
            provider="runpod-spot",
            rendered_script="",
            env={},
            metadata={
                "service_name": "test-svc-spot",
                "gpu_type_id": "NVIDIA A100-SXM4-80GB",
                "gpu_count": "1",
                "docker_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                "docker_cmd": "vllm serve ...",
                "port": "8001",
            },
        )
        result = self.provider.deploy(plan)
        assert result.error is not None
        assert "Pod creation failed" in result.error

    @patch.dict("os.environ", {}, clear=True)
    def test_deploy_no_api_key(self):
        plan = ProviderPlan(
            provider="runpod-spot",
            rendered_script="",
            env={},
            metadata={
                "service_name": "test-svc-spot",
                "gpu_type_id": "NVIDIA A100-SXM4-80GB",
                "gpu_count": "1",
                "docker_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                "docker_cmd": "vllm serve ...",
                "port": "8001",
            },
        )
        result = self.provider.deploy(plan)
        assert result.error is not None
        assert "RUNPOD_API_KEY" in result.error


class TestRunPodSpotDestroy:
    def setup_method(self):
        self.provider = RunPodSpotProvider()

    @patch("tuna.spot.runpod_spot.requests.delete")
    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    def test_destroy_calls_delete(self, mock_delete):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_delete.return_value = mock_resp

        result = DeploymentResult(
            provider="runpod-spot",
            metadata={"pod_id": "abc123", "service_name": "test-svc-spot"},
        )
        self.provider.destroy(result)

        mock_delete.assert_called_once()
        assert "abc123" in mock_delete.call_args[0][0]

    def test_destroy_no_pod_id(self):
        result = DeploymentResult(provider="runpod-spot", metadata={})
        # Should not raise
        self.provider.destroy(result)


class TestRunPodSpotStatus:
    def setup_method(self):
        self.provider = RunPodSpotProvider()

    @patch("tuna.spot.runpod_spot.requests.get")
    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    def test_status_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {
                "name": "my-svc-spot",
                "id": "pod123",
                "desiredStatus": "RUNNING",
                "runtime": {"gpus": [{"publicIp": "1.2.3.4"}]},
            }
        ]
        mock_get.return_value = mock_resp

        result = self.provider.status("my-svc")
        assert result["provider"] == "runpod-spot"
        assert result["status"] == "running"
        assert result["pod_id"] == "pod123"

    @patch("tuna.spot.runpod_spot.requests.get")
    @patch.dict("os.environ", {"RUNPOD_API_KEY": "test-key"})
    def test_status_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        result = self.provider.status("my-svc")
        assert result["status"] == "not found"
