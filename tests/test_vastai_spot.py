"""Tests for tuna.spot.vastai_spot — plan/deploy/destroy, no real API calls."""

from unittest.mock import MagicMock, patch

import pytest

from tuna.models import DeployRequest, DeploymentResult, ProviderPlan
from tuna.providers.base import InferenceProvider
from tuna.scaling import ScalingPolicy, ServerlessScaling, SpotScaling
from tuna.spot.vastai_spot import VastAISpotProvider


class TestVastAISpotProvider:
    def test_is_inference_provider(self):
        assert issubclass(VastAISpotProvider, InferenceProvider)
        assert isinstance(VastAISpotProvider(), InferenceProvider)

    def test_name(self):
        provider = VastAISpotProvider()
        assert provider.name() == "vastai-spot"


class TestVastAISpotPreflight:
    def setup_method(self):
        self.provider = VastAISpotProvider()
        self.request = DeployRequest(
            model_name="Qwen/Qwen3-0.6B",
            gpu="A100_80GB",
            service_name="test-svc",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_preflight_no_api_key(self):
        result = self.provider.preflight(self.request)
        assert not result.ok
        assert any("VASTAI_API_KEY" in c.message for c in result.checks)

    @patch("tuna.spot.vastai_spot.requests.get")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_preflight_valid_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = self.provider.preflight(self.request)
        assert result.ok

    @patch("tuna.spot.vastai_spot.requests.get")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "bad-key"})
    def test_preflight_invalid_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = self.provider.preflight(self.request)
        assert not result.ok

    @patch("tuna.spot.vastai_spot.requests.get")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_preflight_unsupported_gpu(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        self.request.gpu = "UNKNOWN_GPU"
        result = self.provider.preflight(self.request)
        assert not result.ok
        assert any("not mapped" in c.message.lower() for c in result.checks)


class TestVastAISpotPlan:
    def setup_method(self):
        self.provider = VastAISpotProvider()
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
        assert plan.provider == "vastai-spot"

    def test_plan_service_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["service_name"] == "test-svc-spot"

    def test_plan_gpu_name(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_name"] == "A100_SXM4"

    def test_plan_gpu_count(self):
        self.request.gpu_count = 4
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_count"] == "4"

    def test_plan_onstart_contains_vllm(self):
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert "vllm serve" in plan.metadata["onstart_cmd"]
        assert "Qwen/Qwen3-0.6B" in plan.metadata["onstart_cmd"]

    def test_plan_unknown_gpu_raises(self):
        self.request.gpu = "UNKNOWN_GPU"
        with pytest.raises(ValueError, match="Unknown GPU type for Vast.ai"):
            self.provider.plan(self.request, self.vllm_cmd)

    def test_plan_rtx4090(self):
        self.request.gpu = "RTX4090"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_name"] == "RTX 4090"

    def test_plan_h100(self):
        self.request.gpu = "H100"
        plan = self.provider.plan(self.request, self.vllm_cmd)
        assert plan.metadata["gpu_name"] == "H100"


class TestVastAISpotDeploy:
    def setup_method(self):
        self.provider = VastAISpotProvider()
        self.plan = ProviderPlan(
            provider="vastai-spot",
            rendered_script="",
            env={"HF_TOKEN": ""},
            metadata={
                "service_name": "test-svc-spot",
                "gpu_name": "A100_SXM4",
                "gpu_count": "1",
                "docker_image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel",
                "onstart_cmd": "pip install vllm && vllm serve ...",
                "port": "8001",
                "disk_gb": "100",
            },
        )

    @patch("tuna.spot.vastai_spot.VastAISpotProvider._wait_for_instance_ready",
           return_value="http://1.2.3.4:40001")
    @patch("tuna.spot.vastai_spot.VastAISpotProvider._find_best_offer",
           return_value="12345")
    @patch("tuna.spot.vastai_spot.requests.put")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_deploy_success(self, mock_put, mock_find, mock_wait):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": True, "new_contract": 67890}
        mock_put.return_value = mock_resp

        result = self.provider.deploy(self.plan)

        assert result.error is None
        assert result.endpoint_url == "http://1.2.3.4:40001"
        assert result.metadata["instance_id"] == "67890"

    @patch("tuna.spot.vastai_spot.VastAISpotProvider._find_best_offer",
           return_value=None)
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_deploy_no_offers(self, mock_find):
        result = self.provider.deploy(self.plan)
        assert result.error is not None
        assert "No Vast.ai offers" in result.error

    @patch("tuna.spot.vastai_spot.VastAISpotProvider._find_best_offer",
           return_value="12345")
    @patch("tuna.spot.vastai_spot.requests.put")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_deploy_creation_failure(self, mock_put, mock_find):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": False, "msg": "Insufficient funds"}
        mock_put.return_value = mock_resp

        result = self.provider.deploy(self.plan)
        assert result.error is not None
        assert "Insufficient funds" in result.error

    @patch.dict("os.environ", {}, clear=True)
    def test_deploy_no_api_key(self):
        result = self.provider.deploy(self.plan)
        assert result.error is not None
        assert "VASTAI_API_KEY" in result.error


class TestVastAISpotDestroy:
    def setup_method(self):
        self.provider = VastAISpotProvider()

    @patch("tuna.spot.vastai_spot.requests.delete")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_destroy_calls_delete(self, mock_delete):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_delete.return_value = mock_resp

        result = DeploymentResult(
            provider="vastai-spot",
            metadata={"instance_id": "67890", "service_name": "test-svc-spot"},
        )
        self.provider.destroy(result)

        mock_delete.assert_called_once()
        assert "67890" in mock_delete.call_args[0][0]

    def test_destroy_no_instance_id(self):
        result = DeploymentResult(provider="vastai-spot", metadata={})
        # Should not raise
        self.provider.destroy(result)


class TestVastAISpotStatus:
    def setup_method(self):
        self.provider = VastAISpotProvider()

    @patch("tuna.spot.vastai_spot.requests.get")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_status_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "instances": [
                {
                    "label": "my-svc-spot",
                    "id": 67890,
                    "actual_status": "running",
                    "public_ipaddr": "1.2.3.4",
                    "ports": {
                        "8001/tcp": [{"HostPort": "40001"}],
                    },
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = self.provider.status("my-svc")
        assert result["provider"] == "vastai-spot"
        assert result["status"] == "running"
        assert result["instance_id"] == "67890"
        assert result["endpoint"] == "http://1.2.3.4:40001"

    @patch("tuna.spot.vastai_spot.requests.get")
    @patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"})
    def test_status_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"instances": []}
        mock_get.return_value = mock_resp

        result = self.provider.status("my-svc")
        assert result["status"] == "not found"

    @patch.dict("os.environ", {}, clear=True)
    def test_status_no_api_key(self):
        result = self.provider.status("my-svc")
        assert result["status"] == "unknown"
        assert "VASTAI_API_KEY" in result["error"]
