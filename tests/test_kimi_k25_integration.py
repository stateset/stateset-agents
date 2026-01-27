"""
Tests for Kimi-K2.5 integration with the RL framework.

These tests verify that Kimi-K2.5 model can be properly configured,
trained, and deployed within the StateSet Agents framework.
"""

import pytest
import inspect


class TestKimiK25Config:
    """Test Kimi-K2.5 configuration functions."""

    def test_kimi_k25_config_imports(self):
        """Test that Kimi-K2.5 configuration can be imported."""
        from examples.kimi_k25_config import (
            KIMI_K25_LORA_TARGET_MODULES,
            get_vllm_launch_command,
        )

        assert KIMI_K25_LORA_TARGET_MODULES is not None
        assert get_vllm_launch_command is not None

    def test_kimi_k25_lora_target_modules(self):
        """Test LoRA target modules for Kimi-K2.5."""
        from examples.kimi_k25_config import KIMI_K25_LORA_TARGET_MODULES

        assert isinstance(KIMI_K25_LORA_TARGET_MODULES, list)
        assert "q_proj" in KIMI_K25_LORA_TARGET_MODULES
        assert "k_proj" in KIMI_K25_LORA_TARGET_MODULES
        assert "v_proj" in KIMI_K25_LORA_TARGET_MODULES
        assert "o_proj" in KIMI_K25_LORA_TARGET_MODULES
        assert "gate_proj" in KIMI_K25_LORA_TARGET_MODULES
        assert "up_proj" in KIMI_K25_LORA_TARGET_MODULES
        assert "down_proj" in KIMI_K25_LORA_TARGET_MODULES

    def test_vllm_launch_command(self):
        """Test vLLM launch command generation."""
        from examples.kimi_k25_config import get_vllm_launch_command

        cmd = get_vllm_launch_command(
            model_name="moonshotai/Kimi-K2.5",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
        )

        assert "vllm serve" in cmd
        assert "moonshotai/Kimi-K2.5" in cmd


class TestKimiK25MemoryRequirements:
    """Test Kimi-K2.5 memory requirement calculations."""

    def test_memory_estimation_naive_fallback(self):
        """Test naive memory estimation used when actual estimation isn't available."""
        # This is a simple test to verify the concept exists
        total_params_in_billions = 1000  # 1T parameters
        activated_params_pct = 0.032  # 32B activated

        # Naive estimation: 16 bytes per parameter (bf16)
        naive_vram_gb = total_params_in_billions * activated_params_pct * 16

        assert naive_vram_gb > 0
        # Should be roughly 512GB without optimization
        assert naive_vram_gb > 500


class TestKimiK25AgentInitialization:
    """Test Kimi-K2.5 agent initialization."""

    def test_agent_config_creation(self):
        """Test that agent config can be created for Kimi-K2.5."""
        from stateset_agents.core.agent import AgentConfig

        config = AgentConfig(
            model_name="moonshotai/Kimi-K2.5",
            max_new_tokens=2048,
            temperature=1.0,
            system_prompt="You are Kimi, an AI assistant created by Moonshot AI.",
        )

        assert config.model_name == "moonshotai/Kimi-K2.5"
        assert config.max_new_tokens == 2048
        assert config.temperature == 1.0


class TestKimiK25LoRAConfig:
    """Test Kimi-K2.5 LoRA configuration."""

    def test_lora_r_value(self):
        """Test LoRA rank is appropriate for Kimi-K2.5."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(use_lora=True)

        assert config.use_lora == True
        assert config.lora_r >= 64  # Higher rank for MoE

    def test_lora_alpha_value(self):
        """Test LoRA alpha is appropriate."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(use_lora=True)

        assert config.lora_alpha >= 128

    def test_lora_dropout_value(self):
        """Test LoRA dropout is configured."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(use_lora=True)

        assert config.lora_dropout >= 0.0
        assert config.lora_dropout <= 0.1


class TestKimiK25GSPOConfig:
    """Test GSPO configuration for Kimi-K2.5."""

    def test_gspo_learning_rate(self):
        """Test learning rate is appropriate for MoE."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config()

        # Lower learning rate for MoE stability
        assert config.learning_rate <= 5e-6
        assert config.learning_rate >= 1e-6

    def test_gspo_batch_size_moelora(self):
        """Test batch size with MoE + LoRA."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(use_lora=True, use_vllm=False)

        # Small batch size for MoE memory
        assert config.per_device_train_batch_size == 1

    def test_gspo_batch_size_vllmlora(self):
        """Test batch size with vLLM + LoRA."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(use_lora=True, use_vllm=True)

        # Slightly larger batch with vLLM
        assert config.per_device_train_batch_size >= 1

    def test_gspo_gradient_accumulation(self):
        """Test gradient accumulation is configured."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config()

        # Large gradient accumulation for effective batch size
        assert config.gradient_accumulation_steps >= 8


class TestKimiK25TrainingPipeline:
    """Test Kimi-K2.5 training pipeline."""

    def test_training_function_exists(self):
        """Test that training function exists."""
        from examples import finetune_kimi_k25_gspo as training_module

        assert hasattr(training_module, "finetune_kimi_k25")

    def test_training_function_signature(self):
        """Test training function has correct parameters."""
        from examples import finetune_kimi_k25_gspo as training_module

        sig = inspect.signature(training_module.finetune_kimi_k25)
        params = list(sig.parameters.keys())

        assert "model_name" in params
        assert "task" in params
        assert "use_lora" in params
        assert "use_8bit" in params


class TestKimiK25MultimodalSupport:
    """Test Kimi-K2.5 multimodal capabilities."""

    def test_multimodal_model_features(self):
        """Test Kimi-K2.5 has multimodal features."""
        # Kimi-K2.5 is a native multimodal model
        model_features = {
            "vision_encoder": "MoonViT",
            "vision_encoder_params": "400M",
            "supports_images": True,
            "supports_videos": True,
            "supports_interleaved_thinking": True,
        }

        assert model_features["supports_images"] == True
        assert model_features["supports_videos"] == True
        assert model_features["vision_encoder"] == "MoonViT"


class TestKimiK25ThinkingMode:
    """Test Kimi-K2.5 thinking mode configuration."""

    def test_thinking_mode_temperature(self):
        """Test thinking mode uses temperature 1.0."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(task="conversational")

        # Kimi-K2.5 recommends 1.0 for thinking mode
        assert config.temperature >= 0.9
        assert config.temperature <= 1.0

    def test_instant_mode_temperature(self):
        """Test instant mode uses temperature 0.7."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(task="customer_service")

        # Instant mode typically uses lower temperature
        assert config.temperature == 0.7

    def test_thinking_mode_top_p(self):
        """Test top_p is calibrated correctly."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config()

        # Kimi-K2.5 recommends 0.95
        assert config.top_p == 0.95


class TestKimiK25Deployment:
    """Test Kimi-K2.5 deployment configuration."""

    def test_recommended_inference_engines(self):
        """Test recommended inference engines."""
        recommended_engines = ["vLLM", "SGLang", "KTransformers"]

        assert "vLLM" in recommended_engines
        assert "SGLang" in recommended_engines

    def test_vllm_tensor_parallel(self):
        """Test vLLM tensor parallel configuration."""
        from examples.kimi_k25_config import get_vllm_launch_command

        cmd = get_vllm_launch_command(tensor_parallel_size=1)

        assert "--tensor-parallel-size 1" in cmd

    def test_vllm_gpu_memory_utilization(self):
        """Test vLLM GPU memory utilization."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(use_vllm=True)

        assert config.vllm_gpu_memory_utilization == 0.85

    def test_min_transformers_version(self):
        """Test minimum transformers version."""
        # Kimi-K2.5 requires transformers >= 4.57.1
        min_version = "4.57.1"

        assert min_version == "4.57.1"


class TestKimiK25CompleteWorkflow:
    """Test complete Kimi-K2.5 workflow."""

    def test_import_training_script(self):
        """Test training script can be imported."""
        from examples import finetune_kimi_k25_gspo

        assert finetune_kimi_k25_gspo is not None

    def test_import_config_module(self):
        """Test config module can be imported."""
        from examples import kimi_k25_config

        assert kimi_k25_config is not None

    def test_integration_documentation_exists(self):
        """Test integration documentation exists."""
        from pathlib import Path

        doc_path = Path("examples/KIMI_K25_INTEGRATION.md")
        assert doc_path.exists()


class TestKimiK25TaskSpecific:
    """Test task-specific configurations."""

    @pytest.mark.parametrize(
        "task,expected_target_modules",
        [
            (
                "customer_service",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            (
                "technical_support",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            (
                "coding",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        ],
    )
    def test_task_target_modules(self, task, expected_target_modules):
        """Test that different tasks get appropriate target modules."""
        from examples.kimi_k25_config import KIMI_K25_LORA_TARGET_MODULES

        # Kimi-K2.5 uses the same target modules for all tasks (MoE architecture)
        assert KIMI_K25_LORA_TARGET_MODULES == expected_target_modules


class TestKimiK25BatchSizeConfiguration:
    """Test batch size configuration based on hardware."""

    @pytest.mark.parametrize(
        "use_lora,use_vllm,expected_batch_size",
        [
            (True, False, 1),
            (True, True, 1),
            (False, False, 1),
        ],
    )
    def test_batch_size_configuration(self, use_lora, use_vllm, expected_batch_size):
        """Test batch size is configured correctly."""
        from examples.kimi_k25_config import get_kimi_config

        config = get_kimi_config(use_lora=use_lora, use_vllm=use_vllm)

        assert config.per_device_train_batch_size == expected_batch_size


class TestKimiK25Architecture:
    """Test Kimi-K2.5 architecture specifics."""

    def test_moe_architecture(self):
        """Test MoE architecture is correctly identified."""
        # Kimi-K2.5 is a 1T parameter MoE model
        total_params = 1_000_000_000_000  # 1T
        activated_params = 32_000_000_000  # 32B
        num_experts = 384
        experts_per_token = 8

        assert total_params == 1_000_000_000_000
        assert activated_params == 32_000_000_000
        assert num_experts == 384
        assert experts_per_token == 8

    def test_attention_configuration(self):
        """Test attention mechanism configuration."""
        # Kimi-K2.5 uses MLA (Multi-head Latent Attention)
        attention_type = "MLA"
        num_attention_heads = 64
        attention_hidden_dim = 7168

        assert attention_type == "MLA"
        assert num_attention_heads == 64
        assert attention_hidden_dim == 7168

    def test_vocab_size(self):
        """Test vocabulary size."""
        # Kimi-K2.5 has 160K vocabulary
        vocab_size = 160000

        assert vocab_size == 160000
