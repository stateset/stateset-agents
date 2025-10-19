"""
Integration Test: Verify HuggingFace and W&B Integration

This script tests the complete integration pipeline to ensure seamless
compatibility with HuggingFace transformers and Weights & Biases.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import torch

# Framework imports
from stateset_agents import (
    CompositeReward,
    ConversationEnvironment,
    HelpfulnessReward,
    MultiTurnAgent,
    MultiTurnGRPOTrainer,
    SafetyReward,
    TrainingConfig,
    TrainingProfile,
    WandBLogger,
)
from stateset_agents.core.agent import (
    AgentConfig,
    create_agent,
    create_peft_agent,
    load_agent_from_checkpoint,
    save_agent_checkpoint,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Comprehensive integration testing"""

    def __init__(self):
        self.temp_dir = None
        self.test_results = []

    def setup_temp_directory(self):
        """Setup temporary directory for testing"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="grpo_test_"))
        logger.info(f"Using temporary directory: {self.temp_dir}")

    def cleanup_temp_directory(self):
        """Cleanup temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.info("Temporary directory cleaned up")

    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  Details: {details}")

        self.test_results.append(
            {"test": test_name, "success": success, "details": details}
        )

    async def test_huggingface_model_loading(self):
        """Test HuggingFace model loading with various configurations"""
        logger.info("\n🧪 Testing HuggingFace Model Loading")

        try:
            # Test 1: Basic model loading
            config = AgentConfig(
                model_name="openai/gpt-oss-120b",
                torch_dtype="float32",  # Use float32 for compatibility
                device_map=None,  # Disable auto device mapping for testing
                attn_implementation=None,  # Use default attention
            )

            agent = MultiTurnAgent(config)
            await agent.initialize()

            assert agent.model is not None, "Model not loaded"
            assert agent.tokenizer is not None, "Tokenizer not loaded"

            self.log_test_result(
                "Basic HuggingFace model loading",
                True,
                f"Model: {config.model_name}, dtype: {agent.model.dtype}",
            )

        except Exception as e:
            self.log_test_result("Basic HuggingFace model loading", False, str(e))

    async def test_peft_integration(self):
        """Test PEFT/LoRA integration"""
        logger.info("\n🧪 Testing PEFT/LoRA Integration")

        try:
            # Skip if PEFT not available
            try:
                import peft
            except ImportError:
                self.log_test_result(
                    "PEFT integration", False, "PEFT library not available"
                )
                return

            # Test PEFT agent creation
            peft_config = {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["c_attn"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }

            agent = create_peft_agent(
                model_name="openai/gpt-oss-120b",
                peft_config=peft_config,
                torch_dtype="float32",
                device_map=None,
            )

            await agent.initialize()

            # Verify PEFT is applied
            assert hasattr(agent.model, "peft_config"), "PEFT not applied"

            self.log_test_result(
                "PEFT/LoRA integration",
                True,
                f"LoRA rank: {peft_config['r']}, alpha: {peft_config['lora_alpha']}",
            )

        except Exception as e:
            self.log_test_result("PEFT/LoRA integration", False, str(e))

    async def test_model_checkpoint_saving_loading(self):
        """Test model checkpoint saving and loading"""
        logger.info("\n🧪 Testing Model Checkpoint Saving/Loading")

        try:
            # Create and initialize agent
            agent = create_agent(
                agent_type="multi_turn",
                model_name="openai/gpt-oss-120b",
                torch_dtype="float32",
                device_map=None,
            )
            await agent.initialize()

            # Save checkpoint
            checkpoint_path = self.temp_dir / "test_checkpoint"
            await save_agent_checkpoint(agent, str(checkpoint_path), save_model=True)

            # Verify files exist
            assert (checkpoint_path / "config.json").exists(), "Config file not saved"
            assert (checkpoint_path / "pytorch_model.bin").exists() or (
                checkpoint_path / "model.safetensors"
            ).exists(), "Model file not saved"

            # Load checkpoint
            loaded_agent = await load_agent_from_checkpoint(
                str(checkpoint_path), load_model=True
            )

            assert loaded_agent.model is not None, "Model not loaded from checkpoint"
            assert (
                loaded_agent.tokenizer is not None
            ), "Tokenizer not loaded from checkpoint"

            self.log_test_result(
                "Model checkpoint saving/loading",
                True,
                f"Checkpoint saved to: {checkpoint_path}",
            )

        except Exception as e:
            self.log_test_result("Model checkpoint saving/loading", False, str(e))

    def test_wandb_integration(self):
        """Test W&B integration"""
        logger.info("\n🧪 Testing W&B Integration")

        try:
            # Test W&B logger creation
            wandb_logger = WandBLogger(
                project="grpo-test",
                enabled=False,  # Disable actual W&B logging for test
            )

            # Test configuration preparation
            config = {
                "model_name": "openai/gpt-oss-120b",
                "learning_rate": 5e-6,
                "num_episodes": 100,
            }

            # Test metric logging (should not fail even with W&B disabled)
            wandb_logger.log_metrics({"test_metric": 0.5}, step=1)

            self.log_test_result(
                "W&B integration setup", True, "W&B logger created and configured"
            )

            # Test with W&B enabled if API key available
            if os.getenv("WANDB_API_KEY"):
                enabled_logger = WandBLogger(
                    project="grpo-integration-test", enabled=True
                )

                # Quick init and finish test
                enabled_logger.init_run(
                    config=config, name="integration-test", tags=["test"]
                )

                enabled_logger.log_metrics({"test_metric": 0.8}, step=1)
                enabled_logger.finish_run({"test_completed": True})

                self.log_test_result(
                    "W&B live integration", True, "Successfully logged to W&B"
                )
            else:
                self.log_test_result(
                    "W&B live integration", False, "WANDB_API_KEY not available"
                )

        except Exception as e:
            self.log_test_result("W&B integration", False, str(e))

    def test_training_config(self):
        """Test training configuration"""
        logger.info("\n🧪 Testing Training Configuration")

        try:
            # Test profile-based configuration
            config = TrainingConfig.from_profile(
                TrainingProfile.BALANCED,
                num_episodes=50,
                output_dir=str(self.temp_dir / "training_output"),
            )

            # Test validation
            warnings = config.validate()

            # Test serialization
            config_path = self.temp_dir / "test_config.json"
            config.save(str(config_path))

            # Test loading
            loaded_config = TrainingConfig.load(str(config_path))

            assert (
                loaded_config.learning_rate == config.learning_rate
            ), "Config loading failed"

            self.log_test_result(
                "Training configuration",
                True,
                f"Profile: {TrainingProfile.BALANCED.value}, warnings: {len(warnings)}",
            )

        except Exception as e:
            self.log_test_result("Training configuration", False, str(e))

    async def test_end_to_end_training(self):
        """Test end-to-end training pipeline"""
        logger.info("\n🧪 Testing End-to-End Training Pipeline")

        try:
            # Create minimal agent
            agent = create_agent(
                agent_type="multi_turn",
                model_name="openai/gpt-oss-120b",
                torch_dtype="float32",
                device_map=None,
            )
            await agent.initialize()

            # Create simple environment
            scenarios = [
                {
                    "id": "test_scenario",
                    "context": "Test conversation",
                    "user_responses": ["Hello!", "How are you?", "Thanks!"],
                }
            ]

            environment = ConversationEnvironment(scenarios=scenarios, max_turns=4)

            # Create simple reward
            reward_fn = HelpfulnessReward(weight=1.0)

            # Create minimal training config
            config = TrainingConfig(
                num_episodes=2,  # Very small for test
                num_generations=2,
                gradient_accumulation_steps=1,
                output_dir=str(self.temp_dir / "training_test"),
                logging_steps=1,
                eval_steps=1,
                save_steps=1,
                report_to="none",  # Disable external logging
            )

            # Create trainer
            trainer = MultiTurnGRPOTrainer(
                agent=agent,
                environment=environment,
                reward_fn=reward_fn,
                config=config,
                wandb_logger=None,  # Disable W&B for test
            )

            await trainer.initialize()

            # Run minimal training
            trained_agent = await trainer.train()

            assert trained_agent is not None, "Training failed to return agent"
            assert trainer.global_step > 0, "No training steps completed"

            self.log_test_result(
                "End-to-end training pipeline",
                True,
                f"Completed {trainer.global_step} training steps",
            )

        except Exception as e:
            self.log_test_result("End-to-end training pipeline", False, str(e))

    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting GRPO Framework Integration Tests")
        logger.info("=" * 60)

        self.setup_temp_directory()

        try:
            # Run all tests
            await self.test_huggingface_model_loading()
            await self.test_peft_integration()
            await self.test_model_checkpoint_saving_loading()
            self.test_wandb_integration()
            self.test_training_config()
            await self.test_end_to_end_training()

        finally:
            self.cleanup_temp_directory()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 Integration Test Summary")
        logger.info("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests

        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

        if failed_tests > 0:
            logger.info("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    logger.info(f"  - {result['test']}: {result['details']}")

        logger.info("\n✅ Integration Status:")

        # Check critical integrations
        hf_working = any(
            "HuggingFace" in r["test"] and r["success"] for r in self.test_results
        )

        config_working = any(
            "configuration" in r["test"] and r["success"] for r in self.test_results
        )

        training_working = any(
            "end-to-end" in r["test"] and r["success"] for r in self.test_results
        )

        logger.info(f"  HuggingFace Integration: {'✅' if hf_working else '❌'}")
        logger.info(f"  Training Configuration: {'✅' if config_working else '❌'}")
        logger.info(f"  End-to-End Training: {'✅' if training_working else '❌'}")

        # Overall status
        if passed_tests == total_tests:
            logger.info("\n🎉 All integrations working perfectly!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("\n✅ Core integrations working, minor issues detected")
        else:
            logger.info("\n⚠️ Critical integration issues detected")

        logger.info("\n📋 Next Steps:")
        logger.info("1. Check failed tests and resolve issues")
        logger.info("2. Verify all dependencies are installed")
        logger.info("3. Set WANDB_API_KEY for full W&B integration")
        logger.info("4. Run the examples to verify functionality")


async def main():
    """Run integration tests"""
    tester = IntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Set environment for testing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run tests
    asyncio.run(main())
