"""
Test utilities for Kimi-K2.5 integration

This module provides test helpers and utilities for validating Kimi-K2.5
functionality within the GRPO Agent Framework.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def test_kimi_k25_model_loading(
    model_name: str = "moonshotai/Kimi-K2.5",
) -> Dict[str, Any]:
    """
    Test if Kimi-K2.5 model can be loaded and initialized.

    Args:
        model_name: Model identifier

    Returns:
        Dictionary with test results
    """
    try:
        from transformers import AutoTokenizer

        logger.info(f"Testing model loading: {model_name}")

        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"✅ Tokenizer loaded successfully")
        logger.info(f"   - Vocabulary size: {len(tokenizer)}")
        logger.info(f"   - Max model length: {tokenizer.model_max_length}")

        # Test tokenization
        test_text = "Hello, this is a test for Kimi-K2.5."
        tokens = tokenizer(test_text, return_tensors="pt")
        logger.info(f"✅ Tokenization successful: {len(tokens['input_ids'][0])} tokens")

        return {
            "success": True,
            "model_name": model_name,
            "vocab_size": len(tokenizer),
            "max_length": tokenizer.model_max_length,
            "tokenizer_test_passed": True,
        }

    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return {
            "success": False,
            "model_name": model_name,
            "error": str(e),
        }


async def test_kimi_k25_agent_creation(
    model_name: str = "moonshotai/Kimi-K2.5",
    system_prompt: str = "You are Kimi, an AI assistant created by Moonshot AI.",
) -> Dict[str, Any]:
    """
    Test creating a MultiTurnAgent with Kimi-K2.5.

    Args:
        model_name: Model identifier
        system_prompt: System prompt for the agent

    Returns:
        Dictionary with test results
    """
    try:
        from stateset_agents import MultiTurnAgent
        from stateset_agents.core.agent import AgentConfig

        logger.info(f"Testing agent creation with {model_name}")

        # Create agent configuration
        agent_config = AgentConfig(
            model_name=model_name,
            system_prompt=system_prompt,
            max_new_tokens=2048,
            temperature=0.7,
        )

        # Create agent (without initializing to avoid loading large model)
        from stateset_agents.core.agent import Agent

        agent = Agent(agent_config)

        logger.info(f"✅ Agent created successfully")
        logger.info(f"   - Model: {agent.config.model_name}")
        logger.info(f"   - System prompt: {agent.config.system_prompt[:50]}...")

        return {
            "success": True,
            "model_name": model_name,
            "agent_created": True,
            "config": {
                "max_new_tokens": agent.config.max_new_tokens,
                "temperature": agent.config.temperature,
            },
        }

    except Exception as e:
        logger.error(f"❌ Agent creation test failed: {e}")
        return {
            "success": False,
            "model_name": model_name,
            "error": str(e),
        }


async def test_kimi_k25_config(
    model_name: str = "moonshotai/Kimi-K2.5",
    task: str = "customer_service",
) -> Dict[str, Any]:
    """
    Test Kimi-K2.5 GSPO configuration.

    Args:
        model_name: Model identifier
        task: Task type

    Returns:
        Dictionary with test results
    """
    try:
        import sys
        from pathlib import Path

        # Add examples directory to path
        examples_dir = Path(__file__).parent.parent / "examples"
        sys.path.insert(0, str(examples_dir))

        from examples.kimi_k25_config import get_kimi_k25_config

        logger.info(f"Testing Kimi-K2.5 GSPO configuration")

        # Get configuration
        config = get_kimi_k25_config(
            model_name=model_name,
            task=task,
            use_lora=True,
            use_4bit=False,
            use_8bit=False,
        )

        logger.info(f"✅ Configuration created successfully")
        logger.info(f"   - Model: {config.model_name}")
        logger.info(f"   - Learning rate: {config.learning_rate}")
        logger.info(f"   - LoRA enabled: {config.use_lora}")
        logger.info(f"   - Group size: {getattr(config, 'num_generations', 'N/A')}")

        return {
            "success": True,
            "config_valid": True,
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "use_lora": config.use_lora,
            "use_8bit": config.use_8bit,
            "use_4bit": config.use_4bit,
        }

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return {
            "success": False,
            "model_name": model_name,
            "error": str(e),
        }


async def test_kimi_k25_vision_support(
    model_name: str = "moonshotai/Kimi-K2.5",
) -> Dict[str, Any]:
    """
    Test Kimi-K2.5's vision capabilities.

    Args:
        model_name: Model identifier

    Returns:
        Dictionary with test results
    """
    try:
        from transformers import AutoTokenizer

        logger.info(f"Testing vision support for {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Check if tokenizer supports image tokens
        special_tokens = tokenizer.special_tokens_map
        image_token = special_tokens.get("image_token", None)

        has_vision_support = image_token is not None or "<image>" in str(
            tokenizer.added_tokens_decoder
        )

        logger.info(f"✅ Vision support test completed")
        logger.info(f"   - Vision supported: {has_vision_support}")
        logger.info(f"   - Image token: {image_token}")

        return {
            "success": True,
            "has_vision_support": has_vision_support,
            "image_token": image_token,
        }

    except Exception as e:
        logger.error(f"❌ Vision support test failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def test_kimi_k25_thinking_mode(
    model_name: str = "moonshotai/Kimi-K2.5",
) -> Dict[str, Any]:
    """
    Test Kimi-K2.5's thinking mode configuration.

    Args:
        model_name: Model identifier

    Returns:
        Dictionary with test results
    """
    try:
        from transformers import AutoConfig

        logger.info(f"Testing thinking mode support for {model_name}")

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Check for thinking-related settings
        has_thinking_mode = hasattr(config, "thinking_enabled") or hasattr(
            config, "use_thinking"
        )

        logger.info(f"✅ Thinking mode test completed")
        logger.info(f"   - Thinking mode supported: {has_thinking_mode}")

        return {
            "success": True,
            "has_thinking_mode": has_thinking_mode,
        }

    except Exception as e:
        logger.error(f"❌ Thinking mode test failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def run_all_kimi_k25_tests(
    model_name: str = "moonshotai/Kimi-K2.5",
) -> Dict[str, Any]:
    """
    Run all Kimi-K2.5 integration tests.

    Args:
        model_name: Model identifier

    Returns:
        Dictionary with all test results
    """
    logger.info("=" * 80)
    logger.info("Running Kimi-K2.5 Integration Tests")
    logger.info("=" * 80)

    tests = [
        ("Model Loading", lambda: test_kimi_k25_model_loading(model_name)),
        ("Agent Creation", lambda: test_kimi_k25_agent_creation(model_name)),
        ("Configuration", lambda: test_kimi_k25_config(model_name, "customer_service")),
        ("Vision Support", lambda: test_kimi_k25_vision_support(model_name)),
        ("Thinking Mode", lambda: test_kimi_k25_thinking_mode(model_name)),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running: {test_name}")
        logger.info("=" * 80)
        result = await test_func()
        results[test_name] = result

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Test Summary")
    logger.info("=" * 80)

    passed = sum(1 for r in results.values() if r.get("success", False))
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
        },
        "tests": results,
    }


def main():
    """Run all tests and print results"""
    import asyncio

    results = asyncio.run(run_all_kimi_k25_tests())

    # Print JSON summary
    import json

    print("\n" + "=" * 80)
    print("JSON Output")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))

    # Exit with appropriate code
    exit_code = 0 if results["summary"]["passed"] == results["summary"]["total"] else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
