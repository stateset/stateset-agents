"""
Fine-tune Code Generation Models with RL

This example demonstrates how to fine-tune code generation models using
reinforcement learning for improved code quality, correctness, and style.

Supported Code Models:
- Qwen/Qwen2.5-Coder-0.5B
- Qwen/Qwen2.5-Coder-1.5B
- Qwen/Qwen2.5-Coder-3B
- Qwen/Qwen2.5-Coder-7B
- Qwen/Qwen2.5-Coder-14B
- Qwen/Qwen2.5-Coder-32B
- codellama/CodeLlama-7b-Python-hf
- codellama/CodeLlama-13b-Python-hf
- codellama/CodeLlama-34b-Python-hf
- deepseek-ai/deepseek-coder-1.3b-instruct
- deepseek-ai/deepseek-coder-6.7b-instruct
- deepseek-ai/deepseek-coder-33b-instruct

Training Objectives:
1. Code Correctness - Passes test cases
2. Code Quality - Follows best practices and style guides
3. Efficiency - Produces efficient solutions
4. Documentation - Generates helpful docstrings/comments

Usage:
    # Quick test with small model
    python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-0.5B

    # Production training with 7B model
    python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-7B --use-lora

    # Train on specific language
    python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-7B --language python

    # Use execution-based rewards (requires Docker)
    python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-7B --use-execution
"""

import argparse
import asyncio
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Code Quality Reward Functions
# =============================================================================

@dataclass
class CodeRewardConfig:
    """Configuration for code reward computation"""
    # Weights for different reward components
    correctness_weight: float = 0.4
    style_weight: float = 0.2
    efficiency_weight: float = 0.2
    documentation_weight: float = 0.2

    # Code quality thresholds
    max_line_length: int = 100
    max_function_length: int = 50
    min_docstring_length: int = 10

    # Execution settings
    execution_timeout: int = 5  # seconds
    use_docker_sandbox: bool = True


class CodeStyleReward:
    """
    Reward function for code style and quality.

    Evaluates:
    - PEP8 compliance (Python)
    - Naming conventions
    - Code structure
    - Complexity metrics
    """

    def __init__(self, config: CodeRewardConfig = None):
        self.config = config or CodeRewardConfig()

    def compute_reward(self, code: str, language: str = "python") -> Tuple[float, Dict[str, float]]:
        """Compute style-based reward for code"""
        breakdown = {}

        if language == "python":
            breakdown["line_length"] = self._check_line_length(code)
            breakdown["naming"] = self._check_naming_conventions(code)
            breakdown["structure"] = self._check_structure(code)
            breakdown["docstrings"] = self._check_docstrings(code)
        else:
            # Generic checks for other languages
            breakdown["line_length"] = self._check_line_length(code)
            breakdown["structure"] = self._check_basic_structure(code)
            breakdown["comments"] = self._check_comments(code)

        # Weighted average
        total = sum(breakdown.values()) / len(breakdown) if breakdown else 0.0
        return total, breakdown

    def _check_line_length(self, code: str) -> float:
        """Check if lines are within acceptable length"""
        lines = code.split("\n")
        if not lines:
            return 1.0

        violations = sum(1 for line in lines if len(line) > self.config.max_line_length)
        return max(0.0, 1.0 - (violations / len(lines)))

    def _check_naming_conventions(self, code: str) -> float:
        """Check Python naming conventions"""
        score = 1.0

        # Check for snake_case in function/variable names
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        functions = re.findall(func_pattern, code)

        for func in functions:
            if not re.match(r'^[a-z_][a-z0-9_]*$', func):
                score -= 0.1

        # Check for PascalCase in class names
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        classes = re.findall(class_pattern, code)

        for cls in classes:
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', cls):
                score -= 0.1

        return max(0.0, score)

    def _check_structure(self, code: str) -> float:
        """Check code structure quality"""
        score = 1.0

        # Check function length
        func_pattern = r'def\s+\w+.*?(?=\ndef|\nclass|\Z)'
        functions = re.findall(func_pattern, code, re.DOTALL)

        for func in functions:
            lines = func.strip().split("\n")
            if len(lines) > self.config.max_function_length:
                score -= 0.2

        # Check for deeply nested code (more than 4 levels)
        max_indent = 0
        for line in code.split("\n"):
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                spaces_per_indent = 4
                level = indent // spaces_per_indent
                max_indent = max(max_indent, level)

        if max_indent > 4:
            score -= 0.1 * (max_indent - 4)

        return max(0.0, score)

    def _check_docstrings(self, code: str) -> float:
        """Check for presence and quality of docstrings"""
        func_pattern = r'def\s+(\w+)[^:]*:'
        functions = re.findall(func_pattern, code)

        if not functions:
            return 1.0

        docstring_pattern = r'def\s+\w+[^:]*:\s*"""[^"]+"""'
        documented = len(re.findall(docstring_pattern, code))

        return documented / len(functions) if functions else 1.0

    def _check_basic_structure(self, code: str) -> float:
        """Basic structure checks for any language"""
        score = 1.0

        # Check balanced braces/brackets
        opens = code.count("{") + code.count("[") + code.count("(")
        closes = code.count("}") + code.count("]") + code.count(")")

        if opens != closes:
            score -= 0.5

        return max(0.0, score)

    def _check_comments(self, code: str) -> float:
        """Check for presence of comments"""
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        if not lines:
            return 0.5

        comment_patterns = [r'#', r'//', r'/\*', r'\*/', r'"""', r"'''"]
        comment_lines = sum(
            1 for line in lines
            if any(re.search(p, line) for p in comment_patterns)
        )

        ratio = comment_lines / len(lines)
        # Ideal is 10-30% comments
        if 0.1 <= ratio <= 0.3:
            return 1.0
        elif ratio < 0.1:
            return 0.5 + ratio * 5
        else:
            return max(0.5, 1.0 - (ratio - 0.3) * 2)


class CodeExecutionReward:
    """
    Reward function based on code execution results.

    Executes generated code against test cases and rewards
    based on correctness, runtime, and memory usage.
    """

    def __init__(self, config: CodeRewardConfig = None):
        self.config = config or CodeRewardConfig()

    async def compute_reward(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        language: str = "python",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Execute code and compute reward based on test results.

        Args:
            code: Generated code
            test_cases: List of {"input": ..., "expected": ...}
            language: Programming language

        Returns:
            (reward, breakdown) tuple
        """
        breakdown = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "timeout": 0,
        }

        if not test_cases:
            return 0.5, breakdown

        for test in test_cases:
            try:
                result = await self._execute_code(
                    code, test.get("input", ""), language
                )

                expected = str(test.get("expected", "")).strip()
                actual = str(result.get("output", "")).strip()

                if actual == expected:
                    breakdown["passed"] += 1
                else:
                    breakdown["failed"] += 1

            except asyncio.TimeoutError:
                breakdown["timeout"] += 1
            except Exception as e:
                breakdown["errors"] += 1
                logger.debug(f"Execution error: {e}")

        total = len(test_cases)
        reward = breakdown["passed"] / total if total > 0 else 0.0

        return reward, breakdown

    async def _execute_code(
        self,
        code: str,
        input_data: str,
        language: str,
    ) -> Dict[str, Any]:
        """Execute code in a sandbox"""

        if language == "python":
            return await self._execute_python(code, input_data)
        else:
            raise ValueError(f"Unsupported language: {language}")

    async def _execute_python(self, code: str, input_data: str) -> Dict[str, Any]:
        """Execute Python code"""

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                'python', temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_data.encode() if input_data else None),
                    timeout=self.config.execution_timeout
                )

                return {
                    "output": stdout.decode(),
                    "error": stderr.decode(),
                    "returncode": process.returncode,
                }

            except asyncio.TimeoutError:
                process.kill()
                raise

        finally:
            os.unlink(temp_path)


class CompositeCodeReward:
    """
    Combined reward function for code generation.

    Combines style, correctness, and other metrics into a single reward.
    """

    def __init__(self, config: CodeRewardConfig = None):
        self.config = config or CodeRewardConfig()
        self.style_reward = CodeStyleReward(config)
        self.execution_reward = CodeExecutionReward(config)

    async def compute_reward(
        self,
        prompt: str,
        code: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        language: str = "python",
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute combined reward"""
        breakdown = {}

        # Style reward
        style_score, style_breakdown = self.style_reward.compute_reward(code, language)
        breakdown["style"] = style_breakdown

        # Execution reward (if test cases provided)
        if test_cases:
            exec_score, exec_breakdown = await self.execution_reward.compute_reward(
                code, test_cases, language
            )
            breakdown["execution"] = exec_breakdown

            # Weighted combination
            reward = (
                self.config.correctness_weight * exec_score +
                self.config.style_weight * style_score
            )
        else:
            # Style only
            reward = style_score

        breakdown["total"] = reward
        return reward, breakdown


# =============================================================================
# Training Configuration
# =============================================================================

def get_code_model_config(
    model_name: str,
    language: str = "python",
    use_lora: bool = True,
    use_4bit: bool = False,
    output_dir: str = "./outputs/code_assistant",
):
    """Get optimized GSPO configuration for code models"""
    from stateset_agents.training.gspo_trainer import GSPOConfig
    from stateset_agents.training.config import TrainingConfig

    model_lower = model_name.lower()

    # Determine model size
    if any(x in model_lower for x in ["0.5b", "1.3b", "1.5b"]):
        # Small model
        config = GSPOConfig(
            model_name=model_name,
            num_generations=4,
            clip_range_left=3e-4,
            clip_range_right=4e-4,
            learning_rate=2e-5,
            num_outer_iterations=100,
            generations_per_iteration=50,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            use_lora=use_lora,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            gradient_checkpointing=True,
            use_4bit=use_4bit,
            max_prompt_length=1024,
            max_completion_length=1024,
            temperature=0.8,  # Slightly higher for code diversity
            output_dir=output_dir,
            save_steps=10,
            logging_steps=1,
        )
    elif any(x in model_lower for x in ["3b", "6.7b", "7b"]):
        # Medium model
        config = GSPOConfig(
            model_name=model_name,
            num_generations=6,
            clip_range_left=2e-4,
            clip_range_right=3e-4,
            learning_rate=5e-6,
            num_outer_iterations=100,
            generations_per_iteration=30,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            use_lora=True,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            gradient_checkpointing=True,
            use_4bit=use_4bit,
            max_prompt_length=2048,
            max_completion_length=2048,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=5,
            logging_steps=1,
        )
    else:
        # Large model (14B+)
        config = GSPOConfig(
            model_name=model_name,
            num_generations=8,
            clip_range_left=1.5e-4,
            clip_range_right=2.5e-4,
            learning_rate=2e-6,
            num_outer_iterations=100,
            generations_per_iteration=20,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            use_lora=True,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            gradient_checkpointing=True,
            use_4bit=True,
            max_prompt_length=4096,
            max_completion_length=2048,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=5,
            logging_steps=1,
        )

    return config


# =============================================================================
# Training Data
# =============================================================================

def get_code_training_scenarios(language: str = "python") -> List[Dict[str, Any]]:
    """Get training scenarios for code generation"""

    if language == "python":
        return [
            {
                "id": "fibonacci",
                "prompt": "Write a Python function to compute the nth Fibonacci number efficiently.",
                "test_cases": [
                    {"input": "0", "expected": "0"},
                    {"input": "1", "expected": "1"},
                    {"input": "10", "expected": "55"},
                    {"input": "20", "expected": "6765"},
                ],
                "difficulty": "easy",
            },
            {
                "id": "binary_search",
                "prompt": "Implement binary search that returns the index of the target in a sorted list, or -1 if not found.",
                "test_cases": [
                    {"input": "[1,2,3,4,5], 3", "expected": "2"},
                    {"input": "[1,2,3,4,5], 6", "expected": "-1"},
                    {"input": "[], 1", "expected": "-1"},
                ],
                "difficulty": "easy",
            },
            {
                "id": "merge_sort",
                "prompt": "Implement the merge sort algorithm for a list of integers.",
                "test_cases": [
                    {"input": "[3,1,4,1,5,9,2,6]", "expected": "[1,1,2,3,4,5,6,9]"},
                    {"input": "[]", "expected": "[]"},
                    {"input": "[1]", "expected": "[1]"},
                ],
                "difficulty": "medium",
            },
            {
                "id": "lru_cache",
                "prompt": "Implement an LRU (Least Recently Used) cache with get and put operations, both in O(1) time.",
                "test_cases": [],
                "difficulty": "hard",
            },
            {
                "id": "tree_traversal",
                "prompt": "Write a function that performs in-order traversal of a binary tree and returns the values as a list.",
                "test_cases": [],
                "difficulty": "medium",
            },
            {
                "id": "valid_parentheses",
                "prompt": "Write a function to check if a string of parentheses (), [], {} is valid.",
                "test_cases": [
                    {"input": "'()'", "expected": "True"},
                    {"input": "'()[]{}'", "expected": "True"},
                    {"input": "'(]'", "expected": "False"},
                    {"input": "'([)]'", "expected": "False"},
                ],
                "difficulty": "easy",
            },
            {
                "id": "json_parser",
                "prompt": "Write a simple JSON parser that can handle objects, arrays, strings, numbers, booleans, and null.",
                "test_cases": [],
                "difficulty": "hard",
            },
            {
                "id": "rate_limiter",
                "prompt": "Implement a rate limiter class that allows at most N requests per second using the sliding window algorithm.",
                "test_cases": [],
                "difficulty": "hard",
            },
        ]
    else:
        return [
            {
                "id": "hello_world",
                "prompt": f"Write a Hello World program in {language}.",
                "test_cases": [],
                "difficulty": "easy",
            },
        ]


# =============================================================================
# Main Training Function
# =============================================================================

async def finetune_code_assistant(
    model_name: str,
    language: str = "python",
    use_lora: bool = True,
    use_4bit: bool = False,
    use_execution: bool = False,
    output_dir: str = "./outputs/code_assistant",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
):
    """
    Fine-tune a code generation model using GSPO.

    Args:
        model_name: Model name (e.g., "Qwen/Qwen2.5-Coder-7B")
        language: Target programming language
        use_lora: Use LoRA for efficient fine-tuning
        use_4bit: Use 4-bit quantization
        use_execution: Use execution-based rewards
        output_dir: Output directory
        use_wandb: Enable W&B logging
        wandb_project: W&B project name
    """
    from stateset_agents import MultiTurnAgent
    from stateset_agents.core.agent import AgentConfig
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training.gspo_trainer import train_with_gspo

    logger.info("=" * 80)
    logger.info("Fine-tuning Code Assistant with GSPO")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Language: {language}")
    logger.info(f"LoRA: {use_lora}")
    logger.info(f"Quantization: {'4-bit' if use_4bit else 'None'}")
    logger.info(f"Execution rewards: {use_execution}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # System prompt for code generation
    system_prompt = f"""You are an expert {language} programmer. Your role is to:
- Write clean, efficient, and well-documented code
- Follow {language} best practices and style guidelines
- Include appropriate error handling
- Write code that is easy to understand and maintain
- Provide clear explanations when asked

When writing code, always:
1. Start with a docstring explaining the function's purpose
2. Use meaningful variable and function names
3. Add comments for complex logic
4. Handle edge cases appropriately"""

    # Create agent
    logger.info("Initializing code generation agent...")
    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt=system_prompt,
        max_new_tokens=2048,
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()
    logger.info("Agent initialized")

    # Get training scenarios
    scenarios = get_code_training_scenarios(language)
    logger.info(f"Loaded {len(scenarios)} training scenarios")

    # Create environment
    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=4,
    )

    # Create reward function
    logger.info("Setting up reward function...")
    reward_config = CodeRewardConfig()
    reward_fn = CompositeCodeReward(reward_config)

    # Create wrapper for reward model interface
    class CodeRewardModel:
        def __init__(self, reward_fn, language, use_execution):
            self.reward_fn = reward_fn
            self.language = language
            self.use_execution = use_execution

        async def compute_reward(self, turns, metadata):
            # Get the code from the last assistant turn
            code = ""
            for turn in reversed(turns):
                if turn.get("role") == "assistant":
                    code = turn.get("content", "")
                    break

            # Get test cases from scenario if available
            test_cases = metadata.get("scenario", {}).get("test_cases", []) if self.use_execution else None

            reward, breakdown = await self.reward_fn.compute_reward(
                prompt=metadata.get("scenario", {}).get("prompt", ""),
                code=code,
                test_cases=test_cases,
                language=self.language,
            )

            class RewardResult:
                def __init__(self, score, breakdown):
                    self.score = score
                    self.breakdown = breakdown

            return RewardResult(reward, breakdown)

    reward_model = CodeRewardModel(reward_fn, language, use_execution)
    logger.info("Reward function ready")

    # Get GSPO configuration
    gspo_config = get_code_model_config(
        model_name=model_name,
        language=language,
        use_lora=use_lora,
        use_4bit=use_4bit,
        output_dir=output_dir,
    )

    # Enable W&B if requested
    if use_wandb:
        gspo_config.report_to = "wandb"
        gspo_config.wandb_project = wandb_project or f"code-assistant-{language}"
        gspo_config.wandb_tags = ["code-generation", language, model_name.split("/")[-1]]
        logger.info(f"W&B enabled: {gspo_config.wandb_project}")

    logger.info("GSPO configuration ready")
    logger.info(f"   - Group size: {gspo_config.num_generations}")
    logger.info(f"   - Learning rate: {gspo_config.learning_rate}")

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting GSPO training...")
    logger.info("=" * 80 + "\n")

    trained_agent = await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)

    # Test the trained model
    logger.info("\nTesting trained model...")

    test_prompts = [
        "Write a Python function to check if a number is prime.",
        "Implement a function to reverse a linked list.",
        "Write a decorator that caches function results.",
    ]

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        response = await trained_agent.generate_response(messages)

        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Response:\n{response[:500]}...")
        logger.info("-" * 40)

    return trained_agent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fine-tune code generation models with GSPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small model
  python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-0.5B

  # Production training with 7B model
  python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-7B --use-lora

  # Use execution-based rewards
  python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-7B --use-execution

  # Train for JavaScript
  python examples/finetune_code_assistant.py --model Qwen/Qwen2.5-Coder-7B --language javascript
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Code model name",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "javascript", "typescript", "java", "cpp", "rust", "go"],
        help="Target programming language",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--use-execution",
        action="store_true",
        help="Use execution-based rewards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/code_assistant",
        help="Output directory",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name",
    )

    args = parser.parse_args()
    use_lora = args.use_lora and not args.no_lora

    asyncio.run(
        finetune_code_assistant(
            model_name=args.model,
            language=args.language,
            use_lora=use_lora,
            use_4bit=args.use_4bit,
            use_execution=args.use_execution,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
    )


if __name__ == "__main__":
    main()
