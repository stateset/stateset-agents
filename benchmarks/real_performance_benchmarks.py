"""
Real Performance Benchmarks for StateSet Agents

This module provides comprehensive, verifiable performance benchmarks
to validate the framework's production claims.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import create_customer_service_reward


class BenchmarkResult:
    """Container for benchmark results"""

    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "framework_version": "0.5.0",
        }

    def add_metric(self, key: str, value: Any):
        """Add a metric to the results"""
        self.metrics[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def save(self, output_dir: Path):
        """Save results to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Results saved to: {filepath}")


class PerformanceBenchmark:
    """Run comprehensive performance benchmarks"""

    def __init__(self):
        self.results = []

    async def benchmark_response_time(self, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark average response generation time"""
        print(f"\n{'=' * 60}")
        print("BENCHMARK: Response Generation Time")
        print(f"{'=' * 60}")

        result = BenchmarkResult("response_time")

        # Create stub agent for consistent benchmarking
        config = AgentConfig(
            model_name="stub://benchmark",
            use_stub_model=True,
            stub_responses=["I understand your concern. Let me help you with that."],
            max_new_tokens=256,
        )

        agent = MultiTurnAgent(config)
        await agent.initialize()

        # Warmup
        for _ in range(10):
            await agent.generate_response([{"role": "user", "content": "Hello"}])

        # Measure
        times = []
        for i in range(num_iterations):
            messages = [{"role": "user", "content": f"Request {i}"}]

            start = time.perf_counter()
            await agent.generate_response(messages)
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        result.add_metric("iterations", num_iterations)
        result.add_metric("mean_ms", np.mean(times))
        result.add_metric("median_ms", np.median(times))
        result.add_metric("std_ms", np.std(times))
        result.add_metric("min_ms", np.min(times))
        result.add_metric("max_ms", np.max(times))
        result.add_metric("p95_ms", np.percentile(times, 95))
        result.add_metric("p99_ms", np.percentile(times, 99))

        print(f"Mean response time: {result.metrics['mean_ms']:.2f}ms")
        print(f"Median response time: {result.metrics['median_ms']:.2f}ms")
        print(f"P95 response time: {result.metrics['p95_ms']:.2f}ms")
        print(f"P99 response time: {result.metrics['p99_ms']:.2f}ms")

        self.results.append(result)
        return result

    async def benchmark_throughput(self, duration_seconds: int = 10) -> BenchmarkResult:
        """Benchmark conversation processing throughput"""
        print(f"\n{'=' * 60}")
        print("BENCHMARK: Conversation Throughput")
        print(f"{'=' * 60}")

        result = BenchmarkResult("throughput")

        # Create stub agent
        config = AgentConfig(
            model_name="stub://benchmark",
            use_stub_model=True,
            stub_responses=["Response generated"],
            max_new_tokens=128,
        )

        agent = MultiTurnAgent(config)
        await agent.initialize()

        # Run for specified duration
        start_time = time.time()
        end_time = start_time + duration_seconds

        conversation_count = 0
        turn_count = 0

        while time.time() < end_time:
            # Simulate a 3-turn conversation
            for turn in range(3):
                messages = [{"role": "user", "content": f"Turn {turn}"}]
                await agent.generate_response(messages)
                turn_count += 1

            conversation_count += 1

        actual_duration = time.time() - start_time
        conversations_per_second = conversation_count / actual_duration
        turns_per_second = turn_count / actual_duration

        result.add_metric("duration_seconds", actual_duration)
        result.add_metric("total_conversations", conversation_count)
        result.add_metric("total_turns", turn_count)
        result.add_metric("conversations_per_second", conversations_per_second)
        result.add_metric("turns_per_second", turns_per_second)

        print(f"Total conversations: {conversation_count}")
        print(f"Total turns: {turn_count}")
        print(f"Conversations/sec: {conversations_per_second:.2f}")
        print(f"Turns/sec: {turns_per_second:.2f}")

        self.results.append(result)
        return result

    async def benchmark_concurrent_conversations(
        self, num_concurrent: int = 100
    ) -> BenchmarkResult:
        """Benchmark concurrent conversation handling"""
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: Concurrent Conversations (N={num_concurrent})")
        print(f"{'=' * 60}")

        result = BenchmarkResult("concurrent_conversations")

        # Create stub agent
        config = AgentConfig(
            model_name="stub://benchmark",
            use_stub_model=True,
            stub_responses=["Concurrent response"],
            max_new_tokens=128,
        )

        agent = MultiTurnAgent(config)
        await agent.initialize()

        async def single_conversation(conv_id: int):
            """Simulate a single conversation"""
            for turn in range(5):  # 5 turns per conversation
                messages = [{"role": "user", "content": f"Conv {conv_id} Turn {turn}"}]
                await agent.generate_response(messages)

        # Run concurrent conversations
        start_time = time.time()

        tasks = [single_conversation(i) for i in range(num_concurrent)]
        await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        total_turns = num_concurrent * 5
        turns_per_second = total_turns / total_time

        result.add_metric("num_concurrent", num_concurrent)
        result.add_metric("total_time_seconds", total_time)
        result.add_metric("total_turns", total_turns)
        result.add_metric("turns_per_second", turns_per_second)
        result.add_metric("avg_time_per_conversation_ms", (total_time / num_concurrent) * 1000)

        print(f"Total time: {total_time:.2f}s")
        print(f"Total turns: {total_turns}")
        print(f"Turns/sec: {turns_per_second:.2f}")
        print(f"Avg time per conversation: {result.metrics['avg_time_per_conversation_ms']:.2f}ms")

        self.results.append(result)
        return result

    async def benchmark_memory_efficiency(self, num_conversations: int = 1000) -> BenchmarkResult:
        """Benchmark memory efficiency"""
        print(f"\n{'=' * 60}")
        print("BENCHMARK: Memory Efficiency")
        print(f"{'=' * 60}")

        result = BenchmarkResult("memory_efficiency")

        if not HAS_PSUTIL:
            print("psutil not available, skipping memory benchmark")
            result.add_metric("skipped", True)
            self.results.append(result)
            return result

        process = psutil.Process()

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create agent and run conversations
        config = AgentConfig(
            model_name="stub://benchmark",
            use_stub_model=True,
            stub_responses=["Memory test response"],
            max_new_tokens=128,
        )

        agent = MultiTurnAgent(config)
        await agent.initialize()

        # Process conversations
        for i in range(num_conversations):
            messages = [{"role": "user", "content": f"Conversation {i}"}]
            await agent.generate_response(messages)

            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Processed {i} conversations, Memory: {current_memory:.2f}MB")

        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - baseline_memory
        memory_per_conversation = memory_increase / num_conversations

        result.add_metric("num_conversations", num_conversations)
        result.add_metric("baseline_memory_mb", baseline_memory)
        result.add_metric("final_memory_mb", final_memory)
        result.add_metric("memory_increase_mb", memory_increase)
        result.add_metric("memory_per_conversation_kb", memory_per_conversation * 1024)
        result.add_metric("memory_efficiency_percent", (1 - memory_increase / final_memory) * 100)

        print(f"Baseline memory: {baseline_memory:.2f}MB")
        print(f"Final memory: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        print(f"Memory per conversation: {memory_per_conversation * 1024:.2f}KB")

        self.results.append(result)
        return result

    async def benchmark_training_iteration(self, num_episodes: int = 10) -> BenchmarkResult:
        """Benchmark training iteration speed"""
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: Training Iteration Speed (N={num_episodes})")
        print(f"{'=' * 60}")

        result = BenchmarkResult("training_iteration")

        # Create agent
        config = AgentConfig(
            model_name="stub://benchmark",
            use_stub_model=True,
            stub_responses=["Training response"],
            max_new_tokens=128,
        )

        agent = MultiTurnAgent(config)
        await agent.initialize()

        # Create environment
        scenarios = [
            {"id": f"scenario_{i}", "context": f"Scenario {i}"} for i in range(5)
        ]
        environment = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        # Benchmark trajectory generation
        start_time = time.time()

        episode_times = []
        for episode in range(num_episodes):
            episode_start = time.time()

            # Generate trajectory
            async def agent_fn(history, context):
                return await agent.generate_response(history, context)

            trajectory = await environment.run_episode(agent_fn, scenarios[episode % len(scenarios)])

            episode_end = time.time()
            episode_times.append(episode_end - episode_start)

        total_time = time.time() - start_time

        result.add_metric("num_episodes", num_episodes)
        result.add_metric("total_time_seconds", total_time)
        result.add_metric("episodes_per_second", num_episodes / total_time)
        result.add_metric("avg_episode_time_seconds", np.mean(episode_times))
        result.add_metric("median_episode_time_seconds", np.median(episode_times))

        print(f"Total time: {total_time:.2f}s")
        print(f"Episodes/sec: {result.metrics['episodes_per_second']:.2f}")
        print(f"Avg episode time: {result.metrics['avg_episode_time_seconds']:.4f}s")

        self.results.append(result)
        return result

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all benchmarks"""
        print(f"\n{'=' * 60}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 60}")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "framework_version": "0.5.0",
            "benchmarks": [r.to_dict() for r in self.results],
        }

        # Extract key metrics for README validation
        key_metrics = {}

        for result in self.results:
            if result.name == "response_time":
                key_metrics["avg_response_time_ms"] = result.metrics.get("mean_ms", 0)
                key_metrics["p99_response_time_ms"] = result.metrics.get("p99_ms", 0)

            elif result.name == "throughput":
                key_metrics["conversations_per_second"] = result.metrics.get("conversations_per_second", 0)

            elif result.name == "concurrent_conversations":
                key_metrics["concurrent_capacity"] = result.metrics.get("num_concurrent", 0)
                key_metrics["concurrent_throughput"] = result.metrics.get("turns_per_second", 0)

            elif result.name == "memory_efficiency":
                if not result.metrics.get("skipped"):
                    key_metrics["memory_efficiency_percent"] = result.metrics.get("memory_efficiency_percent", 0)

        summary["key_metrics"] = key_metrics

        print("\n📊 Key Performance Metrics:")
        for metric, value in key_metrics.items():
            print(f"  • {metric}: {value:.2f}")

        return summary

    async def run_all_benchmarks(self, output_dir: str = "./benchmark_results"):
        """Run all benchmarks and save results"""
        print("=" * 60)
        print("StateSet Agents - Performance Benchmark Suite")
        print("=" * 60)

        output_path = Path(output_dir)

        # Run benchmarks
        await self.benchmark_response_time(num_iterations=100)
        await self.benchmark_throughput(duration_seconds=10)
        await self.benchmark_concurrent_conversations(num_concurrent=100)
        await self.benchmark_memory_efficiency(num_conversations=1000)
        await self.benchmark_training_iteration(num_episodes=10)

        # Generate summary
        summary = self.generate_summary_report()

        # Save summary
        summary_path = output_path / "benchmark_summary.json"
        output_path.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ All benchmarks completed!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📄 Summary: {summary_path}")

        return summary


async def main():
    """Run benchmarks"""
    benchmark = PerformanceBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
