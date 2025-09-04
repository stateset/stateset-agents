#!/usr/bin/env python3
"""
Benchmarking script for StateSet Agents framework.

This script provides comprehensive benchmarking capabilities for:
- Agent response generation
- Training performance
- Memory usage patterns
- End-to-end workflows
"""

import asyncio
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from utils.performance_monitor import (
    PerformanceMonitor, 
    BenchmarkRunner, 
    benchmark_async_function,
    MemoryProfiler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FrameworkBenchmark:
    """Comprehensive benchmark suite for the framework."""
    
    def __init__(self, output_dir: str = "./benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = PerformanceMonitor()
        self.memory_profiler = MemoryProfiler()
        self.runner = BenchmarkRunner()
    
    async def benchmark_agent_response_generation(
        self, 
        agent_class=None, 
        config=None,
        num_messages: int = 10,
        message_length: int = 50
    ) -> Dict[str, Any]:
        """Benchmark agent response generation performance."""
        logger.info("🔬 Benchmarking agent response generation...")
        
        if agent_class is None:
            # Use a mock agent for basic benchmarking
            class MockAgent:
                async def generate_response(self, messages, **kwargs):
                    await asyncio.sleep(0.01)  # Simulate processing time
                    return "This is a mock response for benchmarking purposes."
            
            agent = MockAgent()
        else:
            # Use real agent
            agent = agent_class(config)
            await agent.initialize()
        
        # Generate test messages
        test_messages = [
            [{"role": "user", "content": f"Test message {i} with some content" * (message_length // 10)}]
            for i in range(num_messages)
        ]
        
        # Benchmark response generation
        results = await benchmark_async_function(
            agent.generate_response,
            test_messages[0][0],  # Single message for benchmark
            warmup_runs=2,
            benchmark_runs=5
        )
        
        # Test batch processing
        start_time = time.time()
        responses = []
        for messages in test_messages:
            response = await agent.generate_response(messages[0])
            responses.append(response)
        batch_time = time.time() - start_time
        
        results.update({
            "batch_processing": {
                "total_time": batch_time,
                "avg_time_per_message": batch_time / len(test_messages),
                "messages_processed": len(test_messages)
            }
        })
        
        return results
    
    async def benchmark_memory_usage(
        self, 
        agent_class=None, 
        config=None,
        num_conversations: int = 5,
        turns_per_conversation: int = 10
    ) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        logger.info("🔬 Benchmarking memory usage...")
        
        if agent_class is None:
            # Use mock agent
            class MockAgent:
                def __init__(self):
                    self.conversation_history = []
                
                async def generate_response(self, messages, **kwargs):
                    await asyncio.sleep(0.005)
                    response = f"Response to: {messages['content'][:20]}..."
                    self.conversation_history.append({"user": messages["content"], "assistant": response})
                    return response
            
            agent = MockAgent()
        else:
            agent = agent_class(config)
            await agent.initialize()
        
        results = {
            "memory_usage": [],
            "conversation_stats": {
                "num_conversations": num_conversations,
                "turns_per_conversation": turns_per_conversation
            }
        }
        
        with self.memory_profiler.profile_memory("conversation_simulation"):
            for conv in range(num_conversations):
                conversation_id = self.monitor.start_operation(f"conversation_{conv}")
                
                for turn in range(turns_per_conversation):
                    messages = {"role": "user", "content": f"Conversation {conv}, turn {turn}: Hello"}
                    response = await agent.generate_response(messages)
                
                metrics = self.monitor.end_operation(conversation_id)
                results["memory_usage"].append({
                    "conversation": conv,
                    "duration": metrics.duration,
                    "memory_delta": metrics.memory_delta
                })
        
        return results
    
    async def benchmark_training_performance(
        self,
        trainer_class=None,
        agent=None,
        environment=None,
        num_episodes: int = 3
    ) -> Dict[str, Any]:
        """Benchmark training performance."""
        logger.info("🔬 Benchmarking training performance...")
        
        if trainer_class is None:
            # Mock training for benchmarking
            async def mock_training():
                await asyncio.sleep(0.1)  # Simulate training time
                return {"loss": 0.5, "reward": 0.8}
            
            training_func = mock_training
        else:
            # Use real trainer
            trainer = trainer_class(agent, environment)
            training_func = trainer.train_episode
        
        results = await benchmark_async_function(
            training_func,
            warmup_runs=1,
            benchmark_runs=num_episodes
        )
        
        return results
    
    async def run_full_benchmark_suite(
        self,
        agent_class=None,
        trainer_class=None,
        config=None
    ) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("🚀 Running full benchmark suite...")
        
        results = {
            "timestamp": time.time(),
            "benchmarks": {}
        }
        
        # Agent response generation benchmark
        try:
            agent_results = await self.benchmark_agent_response_generation(
                agent_class=agent_class,
                config=config
            )
            results["benchmarks"]["agent_response"] = agent_results
        except Exception as e:
            logger.error(f"Agent response benchmark failed: {e}")
            results["benchmarks"]["agent_response"] = {"error": str(e)}
        
        # Memory usage benchmark
        try:
            memory_results = await self.benchmark_memory_usage(
                agent_class=agent_class,
                config=config
            )
            results["benchmarks"]["memory_usage"] = memory_results
        except Exception as e:
            logger.error(f"Memory usage benchmark failed: {e}")
            results["benchmarks"]["memory_usage"] = {"error": str(e)}
        
        # Training benchmark
        try:
            training_results = await self.benchmark_training_performance(
                trainer_class=trainer_class,
                num_episodes=2
            )
            results["benchmarks"]["training"] = training_results
        except Exception as e:
            logger.error(f"Training benchmark failed: {e}")
            results["benchmarks"]["training"] = {"error": str(e)}
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Benchmark results saved to {filepath}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load benchmark results from file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'r') as f:
            return json.load(f)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark StateSet Agents framework")
    parser.add_argument("--output-dir", default="./benchmarks/results", help="Output directory")
    parser.add_argument("--agent-class", help="Agent class to benchmark")
    parser.add_argument("--trainer-class", help="Trainer class to benchmark")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--num-messages", type=int, default=10, help="Number of messages for agent benchmark")
    parser.add_argument("--num-conversations", type=int, default=5, help="Number of conversations for memory benchmark")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes for training benchmark")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    benchmark = FrameworkBenchmark(output_dir=args.output_dir)
    
    # Run benchmark suite
    results = await benchmark.run_full_benchmark_suite(
        agent_class=args.agent_class,
        trainer_class=args.trainer_class,
        config=args.config
    )
    
    # Save results
    timestamp = int(time.time())
    filename = f"benchmark_results_{timestamp}.json"
    benchmark.save_results(results, filename)
    
    # Print summary
    print("\n📊 Benchmark Summary:")
    for benchmark_name, benchmark_results in results["benchmarks"].items():
        if "error" in benchmark_results:
            print(f"❌ {benchmark_name}: Failed ({benchmark_results['error']})")
        elif "avg_duration" in benchmark_results:
            print(".3f"        else:
            print(f"✅ {benchmark_name}: Completed")
    
    print(f"\n📁 Full results saved to {args.output_dir}/{filename}")


if __name__ == "__main__":
    asyncio.run(main())
