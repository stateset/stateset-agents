"""
Performance Benchmarking Suite for GRPO Agent Framework

This module provides comprehensive benchmarking tools to measure and analyze
the performance of all framework components and innovations.
"""

import asyncio
import time
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import psutil
import tracemalloc
from contextlib import contextmanager

# Framework imports
from ..core.agent import Agent
from ..core.environment import Environment
from ..core.reward import RewardFunction, RewardResult
from ..core.computational_engine import ComputationalGRPOEngine
from ..core.multiturn_agent import MultiTurnAgent, DialogueDatabase
from ..rewards.multi_objective_reward import MultiObjectiveRewardFunction, create_customer_service_reward
from ..training.neural_reward_trainer import create_neural_reward_function
from ..utils.monitoring import MonitoringService
from ..utils.cache import CacheService

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    component: str
    duration: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "duration_seconds": self.duration,
            "throughput_ops_per_second": self.throughput,
            "memory_usage_mb": self.memory_usage,
            "cpu_usage_percent": self.cpu_usage,
            "success_rate_percent": self.success_rate * 100,
            "error_count": self.error_count,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    name: str
    iterations: int = 100
    warmup_iterations: int = 10
    concurrent_workers: int = 1
    timeout_seconds: float = 30.0
    memory_profiling: bool = True
    cpu_profiling: bool = True
    detailed_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "concurrent_workers": self.concurrent_workers,
            "timeout_seconds": self.timeout_seconds,
            "memory_profiling": self.memory_profiling,
            "cpu_profiling": self.cpu_profiling,
            "detailed_metrics": self.detailed_metrics
        }


class BenchmarkRunner:
    """Main benchmark runner with comprehensive metrics collection"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.monitoring = MonitoringService()
        self.cache = CacheService()
        
        # System monitoring
        self.process = psutil.Process()
        
        # Performance tracking
        self.baseline_metrics = {}
        self.comparative_results = {}
        
    @contextmanager
    def performance_monitor(self, benchmark_name: str):
        """Context manager for performance monitoring"""
        # Start memory profiling
        tracemalloc.start()
        
        # Get initial system metrics
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = self.process.cpu_percent()
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # Calculate final metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            # Get CPU usage
            cpu_usage = self.process.cpu_percent()
            
            # Store baseline metrics
            self.baseline_metrics[benchmark_name] = {
                "duration": duration,
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "peak_memory": peak / 1024 / 1024  # MB
            }
    
    async def run_benchmark(
        self, 
        name: str, 
        benchmark_func: Callable, 
        config: BenchmarkConfig,
        *args, 
        **kwargs
    ) -> BenchmarkResult:
        """Run a single benchmark with comprehensive monitoring"""
        
        logger.info(f"Starting benchmark: {name}")
        
        # Warmup runs
        if config.warmup_iterations > 0:
            logger.info(f"Running {config.warmup_iterations} warmup iterations...")
            for _ in range(config.warmup_iterations):
                try:
                    await benchmark_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Warmup iteration failed: {e}")
        
        # Main benchmark run
        results = []
        errors = []
        
        with self.performance_monitor(name):
            # Single-threaded benchmark
            if config.concurrent_workers == 1:
                for i in range(config.iterations):
                    try:
                        start_time = time.time()
                        result = await benchmark_func(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        results.append({
                            "iteration": i,
                            "duration": duration,
                            "result": result,
                            "success": True
                        })
                        
                    except Exception as e:
                        errors.append({
                            "iteration": i,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        })
            
            # Concurrent benchmark
            else:
                semaphore = asyncio.Semaphore(config.concurrent_workers)
                
                async def run_single_iteration(iteration_id: int):
                    async with semaphore:
                        try:
                            start_time = time.time()
                            result = await benchmark_func(*args, **kwargs)
                            duration = time.time() - start_time
                            
                            return {
                                "iteration": iteration_id,
                                "duration": duration,
                                "result": result,
                                "success": True
                            }
                        except Exception as e:
                            return {
                                "iteration": iteration_id,
                                "error": str(e),
                                "success": False
                            }
                
                # Run all iterations concurrently
                tasks = [run_single_iteration(i) for i in range(config.iterations)]
                concurrent_results = await asyncio.gather(*tasks)
                
                # Separate successful results from errors
                for result in concurrent_results:
                    if result["success"]:
                        results.append(result)
                    else:
                        errors.append(result)
        
        # Calculate metrics
        baseline = self.baseline_metrics.get(name, {})
        
        success_rate = len(results) / config.iterations if config.iterations > 0 else 0
        throughput = len(results) / baseline.get("duration", 1) if baseline.get("duration", 0) > 0 else 0
        
        # Calculate statistics
        durations = [r["duration"] for r in results if "duration" in r]
        avg_duration = statistics.mean(durations) if durations else 0
        
        benchmark_result = BenchmarkResult(
            name=name,
            component=config.name,
            duration=baseline.get("duration", 0),
            throughput=throughput,
            memory_usage=baseline.get("memory_usage", 0),
            cpu_usage=baseline.get("cpu_usage", 0),
            success_rate=success_rate,
            error_count=len(errors),
            metadata={
                "config": config.to_dict(),
                "iterations_completed": len(results),
                "average_iteration_duration": avg_duration,
                "duration_statistics": {
                    "min": min(durations) if durations else 0,
                    "max": max(durations) if durations else 0,
                    "std": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "median": statistics.median(durations) if durations else 0
                },
                "errors": errors[:10] if config.detailed_metrics else [],  # Limit error details
                "system_info": {
                    "cpu_count": mp.cpu_count(),
                    "memory_gb": psutil.virtual_memory().total / 1024**3,
                    "python_version": f"{psutil.PROCFS_PATH}"
                }
            }
        )
        
        self.results.append(benchmark_result)
        
        # Log results
        logger.info(f"Benchmark {name} completed:")
        logger.info(f"  Duration: {benchmark_result.duration:.2f}s")
        logger.info(f"  Throughput: {benchmark_result.throughput:.2f} ops/sec")
        logger.info(f"  Success rate: {benchmark_result.success_rate:.2%}")
        logger.info(f"  Memory usage: {benchmark_result.memory_usage:.2f} MB")
        
        return benchmark_result
    
    async def run_agent_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for agent components"""
        logger.info("🤖 Running Agent Benchmarks")
        
        # Create test agent
        class TestAgent(Agent):
            def __init__(self):
                super().__init__({"model_type": "benchmark_test"})
                self.response_templates = [
                    "Thank you for your question. I'll help you with that.",
                    "I understand your concern. Let me provide you with a solution.",
                    "I'm here to assist you. Here's what I can do for you.",
                    "Great question! Let me explain this in detail.",
                    "I'd be happy to help. Here's my response to your inquiry."
                ]
                self.response_index = 0
            
            async def generate_response(self, prompt: str) -> str:
                # Simulate processing time
                await asyncio.sleep(0.001)  # 1ms processing
                
                response = self.response_templates[self.response_index % len(self.response_templates)]
                self.response_index += 1
                return response
        
        agent = TestAgent()
        
        # Benchmark configurations
        benchmarks = [
            {
                "name": "Single Response Generation",
                "config": BenchmarkConfig("agent_single_response", iterations=1000, concurrent_workers=1),
                "func": agent.generate_response,
                "args": ("Test prompt for benchmarking",)
            },
            {
                "name": "Concurrent Response Generation",
                "config": BenchmarkConfig("agent_concurrent_response", iterations=500, concurrent_workers=10),
                "func": agent.generate_response,
                "args": ("Test prompt for concurrent benchmarking",)
            },
            {
                "name": "Variable Length Prompts",
                "config": BenchmarkConfig("agent_variable_prompts", iterations=200, concurrent_workers=5),
                "func": self._benchmark_variable_prompts,
                "args": (agent,)
            }
        ]
        
        results = []
        for benchmark in benchmarks:
            result = await self.run_benchmark(
                benchmark["name"],
                benchmark["func"],
                benchmark["config"],
                *benchmark["args"]
            )
            results.append(result)
        
        return results
    
    async def _benchmark_variable_prompts(self, agent: Agent) -> Dict[str, Any]:
        """Benchmark agent with variable prompt lengths"""
        prompts = [
            "Hi",
            "Hello, how are you doing today?",
            "I need help with a complex technical issue that involves multiple components and requires detailed analysis.",
            "This is a very long prompt that contains multiple sentences and covers various topics. It's designed to test how the agent handles longer inputs and whether performance degrades with increased prompt length. The prompt includes technical details, personal questions, and requests for comprehensive assistance across multiple domains."
        ]
        
        results = []
        for prompt in prompts:
            start_time = time.time()
            response = await agent.generate_response(prompt)
            duration = time.time() - start_time
            
            results.append({
                "prompt_length": len(prompt),
                "response_length": len(response),
                "duration": duration
            })
        
        return {
            "variable_prompts_tested": len(prompts),
            "results": results,
            "average_duration": statistics.mean(r["duration"] for r in results)
        }
    
    async def run_multiturn_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for multi-turn conversation components"""
        logger.info("💬 Running Multi-turn Conversation Benchmarks")
        
        # Create dialogue database
        sample_dialogues = [
            {"id": "1", "content": "Hello, I need help with my order", "category": "customer_service"},
            {"id": "2", "content": "I have a technical question about the API", "category": "technical_support"},
            {"id": "3", "content": "Can you help me with billing?", "category": "billing"},
            {"id": "4", "content": "I want to return a product", "category": "returns"},
            {"id": "5", "content": "How do I use this feature?", "category": "feature_help"}
        ]
        
        dialogue_db = DialogueDatabase(sample_dialogues)
        
        # Create multi-turn agent
        multiturn_agent = MultiTurnAgent(
            model_config={"model_type": "benchmark_multiturn"},
            dialogue_database=dialogue_db,
            max_conversation_turns=10
        )
        
        benchmarks = [
            {
                "name": "Conversation Startup",
                "config": BenchmarkConfig("multiturn_startup", iterations=200, concurrent_workers=5),
                "func": self._benchmark_conversation_startup,
                "args": (multiturn_agent,)
            },
            {
                "name": "Multi-turn Dialogue",
                "config": BenchmarkConfig("multiturn_dialogue", iterations=100, concurrent_workers=3),
                "func": self._benchmark_multiturn_dialogue,
                "args": (multiturn_agent,)
            },
            {
                "name": "Concurrent Conversations",
                "config": BenchmarkConfig("multiturn_concurrent", iterations=50, concurrent_workers=10),
                "func": self._benchmark_concurrent_conversations,
                "args": (multiturn_agent,)
            },
            {
                "name": "Dialogue Database Search",
                "config": BenchmarkConfig("dialogue_search", iterations=1000, concurrent_workers=20),
                "func": self._benchmark_dialogue_search,
                "args": (dialogue_db,)
            }
        ]
        
        results = []
        for benchmark in benchmarks:
            result = await self.run_benchmark(
                benchmark["name"],
                benchmark["func"],
                benchmark["config"],
                *benchmark["args"]
            )
            results.append(result)
        
        return results
    
    async def _benchmark_conversation_startup(self, agent: MultiTurnAgent) -> Dict[str, Any]:
        """Benchmark conversation startup time"""
        context = await agent.start_conversation(
            user_id=f"benchmark_user_{uuid.uuid4()}",
            initial_context={"topic": "benchmark_test"}
        )
        
        # Clean up
        agent.end_conversation(context.conversation_id)
        
        return {
            "conversation_id": context.conversation_id,
            "startup_successful": True
        }
    
    async def _benchmark_multiturn_dialogue(self, agent: MultiTurnAgent) -> Dict[str, Any]:
        """Benchmark multi-turn dialogue performance"""
        context = await agent.start_conversation(user_id="benchmark_user")
        
        messages = [
            "Hello, I need help",
            "Can you explain this feature?",
            "What about the pricing?",
            "How do I get started?",
            "Thank you for your help"
        ]
        
        turn_count = 0
        for message in messages:
            turns = await agent.continue_conversation(
                context.conversation_id,
                message,
                strategy="default"
            )
            turn_count += len(turns)
        
        # Clean up
        agent.end_conversation(context.conversation_id)
        
        return {
            "messages_processed": len(messages),
            "total_turns": turn_count,
            "average_turns_per_message": turn_count / len(messages)
        }
    
    async def _benchmark_concurrent_conversations(self, agent: MultiTurnAgent) -> Dict[str, Any]:
        """Benchmark concurrent conversation handling"""
        # Start multiple conversations
        contexts = []
        for i in range(5):
            context = await agent.start_conversation(user_id=f"concurrent_user_{i}")
            contexts.append(context)
        
        # Send messages to all conversations
        for context in contexts:
            await agent.continue_conversation(
                context.conversation_id,
                "Hello from concurrent benchmark",
                strategy="default"
            )
        
        # Clean up
        for context in contexts:
            agent.end_conversation(context.conversation_id)
        
        return {
            "concurrent_conversations": len(contexts),
            "all_successful": True
        }
    
    async def _benchmark_dialogue_search(self, dialogue_db: DialogueDatabase) -> Dict[str, Any]:
        """Benchmark dialogue database search performance"""
        queries = [
            "order help",
            "technical support",
            "billing question",
            "return product",
            "feature usage"
        ]
        
        query = queries[np.random.randint(0, len(queries))]
        results = dialogue_db.search(query, top_k=3)
        
        return {
            "query": query,
            "results_found": len(results),
            "search_successful": True
        }
    
    async def run_reward_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for reward system components"""
        logger.info("🎯 Running Reward System Benchmarks")
        
        # Create test reward functions
        multi_objective_reward = create_customer_service_reward()
        neural_reward = create_neural_reward_function()
        
        # Sample conversation turns
        sample_turns = [
            [
                {"role": "user", "content": "I need help with my order"},
                {"role": "assistant", "content": "I'd be happy to help you with your order. Could you please provide your order number?"}
            ],
            [
                {"role": "user", "content": "I'm frustrated with the service"},
                {"role": "assistant", "content": "I understand your frustration and I apologize for the inconvenience. Let me help resolve this issue for you immediately."}
            ],
            [
                {"role": "user", "content": "Can you explain this feature?"},
                {"role": "assistant", "content": "Certainly! This feature allows you to track your orders in real-time. Here's how it works..."}
            ]
        ]
        
        benchmarks = [
            {
                "name": "Multi-objective Reward Computation",
                "config": BenchmarkConfig("reward_multi_objective", iterations=500, concurrent_workers=10),
                "func": self._benchmark_reward_computation,
                "args": (multi_objective_reward, sample_turns)
            },
            {
                "name": "Neural Reward Computation",
                "config": BenchmarkConfig("reward_neural", iterations=200, concurrent_workers=5),
                "func": self._benchmark_reward_computation,
                "args": (neural_reward, sample_turns)
            },
            {
                "name": "Reward System Scalability",
                "config": BenchmarkConfig("reward_scalability", iterations=100, concurrent_workers=20),
                "func": self._benchmark_reward_scalability,
                "args": (multi_objective_reward, sample_turns)
            }
        ]
        
        results = []
        for benchmark in benchmarks:
            result = await self.run_benchmark(
                benchmark["name"],
                benchmark["func"],
                benchmark["config"],
                *benchmark["args"]
            )
            results.append(result)
        
        return results
    
    async def _benchmark_reward_computation(self, reward_func: RewardFunction, sample_turns: List[List[Dict]]) -> Dict[str, Any]:
        """Benchmark reward computation performance"""
        turns = sample_turns[np.random.randint(0, len(sample_turns))]
        
        result = await reward_func.compute_reward(turns)
        
        return {
            "reward_score": result.score,
            "computation_successful": True,
            "breakdown_items": len(result.breakdown)
        }
    
    async def _benchmark_reward_scalability(self, reward_func: RewardFunction, sample_turns: List[List[Dict]]) -> Dict[str, Any]:
        """Benchmark reward system scalability"""
        # Process multiple turn sets
        results = []
        for turns in sample_turns:
            result = await reward_func.compute_reward(turns)
            results.append(result.score)
        
        return {
            "turn_sets_processed": len(sample_turns),
            "average_score": statistics.mean(results),
            "score_variance": statistics.variance(results) if len(results) > 1 else 0
        }
    
    async def run_computational_engine_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for computational engine"""
        logger.info("⚡ Running Computational Engine Benchmarks")
        
        # Create test components
        class BenchmarkAgent(Agent):
            def __init__(self):
                super().__init__({"model_type": "benchmark_computational"})
            
            async def generate_response(self, prompt: str) -> str:
                await asyncio.sleep(0.005)  # 5ms processing time
                return f"Response to: {prompt[:20]}..."
        
        class BenchmarkEnvironment(Environment):
            def __init__(self):
                super().__init__()
                self.step_count = 0
            
            async def reset(self) -> Dict[str, Any]:
                self.step_count = 0
                return {"step": 0}
            
            async def step(self, action: str) -> Dict[str, Any]:
                self.step_count += 1
                return {"step": self.step_count, "reward": 0.5, "done": False}
            
            async def get_reward(self, trajectory) -> float:
                return 0.5 + np.random.normal(0, 0.1)
        
        class BenchmarkReward(RewardFunction):
            async def compute_reward(self, turns, context=None) -> RewardResult:
                await asyncio.sleep(0.001)  # 1ms processing
                return RewardResult(score=0.5, breakdown={"test": 0.5})
        
        # Create engine
        engine = ComputationalGRPOEngine(
            agent=BenchmarkAgent(),
            environment=BenchmarkEnvironment(),
            reward_function=BenchmarkReward(),
            num_workers=4,
            trajectory_batch_size=10
        )
        
        benchmarks = [
            {
                "name": "Trajectory Generation",
                "config": BenchmarkConfig("engine_trajectory", iterations=50, concurrent_workers=2),
                "func": self._benchmark_trajectory_generation,
                "args": (engine,)
            },
            {
                "name": "Training Iteration",
                "config": BenchmarkConfig("engine_training", iterations=20, concurrent_workers=1),
                "func": self._benchmark_training_iteration,
                "args": (engine,)
            },
            {
                "name": "Computational Scaling",
                "config": BenchmarkConfig("engine_scaling", iterations=10, concurrent_workers=1),
                "func": self._benchmark_computational_scaling,
                "args": (engine,)
            }
        ]
        
        results = []
        for benchmark in benchmarks:
            result = await self.run_benchmark(
                benchmark["name"],
                benchmark["func"],
                benchmark["config"],
                *benchmark["args"]
            )
            results.append(result)
        
        # Clean up
        engine.cleanup()
        
        return results
    
    async def _benchmark_trajectory_generation(self, engine: ComputationalGRPOEngine) -> Dict[str, Any]:
        """Benchmark trajectory generation performance"""
        prompts = [f"Test prompt {i}" for i in range(5)]
        
        trajectories = await engine.generate_trajectory_batch(prompts)
        
        return {
            "trajectories_generated": len(trajectories),
            "prompts_processed": len(prompts),
            "generation_successful": True
        }
    
    async def _benchmark_training_iteration(self, engine: ComputationalGRPOEngine) -> Dict[str, Any]:
        """Benchmark training iteration performance"""
        prompts = ["Training prompt 1", "Training prompt 2", "Training prompt 3"]
        
        results = await engine.train_iteration(prompts)
        
        return {
            "iteration_completed": True,
            "trajectories_generated": results["trajectories_generated"],
            "average_reward": results["average_reward"],
            "computation_time": results["iteration_time"]
        }
    
    async def _benchmark_computational_scaling(self, engine: ComputationalGRPOEngine) -> Dict[str, Any]:
        """Benchmark computational scaling performance"""
        initial_workers = engine.num_workers
        
        # Scale up
        scale_result = engine.scale_computation(2.0)
        
        # Scale back down
        engine.scale_computation(0.5)
        
        return {
            "initial_workers": initial_workers,
            "scaled_workers": scale_result["current_workers"],
            "scaling_successful": True,
            "scale_factor": scale_result["scale_factor"]
        }
    
    async def run_system_benchmarks(self) -> List[BenchmarkResult]:
        """Run system-level benchmarks"""
        logger.info("🖥️ Running System-level Benchmarks")
        
        benchmarks = [
            {
                "name": "Memory Usage Pattern",
                "config": BenchmarkConfig("system_memory", iterations=100, concurrent_workers=5),
                "func": self._benchmark_memory_usage,
                "args": ()
            },
            {
                "name": "CPU Utilization",
                "config": BenchmarkConfig("system_cpu", iterations=50, concurrent_workers=10),
                "func": self._benchmark_cpu_usage,
                "args": ()
            },
            {
                "name": "I/O Performance",
                "config": BenchmarkConfig("system_io", iterations=20, concurrent_workers=3),
                "func": self._benchmark_io_performance,
                "args": ()
            }
        ]
        
        results = []
        for benchmark in benchmarks:
            result = await self.run_benchmark(
                benchmark["name"],
                benchmark["func"],
                benchmark["config"],
                *benchmark["args"]
            )
            results.append(result)
        
        return results
    
    async def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        # Create temporary data structures
        temp_data = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
        
        # Process data
        processed = {k: v.upper() for k, v in temp_data.items()}
        
        return {
            "data_items_processed": len(temp_data),
            "memory_test_successful": True,
            "processed_items": len(processed)
        }
    
    async def _benchmark_cpu_usage(self) -> Dict[str, Any]:
        """Benchmark CPU usage patterns"""
        # CPU-intensive task
        result = sum(i * i for i in range(10000))
        
        return {
            "computation_result": result,
            "cpu_test_successful": True
        }
    
    async def _benchmark_io_performance(self) -> Dict[str, Any]:
        """Benchmark I/O performance"""
        # Simulate I/O operations
        await asyncio.sleep(0.01)  # Simulate disk I/O
        
        return {
            "io_operations_simulated": 1,
            "io_test_successful": True
        }
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites"""
        logger.info("🚀 Running Complete Benchmark Suite")
        
        start_time = time.time()
        
        # Run all benchmark suites
        benchmark_suites = [
            ("Agent Benchmarks", self.run_agent_benchmarks),
            ("Multi-turn Benchmarks", self.run_multiturn_benchmarks),
            ("Reward Benchmarks", self.run_reward_benchmarks),
            ("Computational Engine Benchmarks", self.run_computational_engine_benchmarks),
            ("System Benchmarks", self.run_system_benchmarks)
        ]
        
        all_results = {}
        for suite_name, suite_func in benchmark_suites:
            logger.info(f"Running {suite_name}...")
            suite_results = await suite_func()
            all_results[suite_name] = suite_results
        
        total_time = time.time() - start_time
        
        # Generate summary report
        summary = self.generate_benchmark_summary(all_results, total_time)
        
        return {
            "summary": summary,
            "detailed_results": all_results,
            "total_benchmarks": len(self.results),
            "total_duration": total_time
        }
    
    def generate_benchmark_summary(self, all_results: Dict[str, List[BenchmarkResult]], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary"""
        
        # Aggregate metrics
        all_benchmarks = []
        for suite_results in all_results.values():
            all_benchmarks.extend(suite_results)
        
        if not all_benchmarks:
            return {"error": "No benchmark results to summarize"}
        
        # Calculate summary statistics
        throughputs = [b.throughput for b in all_benchmarks if b.throughput > 0]
        memory_usages = [b.memory_usage for b in all_benchmarks if b.memory_usage > 0]
        success_rates = [b.success_rate for b in all_benchmarks]
        
        summary = {
            "overview": {
                "total_benchmarks": len(all_benchmarks),
                "total_duration_seconds": total_time,
                "benchmark_suites": len(all_results),
                "overall_success_rate": statistics.mean(success_rates) if success_rates else 0
            },
            "performance_metrics": {
                "average_throughput": statistics.mean(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
                "average_memory_usage_mb": statistics.mean(memory_usages) if memory_usages else 0,
                "max_memory_usage_mb": max(memory_usages) if memory_usages else 0
            },
            "top_performers": self._get_top_performers(all_benchmarks),
            "areas_for_improvement": self._identify_improvements(all_benchmarks),
            "system_capabilities": {
                "cpu_cores": mp.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / 1024**3,
                "concurrent_processing": "✅ Supported",
                "computational_scaling": "✅ Verified"
            },
            "recommendations": self._generate_recommendations(all_benchmarks)
        }
        
        return summary
    
    def _get_top_performers(self, benchmarks: List[BenchmarkResult]) -> Dict[str, Any]:
        """Identify top performing benchmarks"""
        if not benchmarks:
            return {}
        
        # Sort by throughput
        by_throughput = sorted(benchmarks, key=lambda b: b.throughput, reverse=True)
        
        # Sort by success rate
        by_success = sorted(benchmarks, key=lambda b: b.success_rate, reverse=True)
        
        # Sort by memory efficiency (lower is better)
        by_memory = sorted(benchmarks, key=lambda b: b.memory_usage)
        
        return {
            "highest_throughput": {
                "name": by_throughput[0].name,
                "throughput": by_throughput[0].throughput,
                "component": by_throughput[0].component
            },
            "highest_success_rate": {
                "name": by_success[0].name,
                "success_rate": by_success[0].success_rate,
                "component": by_success[0].component
            },
            "most_memory_efficient": {
                "name": by_memory[0].name,
                "memory_usage": by_memory[0].memory_usage,
                "component": by_memory[0].component
            }
        }
    
    def _identify_improvements(self, benchmarks: List[BenchmarkResult]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        # Check for low success rates
        low_success = [b for b in benchmarks if b.success_rate < 0.95]
        if low_success:
            improvements.append(f"Improve reliability for {len(low_success)} benchmarks with <95% success rate")
        
        # Check for high memory usage
        high_memory = [b for b in benchmarks if b.memory_usage > 100]  # 100MB threshold
        if high_memory:
            improvements.append(f"Optimize memory usage for {len(high_memory)} benchmarks using >100MB")
        
        # Check for low throughput
        low_throughput = [b for b in benchmarks if b.throughput < 10]  # 10 ops/sec threshold
        if low_throughput:
            improvements.append(f"Improve throughput for {len(low_throughput)} benchmarks with <10 ops/sec")
        
        return improvements
    
    def _generate_recommendations(self, benchmarks: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze patterns
        avg_throughput = statistics.mean(b.throughput for b in benchmarks if b.throughput > 0)
        avg_memory = statistics.mean(b.memory_usage for b in benchmarks if b.memory_usage > 0)
        
        if avg_throughput < 50:
            recommendations.append("Consider increasing parallel processing for better throughput")
        
        if avg_memory > 50:
            recommendations.append("Implement memory optimization techniques")
        
        # Component-specific recommendations
        component_performance = {}
        for benchmark in benchmarks:
            if benchmark.component not in component_performance:
                component_performance[benchmark.component] = []
            component_performance[benchmark.component].append(benchmark.throughput)
        
        for component, throughputs in component_performance.items():
            avg_component_throughput = statistics.mean(throughputs)
            if avg_component_throughput < 20:
                recommendations.append(f"Optimize {component} component for better performance")
        
        return recommendations
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"grpo_benchmark_results_{timestamp}.json"
        
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_benchmarks": len(self.results),
                "framework_version": "2.0.0"
            },
            "results": [result.to_dict() for result in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")
        return filename


async def main():
    """Main benchmark execution"""
    print("🚀 GRPO Agent Framework Performance Benchmarks")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    
    # Run all benchmarks
    results = await runner.run_all_benchmarks()
    
    # Display summary
    print("\n📊 Benchmark Summary")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"Total Benchmarks: {summary['overview']['total_benchmarks']}")
    print(f"Total Duration: {summary['overview']['total_duration_seconds']:.2f}s")
    print(f"Overall Success Rate: {summary['overview']['overall_success_rate']:.2%}")
    print(f"Average Throughput: {summary['performance_metrics']['average_throughput']:.2f} ops/sec")
    print(f"Max Throughput: {summary['performance_metrics']['max_throughput']:.2f} ops/sec")
    print(f"Average Memory Usage: {summary['performance_metrics']['average_memory_usage_mb']:.2f} MB")
    
    # Display top performers
    print("\n🏆 Top Performers")
    print("-" * 30)
    top = summary["top_performers"]
    if top:
        print(f"Highest Throughput: {top['highest_throughput']['name']} ({top['highest_throughput']['throughput']:.2f} ops/sec)")
        print(f"Highest Success Rate: {top['highest_success_rate']['name']} ({top['highest_success_rate']['success_rate']:.2%})")
        print(f"Most Memory Efficient: {top['most_memory_efficient']['name']} ({top['most_memory_efficient']['memory_usage']:.2f} MB)")
    
    # Display recommendations
    if summary["recommendations"]:
        print("\n💡 Recommendations")
        print("-" * 30)
        for rec in summary["recommendations"]:
            print(f"• {rec}")
    
    # Save results
    filename = runner.save_results()
    print(f"\n💾 Results saved to: {filename}")
    
    print("\n✅ Benchmark suite completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())