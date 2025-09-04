"""
Enhanced GRPO Agent Framework Demonstration

This example demonstrates the major improvements in v0.3.0:
- Enhanced error handling and resilience
- Performance optimization
- Type safety and validation
- Advanced async resource management
- Real-time monitoring and diagnostics
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import enhanced framework features
from stateset_agents import (
    # Core components
    MultiTurnAgent, ConversationEnvironment,
    HelpfulnessReward, SafetyReward, CompositeReward,
    
    # Enhanced features
    PerformanceOptimizer, OptimizationLevel,
    ErrorHandler, RetryConfig, NetworkException,
    TypeValidator, ConfigValidator, create_typed_config,
    AsyncTaskManager, managed_async_resources,
    
    # Types
    ModelConfig, TrainingConfig, DeviceType, ModelSize
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_demo.log')
    ]
)
logger = logging.getLogger(__name__)


async def demonstrate_type_safety():
    """Demonstrate type safety and validation features"""
    logger.info("🔍 Demonstrating Type Safety and Validation")
    
    # Create type-safe configuration
    try:
        model_config = create_typed_config(
            ModelConfig,
            model_name="gpt2",
            device=DeviceType.AUTO,
            torch_dtype="bfloat16",
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True
        )
        logger.info("✅ Type-safe model configuration created successfully")
        
    except ValueError as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False
    
    # Validate configuration
    validator = ConfigValidator()
    is_valid = validator.validate_model_config(model_config)
    report = validator.get_validation_report()
    
    if is_valid:
        logger.info("✅ Model configuration is valid")
    else:
        logger.warning(f"⚠️  Configuration issues: {report['errors']}")
    
    # Test type validation
    type_validator = TypeValidator()
    
    test_cases = [
        (42, int, "Integer validation"),
        ("hello", str, "String validation"),
        ([1, 2, 3], List[int], "List[int] validation"),
        ({"key": "value"}, Dict[str, str], "Dict[str, str] validation")
    ]
    
    for value, expected_type, description in test_cases:
        is_valid_type = type_validator.validate_type(value, expected_type)
        status = "✅" if is_valid_type else "❌"
        logger.info(f"{status} {description}: {is_valid_type}")
    
    return True


async def demonstrate_error_handling():
    """Demonstrate enhanced error handling and resilience"""
    logger.info("🛡️ Demonstrating Enhanced Error Handling")
    
    # Initialize error handler
    error_handler = ErrorHandler()
    
    # Test retry mechanism
    @retry_async(RetryConfig(max_attempts=3, base_delay=0.1))
    async def unreliable_operation(fail_count: int = 2):
        """Simulated unreliable operation"""
        if hasattr(unreliable_operation, 'attempts'):
            unreliable_operation.attempts += 1
        else:
            unreliable_operation.attempts = 1
        
        if unreliable_operation.attempts <= fail_count:
            raise NetworkException(
                f"Simulated failure (attempt {unreliable_operation.attempts})",
                details={"attempt": unreliable_operation.attempts}
            )
        
        return f"Success after {unreliable_operation.attempts} attempts"
    
    try:
        result = await unreliable_operation()
        logger.info(f"✅ Retry mechanism succeeded: {result}")
    except Exception as e:
        error_context = error_handler.handle_error(e, "demo", "retry_test")
        logger.error(f"❌ Operation failed: {error_context.error_id}")
    
    # Test error summary
    error_summary = error_handler.get_error_summary()
    logger.info(f"📊 Error summary: {error_summary}")
    
    return True


async def demonstrate_performance_optimization():
    """Demonstrate performance optimization features"""
    logger.info("⚡ Demonstrating Performance Optimization")
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(
        optimization_level=OptimizationLevel.BALANCED
    )
    
    # Simulate training steps with optimization
    logger.info("Simulating optimized training steps...")
    
    for step in range(10):
        # This would normally be your actual model
        class MockModel:
            def __init__(self):
                self.parameters = []
        
        mock_model = MockModel()
        
        # Optimize training step
        optimization_result = optimizer.optimize_training_step(mock_model)
        
        if step % 5 == 0:  # Log every 5 steps
            logger.info(f"Step {step}: {optimization_result['memory_stats']}")
    
    # Get performance report
    performance_report = optimizer.get_performance_report()
    logger.info(f"📈 Performance report:")
    logger.info(f"   Optimization level: {performance_report['optimization_level']}")
    logger.info(f"   Total steps: {performance_report['total_steps']}")
    logger.info(f"   Current memory: {performance_report['current_memory']}")
    logger.info(f"   Recommendations: {performance_report['recommendations']}")
    
    return True


async def demonstrate_async_resource_management():
    """Demonstrate advanced async resource management"""
    logger.info("🔄 Demonstrating Async Resource Management")
    
    # Use managed async resources context
    async with managed_async_resources():
        # Get task manager
        task_manager = await get_task_manager()
        
        # Submit multiple concurrent tasks
        async def sample_task(task_id: int, delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Task {task_id} completed after {delay}s"
        
        # Create multiple tasks
        tasks = [
            sample_task(i, 0.1 + i * 0.05) 
            for i in range(10)
        ]
        
        # Submit batch of tasks
        logger.info("Submitting batch of tasks...")
        results = await task_manager.submit_batch(tasks)
        
        successful_tasks = [r for r in results if isinstance(r, str)]
        logger.info(f"✅ Completed {len(successful_tasks)}/{len(tasks)} tasks successfully")
        
        # Check task manager status
        status = task_manager.get_status()
        logger.info(f"📊 Task manager status: {status}")
    
    return True


async def demonstrate_integrated_agent():
    """Demonstrate an agent using all enhanced features"""
    logger.info("🤖 Demonstrating Integrated Enhanced Agent")
    
    try:
        # Create type-safe configuration
        model_config = create_typed_config(
            ModelConfig,
            model_name="gpt2",
            device=DeviceType.AUTO,
            torch_dtype="float32",  # Use float32 for compatibility
            max_length=256,
            temperature=0.8,
            top_p=0.9
        )
        
        # Create agent with enhanced features
        agent = MultiTurnAgent(model_config)
        await agent.initialize()
        
        # Create environment
        scenarios = [
            {
                "user_responses": [
                    "Hello!",
                    "How are you doing today?",
                    "That's great to hear!"
                ]
            }
        ]
        env = ConversationEnvironment(scenarios=scenarios)
        
        # Create composite reward with error handling
        reward_fn = CompositeReward([
            HelpfulnessReward(weight=0.6),
            SafetyReward(weight=0.4)
        ])
        
        # Performance-optimized conversation
        with PerformanceOptimizer(OptimizationLevel.BALANCED).memory_monitor.memory_context("conversation"):
            conversation_turns = [
                {"role": "user", "content": "Hello! How can you help me today?"}
            ]
            
            response = await agent.generate_response(conversation_turns)
            logger.info(f"🗣️ Agent response: {response}")
            
            # Evaluate response
            from stateset_agents.core.trajectory import ConversationTurn
            turn = ConversationTurn(
                user_message="Hello! How can you help me today?",
                agent_response=response,
                timestamp=datetime.now().timestamp()
            )
            
            reward_result = await reward_fn.compute_turn_reward(turn)
            logger.info(f"🏆 Reward score: {reward_result.score:.3f}")
            logger.info(f"📋 Reward breakdown: {reward_result.breakdown}")
        
        logger.info("✅ Integrated agent demonstration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated agent demonstration failed: {e}")
        return False


async def run_comprehensive_demo():
    """Run comprehensive demonstration of all enhancements"""
    logger.info("🚀 Starting Enhanced GRPO Framework Demonstration")
    
    demo_functions = [
        ("Type Safety & Validation", demonstrate_type_safety),
        ("Error Handling & Resilience", demonstrate_error_handling),
        ("Performance Optimization", demonstrate_performance_optimization),
        ("Async Resource Management", demonstrate_async_resource_management),
        ("Integrated Enhanced Agent", demonstrate_integrated_agent)
    ]
    
    results = {}
    
    for demo_name, demo_func in demo_functions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {demo_name}")
        logger.info(f"{'='*50}")
        
        try:
            start_time = asyncio.get_event_loop().time()
            success = await demo_func()
            end_time = asyncio.get_event_loop().time()
            
            duration = end_time - start_time
            results[demo_name] = {
                "success": success,
                "duration": duration
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{status} {demo_name} ({duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"❌ {demo_name} failed with exception: {e}")
            results[demo_name] = {
                "success": False,
                "error": str(e)
            }
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 DEMONSTRATION SUMMARY")
    logger.info(f"{'='*50}")
    
    total_demos = len(demo_functions)
    successful_demos = sum(1 for r in results.values() if r["success"])
    
    logger.info(f"Total demonstrations: {total_demos}")
    logger.info(f"Successful: {successful_demos}")
    logger.info(f"Failed: {total_demos - successful_demos}")
    logger.info(f"Success rate: {successful_demos/total_demos*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    for demo_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        duration = result.get("duration", 0)
        logger.info(f"  {status} {demo_name}: {duration:.2f}s")
        
        if not result["success"] and "error" in result:
            logger.info(f"      Error: {result['error']}")
    
    logger.info(f"\n🎉 Enhanced GRPO Framework v0.3.0 demonstration completed!")
    
    return successful_demos == total_demos


if __name__ == "__main__":
    # Run the comprehensive demonstration
    success = asyncio.run(run_comprehensive_demo())
    
    if success:
        print("\n🎉 All demonstrations passed successfully!")
        exit(0)
    else:
        print("\n⚠️ Some demonstrations failed. Check logs for details.")
        exit(1)