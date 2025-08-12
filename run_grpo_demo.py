#!/usr/bin/env python3
"""
GRPO Framework Demo Runner

Quick start script to run the comprehensive GRPO framework demonstration.
This script provides different demo modes for various use cases.
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.comprehensive_grpo_framework_demo import (
    ComprehensiveGRPOFrameworkDemo,
    DemoConfig
)


def setup_logging(level="INFO", log_file=None):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


async def run_quick_demo():
    """Run a quick demonstration (minimal resource usage)"""
    print("🚀 Running Quick GRPO Demo...")
    
    config = DemoConfig(
        num_workers=4,
        batch_size=8,
        num_training_iterations=3,
        use_sentiment_analysis=True,
        enable_parallel_processing=True,
        log_detailed_metrics=False
    )
    
    demo = ComprehensiveGRPOFrameworkDemo(config)
    
    try:
        await demo.initialize_framework()
        
        # Run subset of demos for quick execution
        print("\n📊 Running Computational Training Demo...")
        await demo._demo_computational_training()
        
        print("\n💭 Running Sentiment Analysis Demo...")
        await demo._demo_sentiment_aware_rewards()
        
        print("\n📈 Running Performance Monitoring...")
        await demo._demo_performance_monitoring()
        
        # Generate simplified report
        print("\n📋 Demo Summary")
        print("-" * 40)
        if "computational_training" in demo.training_results:
            ct = demo.training_results["computational_training"]
            print(f"✅ Training: {ct['iterations']} iterations, {ct['average_reward']:.3f} avg reward")
        
        if "sentiment_analysis" in demo.training_results:
            sa = demo.training_results["sentiment_analysis"]
            avg_score = sum(r["score"] for r in sa) / len(sa)
            print(f"✅ Sentiment: {len(sa)} scenarios, {avg_score:.3f} avg score")
        
        print("✅ Quick demo completed successfully!")
        
    finally:
        await demo.cleanup()


async def run_full_demo():
    """Run the complete comprehensive demonstration"""
    print("🚀 Running Full GRPO Framework Demo...")
    
    config = DemoConfig(
        num_workers=8,
        batch_size=16,
        num_training_iterations=10,
        use_sentiment_analysis=True,
        enable_parallel_processing=True,
        log_detailed_metrics=True
    )
    
    demo = ComprehensiveGRPOFrameworkDemo(config)
    
    try:
        await demo.initialize_framework()
        await demo.run_comprehensive_demo()
    finally:
        await demo.cleanup()


async def run_performance_test():
    """Run performance-focused testing"""
    print("🚀 Running GRPO Performance Test...")
    
    config = DemoConfig(
        num_workers=16,  # High parallelism
        batch_size=32,
        num_training_iterations=15,
        use_sentiment_analysis=True,
        enable_parallel_processing=True,
        log_detailed_metrics=True
    )
    
    demo = ComprehensiveGRPOFrameworkDemo(config)
    
    try:
        await demo.initialize_framework()
        
        # Focus on performance-critical demos
        await demo._demo_computational_training()
        await demo._demo_parallel_processing()
        await demo._demo_framework_scaling()
        await demo._demo_performance_monitoring()
        
        print("\n🏆 Performance Test Results")
        print("-" * 50)
        
        # Display performance metrics
        if hasattr(demo, 'performance_metrics'):
            agg = demo.performance_metrics["aggregate_metrics"]
            print(f"• Total Trajectories: {agg['total_trajectories']}")
            print(f"• Computation Time: {agg['total_computation_time']:.2f}s")
            print(f"• Overall Efficiency: {agg['overall_efficiency']:.2f} trajectories/sec")
            print(f"• Worker Pool Size: {agg['total_workers']} workers")
        
        if "parallel_processing" in demo.training_results:
            pp = demo.training_results["parallel_processing"]
            print(f"• Parallel Efficiency: {pp['efficiency_gain']}")
            print(f"• Successful Engines: {pp['successful_engines']}/{pp['total_engines']}")
        
    finally:
        await demo.cleanup()


async def run_sentiment_demo():
    """Run sentiment analysis focused demonstration"""
    print("🚀 Running Sentiment Analysis Demo...")
    
    config = DemoConfig(
        num_workers=6,
        batch_size=12,
        num_training_iterations=5,
        use_sentiment_analysis=True,
        enable_parallel_processing=False,
        log_detailed_metrics=True
    )
    
    demo = ComprehensiveGRPOFrameworkDemo(config)
    
    try:
        await demo.initialize_framework()
        
        # Focus on sentiment analysis features
        await demo._demo_sentiment_aware_rewards()
        await demo._demo_multiturn_conversations()
        
        print("\n💭 Sentiment Analysis Results")
        print("-" * 45)
        
        if "sentiment_analysis" in demo.training_results:
            for result in demo.training_results["sentiment_analysis"]:
                print(f"• {result['scenario']}: {result['score']:.3f}")
        
        if "multiturn_conversation" in demo.training_results:
            mc = demo.training_results["multiturn_conversation"]
            print(f"• Conversation Quality: {mc['average_quality']:.3f}")
            print(f"• Response Time: {mc['average_response_time']:.3f}s")
        
    finally:
        await demo.cleanup()


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="GRPO Framework Demonstration Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo Modes:
  quick       - Quick demonstration (3-5 minutes)
  full        - Complete comprehensive demo (10-15 minutes)
  performance - Performance-focused testing (5-10 minutes)
  sentiment   - Sentiment analysis demonstration (5-8 minutes)

Examples:
  python run_grpo_demo.py quick
  python run_grpo_demo.py full --log-level DEBUG
  python run_grpo_demo.py performance --log-file grpo_perf.log
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["quick", "full", "performance", "sentiment"],
        help="Demonstration mode to run"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Optional log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Map modes to functions
    mode_functions = {
        "quick": run_quick_demo,
        "full": run_full_demo,
        "performance": run_performance_test,
        "sentiment": run_sentiment_demo
    }
    
    # Run selected demo
    try:
        asyncio.run(mode_functions[args.mode]())
        print(f"\n✅ {args.mode.title()} demo completed successfully!")
        if args.log_file:
            print(f"📝 Detailed logs saved to: {args.log_file}")
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
