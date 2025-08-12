#!/usr/bin/env python3
"""
GSPO Framework Quick Start Runner

This script provides a quick way to run and explore the Group Sequence Parameter 
Optimization (GSPO) framework with different demo modes and configurations.

GSPO Innovation:
- Group Sequence Parameter Optimization extends GRPO with intelligent sequence grouping
- Multi-level optimization: sequence → intra-group → inter-group → global
- Adaptive grouping strategies based on similarity, performance, domain, or learning patterns
- Cross-group knowledge transfer for faster convergence
- Parameter sequence optimization for improved learning dynamics
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from examples.comprehensive_gspo_demo import ComprehensiveGSPODemo


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


async def run_quick_gspo_demo():
    """Run a quick GSPO demonstration focusing on key features"""
    print("🚀 Running Quick GSPO Demo...")
    print("   Focus: Sequence grouping and basic parameter optimization")
    
    demo = ComprehensiveGSPODemo()
    
    try:
        await demo.initialize_framework()
        
        # Quick demos
        print("\n📊 Demonstrating Sequence Grouping...")
        await demo.demo_sequence_grouping_strategies()
        
        print("\n⚙️ Demonstrating Parameter Optimization...")
        await demo.demo_parameter_optimization_levels()
        
        # Generate quick summary
        print("\n" + "=" * 50)
        print("📋 Quick Demo Summary")
        print("=" * 50)
        
        if "sequence_grouping" in demo.demo_results:
            sg = demo.demo_results["sequence_grouping"]
            print(f"✅ Sequence Grouping: {sg['total_sequences_created']} sequences → {sg['total_groups_created']} groups")
        
        if "parameter_optimization" in demo.demo_results:
            po = demo.demo_results["parameter_optimization"]
            improvements = len(po["performance_improvements"])
            successful = len([r for r in po["performance_improvements"] if r["improvement"] > 0])
            print(f"✅ Parameter Optimization: {successful}/{improvements} sequences improved")
        
        print("\n🎯 GSPO Key Features Demonstrated:")
        print("  • Intelligent sequence grouping")
        print("  • Multi-level parameter optimization")
        print("  • Performance-driven adaptation")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


async def run_grouping_strategy_demo():
    """Focus on different grouping strategies"""
    print("🚀 Running Grouping Strategy Demo...")
    print("   Focus: Comparison of different sequence grouping approaches")
    
    demo = ComprehensiveGSPODemo()
    
    try:
        await demo.initialize_framework()
        
        print("\n📊 Testing Sequence Grouping Strategies...")
        await demo.demo_sequence_grouping_strategies()
        
        print("\n🧬 Testing Adaptive Evolution...")
        await demo.demo_adaptive_grouping_evolution()
        
        # Results summary
        print("\n" + "=" * 50)
        print("📋 Grouping Strategy Results")
        print("=" * 50)
        
        if "sequence_grouping" in demo.demo_results:
            sg = demo.demo_results["sequence_grouping"]
            print(f"📊 Grouping Efficiency: {sg['sequences_per_group']:.2f} sequences/group")
            
            for group in sg["group_details"]:
                print(f"  • {group['name']}: {group['size']} sequences")
        
        if "adaptive_evolution" in demo.demo_results:
            ae = demo.demo_results["adaptive_evolution"]
            print(f"🧬 Group Evolution: {ae['initial_groups']} → {ae['final_groups']} groups")
            print(f"   Adaptation: {'Yes' if ae['group_changes']['adaptation_occurred'] else 'No'}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


async def run_knowledge_transfer_demo():
    """Focus on cross-group knowledge transfer"""
    print("🚀 Running Knowledge Transfer Demo...")
    print("   Focus: Cross-group learning and parameter transfer")
    
    demo = ComprehensiveGSPODemo()
    
    try:
        await demo.initialize_framework()
        
        # Need groups first
        print("\n📊 Setting up sequence groups...")
        await demo.demo_sequence_grouping_strategies()
        
        print("\n🔄 Testing Knowledge Transfer...")
        await demo.demo_knowledge_transfer()
        
        # Results summary
        print("\n" + "=" * 50)
        print("📋 Knowledge Transfer Results")
        print("=" * 50)
        
        if "knowledge_transfer" in demo.demo_results:
            kt = demo.demo_results["knowledge_transfer"]
            success_rate = kt["successful_transfers"] / max(1, kt["transfer_attempts"])
            print(f"🎯 Transfer Success Rate: {success_rate:.1%}")
            print(f"📊 Successful Transfers: {kt['successful_transfers']}")
            print(f"🔄 Transfer Attempts: {kt['transfer_attempts']}")
            
            if kt["transfer_effectiveness"]:
                print("\n💡 Effective Transfers:")
                for transfer in kt["transfer_effectiveness"]:
                    print(f"  • {transfer['source_group']} → {transfer['target_group']}")
                    print(f"    Potential: {transfer['transfer_potential']:.3f}")
                    print(f"    Parameters: {', '.join(transfer['transferred_parameters'])}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


async def run_comprehensive_demo():
    """Run the full comprehensive GSPO demonstration"""
    print("🚀 Running Comprehensive GSPO Demo...")
    print("   Focus: Complete GSPO framework capabilities")
    
    demo = ComprehensiveGSPODemo()
    
    try:
        await demo.run_comprehensive_demo()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


async def run_performance_analysis():
    """Run performance-focused analysis"""
    print("🚀 Running GSPO Performance Analysis...")
    print("   Focus: Performance optimization and metrics")
    
    demo = ComprehensiveGSPODemo()
    
    try:
        await demo.initialize_framework()
        
        print("\n⚙️ Analyzing Parameter Optimization...")
        await demo.demo_parameter_optimization_levels()
        
        print("\n🚀 Running Workflow Analysis...")
        await demo.demo_comprehensive_gspo_workflow()
        
        # Performance summary
        print("\n" + "=" * 50)
        print("📋 Performance Analysis Results")
        print("=" * 50)
        
        if "parameter_optimization" in demo.demo_results:
            po = demo.demo_results["parameter_optimization"]
            if po["performance_improvements"]:
                improvements = [r["improvement"] for r in po["performance_improvements"]]
                positive_improvements = [imp for imp in improvements if imp > 0]
                
                print(f"📊 Optimization Statistics:")
                print(f"  • Total sequences: {len(improvements)}")
                print(f"  • Improved sequences: {len(positive_improvements)}")
                print(f"  • Success rate: {len(positive_improvements)/len(improvements):.1%}")
                
                if positive_improvements:
                    import numpy as np
                    print(f"  • Average improvement: {np.mean(positive_improvements):.4f}")
                    print(f"  • Max improvement: {np.max(positive_improvements):.4f}")
        
        if "comprehensive_workflow" in demo.demo_results:
            cw = demo.demo_results["comprehensive_workflow"]
            print(f"\n🚀 Workflow Performance:")
            print(f"  • Scenarios processed: {cw['scenarios_processed']}")
            print(f"  • Total interactions: {cw['total_interactions']}")
            print(f"  • Optimization cycles: {cw['optimization_cycles']}")
            
            overall_perf = cw["overall_performance"]["average_across_domains"]
            consistency = cw["overall_performance"]["cross_domain_consistency"]
            print(f"  • Average performance: {overall_perf:.3f}")
            print(f"  • Cross-domain consistency: {consistency:.2f}")
            
            if "final_insights" in cw:
                insights = cw["final_insights"]
                print(f"\n💡 Optimization Insights:")
                print(f"  • Sequences processed: {insights['total_sequences_processed']}")
                print(f"  • Groups created: {insights['groups_created']}")
                print(f"  • Knowledge transfers: {insights['successful_knowledge_transfers']}")
                print(f"  • Group efficiency: {insights['group_efficiency']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


def save_results_to_file(demo_results: dict, filename: str):
    """Save demonstration results to JSON file"""
    with open(filename, 'w') as f:
        # Convert any datetime objects to strings for JSON serialization
        serializable_results = json.loads(json.dumps(demo_results, default=str))
        json.dump(serializable_results, f, indent=2)
    print(f"📝 Results saved to: {filename}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="GSPO Framework Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo Modes:
  quick       - Quick demonstration of key GSPO features (5-7 minutes)
  grouping    - Focus on sequence grouping strategies (8-10 minutes)
  transfer    - Focus on knowledge transfer between groups (6-8 minutes)
  comprehensive - Complete GSPO demonstration (15-20 minutes)
  performance - Performance analysis and optimization metrics (10-12 minutes)

GSPO Framework Features:
  • Group Sequence Parameter Optimization
  • Multi-level optimization (sequence → intra-group → inter-group)
  • Adaptive grouping strategies
  • Cross-group knowledge transfer
  • Parameter sequence evolution

Examples:
  python run_gspo_demo.py quick
  python run_gspo_demo.py grouping --log-level DEBUG
  python run_gspo_demo.py transfer --save-results gspo_transfer_results.json
  python run_gspo_demo.py comprehensive --log-file gspo_full.log
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["quick", "grouping", "transfer", "comprehensive", "performance"],
        help="GSPO demonstration mode to run"
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
    
    parser.add_argument(
        "--save-results",
        help="Save demo results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Map modes to functions
    mode_functions = {
        "quick": run_quick_gspo_demo,
        "grouping": run_grouping_strategy_demo,
        "transfer": run_knowledge_transfer_demo,
        "comprehensive": run_comprehensive_demo,
        "performance": run_performance_analysis
    }
    
    # Run selected demo
    try:
        print(f"🔬 GSPO Framework - {args.mode.title()} Demo")
        print("=" * 60)
        print("Group Sequence Parameter Optimization (GSPO)")
        print("Extending GRPO with intelligent sequence grouping and multi-level optimization")
        print("=" * 60)
        
        asyncio.run(mode_functions[args.mode]())
        
        print(f"\n✅ {args.mode.title()} demo completed successfully!")
        
        if args.log_file:
            print(f"📝 Detailed logs saved to: {args.log_file}")
        
        print("\n🔬 GSPO Framework Extends GRPO With:")
        print("  • Intelligent sequence grouping (similarity, domain, performance)")
        print("  • Multi-level parameter optimization")
        print("  • Cross-group knowledge transfer")
        print("  • Adaptive grouping evolution")
        print("  • Parameter sequence learning")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
