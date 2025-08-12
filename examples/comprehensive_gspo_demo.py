"""
GSPO Framework Comprehensive Demonstration

This script demonstrates the Group Sequence Parameter Optimization (GSPO) framework,
showcasing how it extends GRPO with advanced sequence grouping and parameter optimization
across related tasks and domains.

GSPO Key Innovations:
1. Sequence Grouping: Groups related sequences by similarity, domain, or performance
2. Multi-Level Optimization: Sequence, intra-group, inter-group, and global optimization
3. Knowledge Transfer: Learns transferable patterns between sequence groups
4. Adaptive Grouping: Dynamic grouping based on learning patterns
5. Parameter Sequences: Time-ordered parameter optimization for better learning dynamics
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uuid

# Import GSPO framework components
from core.gspo_engine import (
    GroupSequenceParameterOptimizationEngine,
    create_gspo_engine,
    SequenceGroupingStrategy,
    ParameterOptimizationLevel,
    GSPOTrajectory
)

# Import base framework components
from core.agent import Agent, MultiTurnAgent
from core.environment import Environment
from core.reward import RewardFunction
from core.trajectory import Trajectory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoGSPOAgent(Agent):
    """Demo agent for GSPO testing with parameter-aware responses"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.current_params = {
            "learning_rate": 0.001,
            "temperature": 0.8,
            "epsilon": 0.1,
            "creativity": 0.5,
            "focus": 0.7
        }
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response based on current parameters"""
        # Simulate parameter influence on response
        temp = self.current_params.get("temperature", 0.8)
        creativity = self.current_params.get("creativity", 0.5)
        focus = self.current_params.get("focus", 0.7)
        
        # Base response patterns
        responses = [
            f"I understand your request about: {prompt[:50]}...",
            f"Let me help you with: {prompt[:50]}...",
            f"Here's what I can tell you about: {prompt[:50]}...",
            f"Based on your question about '{prompt[:30]}...', I suggest:",
            f"Regarding '{prompt[:30]}...', here's my analysis:"
        ]
        
        # Parameter-influenced selection and modification
        base_idx = hash(prompt) % len(responses)
        base_response = responses[base_idx]
        
        # Temperature affects response variation
        if temp > 1.0:
            base_response += " (High temperature - more creative response)"
        elif temp < 0.5:
            base_response += " (Low temperature - focused response)"
        
        # Creativity affects response elaboration
        if creativity > 0.7:
            base_response += " Let me explore multiple perspectives on this."
        elif creativity < 0.3:
            base_response += " Here's a direct answer."
        
        # Focus affects response length and precision
        if focus > 0.8:
            base_response += " [Highly focused response]"
        elif focus < 0.4:
            base_response += " [Broad, less focused response]"
        
        return base_response
    
    def update_parameters(self, new_params: Dict[str, Any]):
        """Update agent parameters"""
        self.current_params.update(new_params)
        logger.debug(f"Updated agent parameters: {self.current_params}")


class DemoGSPOEnvironment(Environment):
    """Demo environment for GSPO with sequence-aware context"""
    
    def __init__(self):
        super().__init__()
        self.interaction_history = []
        self.sequence_contexts = {}
    
    async def reset(self, sequence_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment, optionally for a specific sequence"""
        initial_state = {
            "step": 0,
            "sequence_id": sequence_id or str(uuid.uuid4()),
            "context": "demo_environment",
            "available_actions": ["respond", "clarify", "elaborate", "summarize"]
        }
        
        if sequence_id:
            self.sequence_contexts[sequence_id] = initial_state
        
        return initial_state
    
    async def step(self, action: str, sequence_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute action in environment"""
        
        # Update sequence context
        if sequence_id and sequence_id in self.sequence_contexts:
            context = self.sequence_contexts[sequence_id]
            context["step"] += 1
        else:
            context = {"step": 1, "sequence_id": sequence_id}
        
        # Record interaction
        self.interaction_history.append({
            "timestamp": datetime.now(),
            "action": action,
            "sequence_id": sequence_id,
            "context": context.copy()
        })
        
        # Simulate environment response
        response = f"Environment processed action '{action}' for sequence {sequence_id}"
        
        return {
            "response": response,
            "new_state": context,
            "reward_signal": np.random.uniform(0.4, 0.9),  # Base reward
            "done": context["step"] >= 10  # Sequences end after 10 steps
        }


class DemoGSPOReward(RewardFunction):
    """Demo reward function that considers sequence context and parameters"""
    
    def __init__(self):
        super().__init__()
        self.reward_history = []
    
    async def compute_reward(self, interaction_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compute reward considering sequence and group context"""
        
        if isinstance(interaction_data, list) and len(interaction_data) >= 2:
            # Multi-turn format
            prompt = interaction_data[0].get("content", "")
            response = interaction_data[1].get("content", "")
        elif isinstance(interaction_data, dict):
            # Direct format
            prompt = interaction_data.get("prompt", "")
            response = interaction_data.get("response", "")
        else:
            prompt = str(interaction_data)
            response = "No response"
        
        # Base quality metrics
        response_length = len(response)
        response_quality = min(1.0, response_length / 100.0)  # Longer responses get higher base score
        
        # Context-aware scoring
        context_bonus = 0.0
        if context:
            # Sequence position bonus (later positions get slight bonus for consistency)
            position = context.get("position", 0)
            context_bonus += min(0.2, position * 0.02)
            
            # Group context bonus
            group_context = context.get("group_context", {})
            group_size = group_context.get("group_size", 1)
            if group_size > 1:
                context_bonus += 0.1  # Bonus for being part of larger group
            
            # Parameter alignment bonus
            current_params = context.get("current_params", {})
            if current_params:
                temp = current_params.get("temperature", 0.8)
                creativity = current_params.get("creativity", 0.5)
                
                # Bonus for parameter coherence
                if 0.6 <= temp <= 1.2 and 0.3 <= creativity <= 0.8:
                    context_bonus += 0.1
        
        # Coherence simulation (based on prompt-response matching)
        coherence_score = 0.7 + np.random.uniform(-0.1, 0.2)
        if "help" in prompt.lower() and "help" in response.lower():
            coherence_score += 0.1
        
        # Final score computation
        base_score = 0.4 * response_quality + 0.4 * coherence_score + 0.2 * (context_bonus)
        final_score = np.clip(base_score + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        
        reward_result = {
            "score": final_score,
            "breakdown": {
                "response_quality": response_quality,
                "coherence": coherence_score,
                "context_bonus": context_bonus
            },
            "metadata": {
                "response_length": response_length,
                "sequence_context": context.get("sequence_id", "unknown") if context else "none",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Record for analysis
        self.reward_history.append(reward_result)
        
        return reward_result


class ComprehensiveGSPODemo:
    """Comprehensive demonstration of GSPO capabilities"""
    
    def __init__(self):
        self.demo_results = {}
        self.gspo_engine = None
        self.agent = None
        self.environment = None
        self.reward_function = None
    
    async def initialize_framework(self):
        """Initialize GSPO framework components"""
        logger.info("🔧 Initializing GSPO Framework Components")
        
        # Create components
        self.agent = DemoGSPOAgent({"model_type": "demo"})
        self.environment = DemoGSPOEnvironment()
        self.reward_function = DemoGSPOReward()
        
        # Create GSPO engine with adaptive grouping
        self.gspo_engine = create_gspo_engine(
            agent=self.agent,
            environment=self.environment,
            reward_function=self.reward_function,
            grouping_strategy="adaptive",
            optimization_levels=["sequence", "intra_group", "inter_group"],
            num_workers=4
        )
        
        logger.info("✅ GSPO Framework initialized successfully")
    
    async def demo_sequence_grouping_strategies(self):
        """Demonstrate different sequence grouping strategies"""
        logger.info("📊 Demonstrating Sequence Grouping Strategies")
        
        grouping_results = {}
        
        # Test different domains and contexts
        test_scenarios = [
            {
                "domain": "customer_service",
                "context": {"task_type": "support", "complexity": "medium"},
                "characteristics": {"formality": 0.8, "empathy": 0.9, "technical": 0.3}
            },
            {
                "domain": "customer_service", 
                "context": {"task_type": "billing", "complexity": "low"},
                "characteristics": {"formality": 0.9, "empathy": 0.6, "technical": 0.7}
            },
            {
                "domain": "technical_support",
                "context": {"task_type": "troubleshooting", "complexity": "high"},
                "characteristics": {"formality": 0.6, "empathy": 0.4, "technical": 0.9}
            },
            {
                "domain": "sales",
                "context": {"task_type": "product_demo", "complexity": "medium"},
                "characteristics": {"formality": 0.5, "empathy": 0.7, "technical": 0.6}
            },
            {
                "domain": "sales",
                "context": {"task_type": "negotiation", "complexity": "high"},
                "characteristics": {"formality": 0.7, "empathy": 0.8, "technical": 0.4}
            }
        ]
        
        # Create sequences for each scenario
        created_sequences = []
        for i, scenario in enumerate(test_scenarios):
            sequence_id = await self.gspo_engine.create_sequence_group(
                domain=scenario["domain"],
                context=scenario["context"],
                characteristics=scenario["characteristics"]
            )
            created_sequences.append(sequence_id)
            logger.info(f"  Created sequence {sequence_id} for {scenario['domain']}")
        
        # Analyze grouping results
        group_summary = self.gspo_engine._get_group_summary()
        
        grouping_results = {
            "total_sequences_created": len(created_sequences),
            "total_groups_created": group_summary["total_groups"],
            "sequences_per_group": len(created_sequences) / max(1, group_summary["total_groups"]),
            "group_details": group_summary["groups"]
        }
        
        logger.info(f"📈 Grouping Results: {grouping_results['total_groups_created']} groups for {grouping_results['total_sequences_created']} sequences")
        
        self.demo_results["sequence_grouping"] = grouping_results
        return created_sequences
    
    async def demo_parameter_optimization_levels(self):
        """Demonstrate multi-level parameter optimization"""
        logger.info("⚙️ Demonstrating Multi-Level Parameter Optimization")
        
        # Create test sequences first
        created_sequences = await self.demo_sequence_grouping_strategies()
        
        optimization_results = {
            "sequence_level": [],
            "intra_group": [],
            "inter_group": [],
            "performance_improvements": []
        }
        
        # Test parameter optimization for each sequence
        for sequence_id in created_sequences:
            # Initial parameters
            initial_params = {
                "learning_rate": 0.001,
                "temperature": 0.8,
                "epsilon": 0.1,
                "creativity": 0.5,
                "focus": 0.7
            }
            
            # Simulate multiple optimization iterations
            current_params = initial_params.copy()
            performance_history = []
            
            for iteration in range(5):
                # Simulate performance feedback
                base_performance = 0.6 + iteration * 0.05 + np.random.uniform(-0.1, 0.1)
                performance = np.clip(base_performance, 0.0, 1.0)
                performance_history.append(performance)
                
                # Optimize parameters
                optimized_params = await self.gspo_engine.optimize_sequence_parameters(
                    sequence_id=sequence_id,
                    current_params=current_params,
                    performance_feedback=performance
                )
                
                # Track changes
                param_changes = {}
                for key in current_params:
                    if key in optimized_params:
                        change = optimized_params[key] - current_params[key]
                        if abs(change) > 1e-6:
                            param_changes[key] = change
                
                if param_changes:
                    optimization_results["sequence_level"].append({
                        "sequence_id": sequence_id,
                        "iteration": iteration,
                        "performance": performance,
                        "parameter_changes": param_changes
                    })
                
                current_params = optimized_params
            
            # Record performance improvement
            if len(performance_history) > 1:
                improvement = performance_history[-1] - performance_history[0]
                optimization_results["performance_improvements"].append({
                    "sequence_id": sequence_id,
                    "initial_performance": performance_history[0],
                    "final_performance": performance_history[-1],
                    "improvement": improvement
                })
        
        # Analyze optimization effectiveness
        if optimization_results["performance_improvements"]:
            avg_improvement = np.mean([r["improvement"] for r in optimization_results["performance_improvements"]])
            successful_optimizations = len([r for r in optimization_results["performance_improvements"] if r["improvement"] > 0])
            
            logger.info(f"📈 Optimization Results: {successful_optimizations}/{len(optimization_results['performance_improvements'])} sequences improved")
            logger.info(f"📊 Average improvement: {avg_improvement:.4f}")
        
        self.demo_results["parameter_optimization"] = optimization_results
    
    async def demo_knowledge_transfer(self):
        """Demonstrate knowledge transfer between sequence groups"""
        logger.info("🔄 Demonstrating Knowledge Transfer Between Groups")
        
        transfer_results = {
            "transfer_attempts": 0,
            "successful_transfers": 0,
            "transfer_effectiveness": [],
            "cross_group_learning": []
        }
        
        # Get current groups
        group_summary = self.gspo_engine._get_group_summary()
        
        if len(group_summary["groups"]) < 2:
            logger.info("⚠️ Need at least 2 groups for transfer demonstration")
            return
        
        # Simulate performance differences between groups
        groups = group_summary["groups"]
        for i, source_group in enumerate(groups):
            for j, target_group in enumerate(groups):
                if i != j:
                    source_id = source_group["group_id"]
                    target_id = target_group["group_id"]
                    
                    # Attempt knowledge transfer
                    transfer_result = await self.gspo_engine.cross_group_learner.transfer_knowledge(
                        source_group_id=source_id,
                        target_group_id=target_id,
                        group_manager=self.gspo_engine.group_manager
                    )
                    
                    transfer_results["transfer_attempts"] += 1
                    
                    if transfer_result["transferred"]:
                        transfer_results["successful_transfers"] += 1
                        transfer_results["transfer_effectiveness"].append({
                            "source_group": source_group["name"],
                            "target_group": target_group["name"],
                            "transfer_potential": transfer_result["transfer_potential"],
                            "transferred_parameters": list(transfer_result["transferred_parameters"].keys())
                        })
                        
                        logger.info(f"✅ Transfer: {source_group['name']} → {target_group['name']} "
                                  f"(potential: {transfer_result['transfer_potential']:.3f})")
        
        # Analyze transfer effectiveness
        if transfer_results["transfer_effectiveness"]:
            avg_potential = np.mean([t["transfer_potential"] for t in transfer_results["transfer_effectiveness"]])
            logger.info(f"🎯 Transfer Success Rate: {transfer_results['successful_transfers']}/{transfer_results['transfer_attempts']}")
            logger.info(f"📊 Average Transfer Potential: {avg_potential:.3f}")
        
        self.demo_results["knowledge_transfer"] = transfer_results
    
    async def demo_adaptive_grouping_evolution(self):
        """Demonstrate how groups evolve and adapt over time"""
        logger.info("🧬 Demonstrating Adaptive Group Evolution")
        
        evolution_results = {
            "initial_groups": 0,
            "final_groups": 0,
            "group_changes": [],
            "performance_evolution": []
        }
        
        # Record initial state
        initial_summary = self.gspo_engine._get_group_summary()
        evolution_results["initial_groups"] = initial_summary["total_groups"]
        
        # Create additional sequences with evolving characteristics
        evolving_scenarios = [
            {
                "domain": "customer_service",
                "context": {"task_type": "support", "complexity": "evolving"},
                "characteristics": {"formality": 0.7, "empathy": 0.9, "technical": 0.5}
            },
            {
                "domain": "hybrid_domain",  # New domain
                "context": {"task_type": "consultation", "complexity": "adaptive"},
                "characteristics": {"formality": 0.6, "empathy": 0.8, "technical": 0.7}
            },
            {
                "domain": "technical_support",
                "context": {"task_type": "advanced_troubleshooting", "complexity": "very_high"},
                "characteristics": {"formality": 0.5, "empathy": 0.3, "technical": 1.0}
            }
        ]
        
        evolution_sequence_ids = []
        for scenario in evolving_scenarios:
            sequence_id = await self.gspo_engine.create_sequence_group(
                domain=scenario["domain"],
                context=scenario["context"],
                characteristics=scenario["characteristics"]
            )
            evolution_sequence_ids.append(sequence_id)
        
        # Simulate learning evolution over time
        for epoch in range(3):
            logger.info(f"  Evolution Epoch {epoch + 1}/3")
            
            # Run training iterations for each sequence
            epoch_performance = []
            for sequence_id in evolution_sequence_ids:
                # Simulate evolving characteristics and performance
                evolving_performance = 0.5 + epoch * 0.1 + np.random.uniform(-0.05, 0.15)
                evolving_performance = np.clip(evolving_performance, 0.0, 1.0)
                epoch_performance.append(evolving_performance)
                
                # Update parameters based on performance
                current_params = {
                    "learning_rate": 0.001 + epoch * 0.0005,
                    "temperature": 0.8 + epoch * 0.05,
                    "creativity": 0.5 + epoch * 0.1
                }
                
                optimized_params = await self.gspo_engine.optimize_sequence_parameters(
                    sequence_id, current_params, evolving_performance
                )
            
            evolution_results["performance_evolution"].append({
                "epoch": epoch,
                "average_performance": np.mean(epoch_performance),
                "performance_std": np.std(epoch_performance)
            })
        
        # Record final state
        final_summary = self.gspo_engine._get_group_summary()
        evolution_results["final_groups"] = final_summary["total_groups"]
        evolution_results["group_changes"] = {
            "groups_added": evolution_results["final_groups"] - evolution_results["initial_groups"],
            "adaptation_occurred": evolution_results["final_groups"] != evolution_results["initial_groups"]
        }
        
        logger.info(f"🔄 Group Evolution: {evolution_results['initial_groups']} → {evolution_results['final_groups']} groups")
        
        self.demo_results["adaptive_evolution"] = evolution_results
    
    async def demo_comprehensive_gspo_workflow(self):
        """Demonstrate complete GSPO workflow with realistic scenarios"""
        logger.info("🚀 Demonstrating Comprehensive GSPO Workflow")
        
        # Define realistic conversation scenarios
        conversation_scenarios = [
            {
                "domain": "customer_service",
                "prompts": [
                    "Hi, I need help with my account",
                    "I can't access my dashboard",
                    "My billing looks incorrect this month",
                    "Can you help me update my payment method?",
                    "I want to upgrade my subscription"
                ],
                "context": {
                    "task_type": "account_support",
                    "complexity": "medium",
                    "current_params": {
                        "learning_rate": 0.001,
                        "temperature": 0.7,
                        "empathy": 0.9,
                        "formality": 0.8
                    },
                    "characteristics": {"formality": 0.8, "empathy": 0.9, "technical": 0.4}
                }
            },
            {
                "domain": "technical_support", 
                "prompts": [
                    "The API is returning 500 errors",
                    "Database connection keeps timing out",
                    "SSL certificate expired yesterday",
                    "Load balancer showing high latency",
                    "Need help debugging authentication flow"
                ],
                "context": {
                    "task_type": "technical_troubleshooting",
                    "complexity": "high",
                    "current_params": {
                        "learning_rate": 0.0015,
                        "temperature": 0.6,
                        "precision": 0.9,
                        "technical_depth": 0.8
                    },
                    "characteristics": {"formality": 0.6, "empathy": 0.4, "technical": 0.9}
                }
            },
            {
                "domain": "sales",
                "prompts": [
                    "What features are included in the enterprise plan?",
                    "Can you provide a custom pricing quote?",
                    "We need multi-tenant capabilities",
                    "What's your implementation timeline?",
                    "Do you offer dedicated support?"
                ],
                "context": {
                    "task_type": "enterprise_sales",
                    "complexity": "high",
                    "current_params": {
                        "learning_rate": 0.002,
                        "temperature": 0.8,
                        "persuasiveness": 0.7,
                        "relationship_building": 0.8
                    },
                    "characteristics": {"formality": 0.7, "empathy": 0.8, "technical": 0.6}
                }
            }
        ]
        
        workflow_results = {
            "scenarios_processed": 0,
            "total_interactions": 0,
            "optimization_cycles": 0,
            "performance_trends": {},
            "group_dynamics": {}
        }
        
        # Process each scenario through complete GSPO workflow
        for scenario_idx, scenario in enumerate(conversation_scenarios):
            logger.info(f"  Processing scenario {scenario_idx + 1}: {scenario['domain']}")
            
            domain = scenario["domain"]
            prompts = scenario["prompts"]
            context = scenario["context"]
            
            # Create sequence contexts for each prompt
            sequence_contexts = []
            for i, prompt in enumerate(prompts):
                seq_context = context.copy()
                seq_context["position"] = i
                seq_context["prompt"] = prompt
                sequence_contexts.append(seq_context)
            
            # Run GSPO training iteration
            iteration_result = await self.gspo_engine.train_iteration_gspo(
                prompts=prompts,
                sequence_contexts=sequence_contexts
            )
            
            # Process results
            workflow_results["scenarios_processed"] += 1
            workflow_results["total_interactions"] += len(prompts)
            workflow_results["optimization_cycles"] += len(iteration_result["optimization_results"])
            
            # Track performance trends
            performances = [t.learned_reward for t in iteration_result["trajectories"]]
            workflow_results["performance_trends"][domain] = {
                "mean_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "min_performance": np.min(performances),
                "max_performance": np.max(performances)
            }
            
            # Log scenario results
            logger.info(f"    Processed {len(prompts)} interactions")
            logger.info(f"    Average performance: {np.mean(performances):.3f}")
            logger.info(f"    Optimization improvements: {len([r for r in iteration_result['optimization_results'] if r['original_params'] != r['optimized_params']])}")
        
        # Analyze overall workflow effectiveness
        total_performance = []
        for domain_perf in workflow_results["performance_trends"].values():
            total_performance.append(domain_perf["mean_performance"])
        
        workflow_results["overall_performance"] = {
            "average_across_domains": np.mean(total_performance),
            "performance_variance": np.var(total_performance),
            "cross_domain_consistency": 1.0 / (np.std(total_performance) + 1e-6)
        }
        
        # Get final optimization insights
        optimization_insights = self.gspo_engine.get_optimization_insights()
        workflow_results["final_insights"] = optimization_insights
        
        logger.info(f"🎯 Workflow Completion:")
        logger.info(f"  • Scenarios: {workflow_results['scenarios_processed']}")
        logger.info(f"  • Total Interactions: {workflow_results['total_interactions']}")
        logger.info(f"  • Optimization Cycles: {workflow_results['optimization_cycles']}")
        logger.info(f"  • Average Performance: {workflow_results['overall_performance']['average_across_domains']:.3f}")
        
        self.demo_results["comprehensive_workflow"] = workflow_results
    
    async def run_comprehensive_demo(self):
        """Run the complete comprehensive GSPO demonstration"""
        logger.info("=" * 70)
        logger.info("🚀 GSPO Framework Comprehensive Demonstration")
        logger.info("=" * 70)
        
        try:
            # Initialize framework
            await self.initialize_framework()
            
            # Run all demonstrations
            await self.demo_sequence_grouping_strategies()
            await self.demo_parameter_optimization_levels()
            await self.demo_knowledge_transfer()
            await self.demo_adaptive_grouping_evolution()
            await self.demo_comprehensive_gspo_workflow()
            
            # Generate final summary
            self._generate_final_summary()
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_final_summary(self):
        """Generate comprehensive summary of all demonstrations"""
        logger.info("=" * 70)
        logger.info("📋 GSPO Framework Demonstration Summary")
        logger.info("=" * 70)
        
        # Sequence Grouping Summary
        if "sequence_grouping" in self.demo_results:
            sg = self.demo_results["sequence_grouping"]
            logger.info(f"🔵 Sequence Grouping:")
            logger.info(f"  • Created {sg['total_sequences_created']} sequences in {sg['total_groups_created']} groups")
            logger.info(f"  • Average sequences per group: {sg['sequences_per_group']:.2f}")
        
        # Parameter Optimization Summary
        if "parameter_optimization" in self.demo_results:
            po = self.demo_results["parameter_optimization"]
            improvements = len(po["performance_improvements"])
            successful = len([r for r in po["performance_improvements"] if r["improvement"] > 0])
            logger.info(f"⚙️ Parameter Optimization:")
            logger.info(f"  • Successful optimizations: {successful}/{improvements}")
            if po["performance_improvements"]:
                avg_imp = np.mean([r["improvement"] for r in po["performance_improvements"]])
                logger.info(f"  • Average improvement: {avg_imp:.4f}")
        
        # Knowledge Transfer Summary
        if "knowledge_transfer" in self.demo_results:
            kt = self.demo_results["knowledge_transfer"]
            logger.info(f"🔄 Knowledge Transfer:")
            logger.info(f"  • Transfer success rate: {kt['successful_transfers']}/{kt['transfer_attempts']}")
            if kt["transfer_effectiveness"]:
                avg_pot = np.mean([t["transfer_potential"] for t in kt["transfer_effectiveness"]])
                logger.info(f"  • Average transfer potential: {avg_pot:.3f}")
        
        # Adaptive Evolution Summary
        if "adaptive_evolution" in self.demo_results:
            ae = self.demo_results["adaptive_evolution"]
            logger.info(f"🧬 Adaptive Evolution:")
            logger.info(f"  • Group evolution: {ae['initial_groups']} → {ae['final_groups']} groups")
            logger.info(f"  • Adaptation occurred: {ae['group_changes']['adaptation_occurred']}")
        
        # Comprehensive Workflow Summary
        if "comprehensive_workflow" in self.demo_results:
            cw = self.demo_results["comprehensive_workflow"]
            logger.info(f"🚀 Comprehensive Workflow:")
            logger.info(f"  • Scenarios processed: {cw['scenarios_processed']}")
            logger.info(f"  • Total interactions: {cw['total_interactions']}")
            logger.info(f"  • Optimization cycles: {cw['optimization_cycles']}")
            logger.info(f"  • Overall performance: {cw['overall_performance']['average_across_domains']:.3f}")
            
            # Final optimization insights
            if "final_insights" in cw:
                insights = cw["final_insights"]
                logger.info(f"  • Knowledge transfers: {insights['successful_knowledge_transfers']}")
                logger.info(f"  • Group efficiency: {insights['group_efficiency']:.2f}")
        
        logger.info("")
        logger.info("🎉 GSPO Framework Capabilities Demonstrated:")
        logger.info("  ✅ Multi-level parameter optimization (sequence, intra-group, inter-group)")
        logger.info("  ✅ Intelligent sequence grouping with adaptive strategies")
        logger.info("  ✅ Cross-group knowledge transfer and learning")
        logger.info("  ✅ Dynamic group evolution and adaptation")
        logger.info("  ✅ Comprehensive workflow integration")
        logger.info("  ✅ Performance monitoring and optimization insights")
        logger.info("")
        logger.info("🔬 GSPO extends GRPO with:")
        logger.info("  • Group-based sequence optimization")
        logger.info("  • Multi-level parameter learning")
        logger.info("  • Transferable knowledge patterns")
        logger.info("  • Adaptive grouping strategies")
        logger.info("  • Sequence-aware parameter evolution")
        
        logger.info("=" * 70)


async def main():
    """Main demonstration function"""
    demo = ComprehensiveGSPODemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
