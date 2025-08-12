"""
GSPO Engine - Group Sequence Parameter Optimization

This module implements Group Sequence Parameter Optimization (GSPO), a novel approach that optimizes
parameters for sequences across groups of related tasks/contexts. GSPO extends GRPO by:

1. Grouping related sequences by similarity, context, or domain
2. Optimizing parameters within and across sequence groups
3. Learning transferable patterns between sequence groups
4. Dynamic parameter adaptation based on sequence characteristics
5. Multi-level optimization (intra-group, inter-group, global)

Key Concepts:
- Sequence Groups: Collections of related sequences with shared characteristics
- Parameter Sequences: Time-ordered parameter configurations for optimization
- Cross-Group Learning: Transfer learning between different sequence groups
- Adaptive Grouping: Dynamic grouping based on performance and similarity
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import logging
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import uuid

from .agent import Agent
from .environment import Environment
from .reward import RewardFunction
from .trajectory import Trajectory
from .computational_engine import ComputationalGRPOEngine, ComputationalTrajectory

logger = logging.getLogger(__name__)


class SequenceGroupingStrategy(Enum):
    """Strategies for grouping sequences"""
    SIMILARITY = "similarity"          # Group by semantic/content similarity
    PERFORMANCE = "performance"        # Group by performance characteristics
    DOMAIN = "domain"                 # Group by domain/context
    TEMPORAL = "temporal"             # Group by time-based patterns
    ADAPTIVE = "adaptive"             # Dynamic grouping based on learning
    HIERARCHICAL = "hierarchical"     # Multi-level hierarchical grouping


class ParameterOptimizationLevel(Enum):
    """Levels of parameter optimization"""
    SEQUENCE = "sequence"             # Individual sequence optimization
    INTRA_GROUP = "intra_group"       # Within group optimization
    INTER_GROUP = "inter_group"       # Between group optimization
    GLOBAL = "global"                 # Global cross-group optimization


@dataclass
class SequenceMetadata:
    """Metadata for a sequence"""
    sequence_id: str
    domain: str
    context: Dict[str, Any]
    characteristics: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    parameter_history: List[Dict[str, Any]] = field(default_factory=list)
    group_assignments: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SequenceGroup:
    """A group of related sequences"""
    group_id: str
    name: str
    sequences: Set[str] = field(default_factory=set)
    characteristics: Dict[str, float] = field(default_factory=dict)
    optimal_parameters: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, float] = field(default_factory=dict)
    transfer_compatibility: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class GSPOTrajectory(ComputationalTrajectory):
    """Extended trajectory for GSPO with sequence and group information"""
    sequence_id: str = ""
    group_id: str = ""
    parameter_sequence: List[Dict[str, Any]] = field(default_factory=list)
    sequence_position: int = 0
    group_context: Dict[str, Any] = field(default_factory=dict)
    cross_group_signals: Dict[str, float] = field(default_factory=dict)


class ParameterSequenceOptimizer:
    """Optimizes parameter sequences for improved learning dynamics"""
    
    def __init__(self, 
                 sequence_length: int = 10,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameter_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
    def optimize_parameter_sequence(self, 
                                   current_params: Dict[str, Any],
                                   performance_feedback: float,
                                   group_context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the next parameter configuration in the sequence"""
        
        # Record current state
        self.parameter_history.append(current_params.copy())
        self.performance_history.append(performance_feedback)
        
        if len(self.parameter_history) < 2:
            return current_params
        
        # Compute parameter gradients based on performance changes
        param_gradients = self._compute_parameter_gradients()
        
        # Apply group-specific adaptations
        adapted_gradients = self._apply_group_adaptations(param_gradients, group_context)
        
        # Update parameters with momentum
        optimized_params = self._update_with_momentum(current_params, adapted_gradients)
        
        return optimized_params
    
    def _compute_parameter_gradients(self) -> Dict[str, float]:
        """Compute gradients for parameters based on performance changes"""
        if len(self.performance_history) < 2:
            return {}
        
        recent_perf = list(self.performance_history)[-2:]
        recent_params = list(self.parameter_history)[-2:]
        
        perf_change = recent_perf[1] - recent_perf[0]
        gradients = {}
        
        for key in recent_params[0]:
            if isinstance(recent_params[0][key], (int, float)):
                param_change = recent_params[1][key] - recent_params[0][key]
                if param_change != 0:
                    gradients[key] = perf_change / param_change
        
        return gradients
    
    def _apply_group_adaptations(self, 
                               gradients: Dict[str, float], 
                               group_context: Dict[str, Any]) -> Dict[str, float]:
        """Apply group-specific adaptations to gradients"""
        adapted = gradients.copy()
        
        # Scale gradients based on group characteristics
        if "learning_scale" in group_context:
            scale = group_context["learning_scale"]
            for key in adapted:
                adapted[key] *= scale
        
        # Apply domain-specific constraints
        if "parameter_bounds" in group_context:
            bounds = group_context["parameter_bounds"]
            for key in adapted:
                if key in bounds:
                    min_val, max_val = bounds[key]
                    adapted[key] = np.clip(adapted[key], min_val, max_val)
        
        return adapted
    
    def _update_with_momentum(self, 
                            current_params: Dict[str, Any], 
                            gradients: Dict[str, float]) -> Dict[str, Any]:
        """Update parameters using momentum-based optimization"""
        updated = current_params.copy()
        
        for key, gradient in gradients.items():
            if key in updated and isinstance(updated[key], (int, float)):
                # Simple momentum update
                update = self.learning_rate * gradient
                updated[key] = current_params[key] + update
        
        return updated


class SequenceGroupManager:
    """Manages grouping and organization of sequences"""
    
    def __init__(self, 
                 grouping_strategy: SequenceGroupingStrategy = SequenceGroupingStrategy.ADAPTIVE,
                 max_groups: int = 10,
                 min_group_size: int = 3):
        self.grouping_strategy = grouping_strategy
        self.max_groups = max_groups
        self.min_group_size = min_group_size
        
        self.sequences: Dict[str, SequenceMetadata] = {}
        self.groups: Dict[str, SequenceGroup] = {}
        self.grouping_model = None
        
    async def add_sequence(self, 
                          sequence_id: str, 
                          domain: str, 
                          context: Dict[str, Any],
                          characteristics: Dict[str, float]) -> str:
        """Add a new sequence and assign it to appropriate groups"""
        
        sequence = SequenceMetadata(
            sequence_id=sequence_id,
            domain=domain,
            context=context,
            characteristics=characteristics
        )
        
        self.sequences[sequence_id] = sequence
        
        # Assign to groups based on strategy
        group_assignments = await self._assign_to_groups(sequence)
        sequence.group_assignments = group_assignments
        
        # Update group membership
        for group_id in group_assignments:
            if group_id not in self.groups:
                await self._create_group(group_id, f"Group_{len(self.groups)}")
            self.groups[group_id].sequences.add(sequence_id)
            self.groups[group_id].last_updated = datetime.now()
        
        return group_assignments[0] if group_assignments else ""
    
    async def _assign_to_groups(self, sequence: SequenceMetadata) -> List[str]:
        """Assign sequence to appropriate groups based on strategy"""
        
        if self.grouping_strategy == SequenceGroupingStrategy.DOMAIN:
            return await self._assign_by_domain(sequence)
        elif self.grouping_strategy == SequenceGroupingStrategy.SIMILARITY:
            return await self._assign_by_similarity(sequence)
        elif self.grouping_strategy == SequenceGroupingStrategy.PERFORMANCE:
            return await self._assign_by_performance(sequence)
        elif self.grouping_strategy == SequenceGroupingStrategy.ADAPTIVE:
            return await self._assign_adaptively(sequence)
        else:
            # Default: create or assign to existing group
            if not self.groups:
                group_id = str(uuid.uuid4())[:8]
                await self._create_group(group_id, "Default_Group")
                return [group_id]
            return [list(self.groups.keys())[0]]
    
    async def _assign_by_domain(self, sequence: SequenceMetadata) -> List[str]:
        """Assign based on domain similarity"""
        domain_groups = {g_id: g for g_id, g in self.groups.items() 
                        if g.name.startswith(sequence.domain)}
        
        if domain_groups:
            # Find best matching domain group
            best_group = max(domain_groups.values(), 
                           key=lambda g: len(g.sequences))
            return [best_group.group_id]
        else:
            # Create new domain group
            group_id = str(uuid.uuid4())[:8]
            await self._create_group(group_id, f"{sequence.domain}_Group")
            return [group_id]
    
    async def _assign_by_similarity(self, sequence: SequenceMetadata) -> List[str]:
        """Assign based on characteristic similarity"""
        if not self.groups:
            group_id = str(uuid.uuid4())[:8]
            await self._create_group(group_id, "Similarity_Group_0")
            return [group_id]
        
        # Compute similarity to existing groups
        similarities = {}
        for group_id, group in self.groups.items():
            similarity = self._compute_similarity(sequence.characteristics, 
                                                group.characteristics)
            similarities[group_id] = similarity
        
        # Assign to most similar group above threshold
        threshold = 0.7
        best_group = max(similarities.items(), key=lambda x: x[1])
        
        if best_group[1] >= threshold:
            return [best_group[0]]
        else:
            # Create new group if no good match
            group_id = str(uuid.uuid4())[:8]
            await self._create_group(group_id, f"Similarity_Group_{len(self.groups)}")
            return [group_id]
    
    async def _assign_adaptively(self, sequence: SequenceMetadata) -> List[str]:
        """Adaptive assignment combining multiple factors"""
        # Use multiple assignment strategies and combine
        domain_assignments = await self._assign_by_domain(sequence)
        
        if len(self.sequences) > 10:  # Only use similarity if we have enough data
            similarity_assignments = await self._assign_by_similarity(sequence)
            
            # Combine assignments (prioritize domain, then similarity)
            combined = domain_assignments.copy()
            for sim_group in similarity_assignments:
                if sim_group not in combined:
                    combined.append(sim_group)
            
            return combined[:2]  # Max 2 group assignments
        
        return domain_assignments
    
    def _compute_similarity(self, chars1: Dict[str, float], chars2: Dict[str, float]) -> float:
        """Compute similarity between characteristic vectors"""
        if not chars1 or not chars2:
            return 0.0
        
        # Get common keys
        common_keys = set(chars1.keys()) & set(chars2.keys())
        if not common_keys:
            return 0.0
        
        # Compute cosine similarity
        vec1 = np.array([chars1[k] for k in common_keys])
        vec2 = np.array([chars2[k] for k in common_keys])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    async def _create_group(self, group_id: str, name: str):
        """Create a new sequence group"""
        self.groups[group_id] = SequenceGroup(
            group_id=group_id,
            name=name
        )
    
    def get_group_context(self, group_id: str) -> Dict[str, Any]:
        """Get context information for a group"""
        if group_id not in self.groups:
            return {}
        
        group = self.groups[group_id]
        return {
            "group_size": len(group.sequences),
            "characteristics": group.characteristics,
            "performance_stats": group.performance_stats,
            "optimal_parameters": group.optimal_parameters,
            "learning_scale": group.performance_stats.get("learning_rate_scale", 1.0),
            "parameter_bounds": self._get_parameter_bounds(group)
        }
    
    def _get_parameter_bounds(self, group: SequenceGroup) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for a group based on historical data"""
        bounds = {}
        
        # Default bounds - would be learned from data in practice
        bounds["learning_rate"] = (1e-5, 1e-2)
        bounds["temperature"] = (0.1, 2.0)
        bounds["epsilon"] = (0.01, 0.5)
        
        return bounds
    
    async def update_group_performance(self, 
                                     group_id: str, 
                                     sequence_id: str, 
                                     performance: float,
                                     parameters: Dict[str, Any]):
        """Update group performance statistics"""
        if group_id not in self.groups:
            return
        
        group = self.groups[group_id]
        
        # Update sequence performance
        if sequence_id in self.sequences:
            self.sequences[sequence_id].performance_history.append(performance)
            self.sequences[sequence_id].parameter_history.append(parameters)
        
        # Update group statistics
        all_performances = []
        for seq_id in group.sequences:
            if seq_id in self.sequences:
                seq_perfs = self.sequences[seq_id].performance_history
                all_performances.extend(seq_perfs)
        
        if all_performances:
            group.performance_stats = {
                "mean_performance": np.mean(all_performances),
                "std_performance": np.std(all_performances),
                "max_performance": np.max(all_performances),
                "min_performance": np.min(all_performances),
                "recent_trend": self._compute_trend(all_performances[-10:])
            }
        
        group.last_updated = datetime.now()
    
    def _compute_trend(self, performances: List[float]) -> float:
        """Compute performance trend (positive = improving)"""
        if len(performances) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(performances))
        y = np.array(performances)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)


class CrossGroupLearningManager:
    """Manages learning and transfer between different sequence groups"""
    
    def __init__(self, transfer_threshold: float = 0.1):
        self.transfer_threshold = transfer_threshold
        self.transfer_history: Dict[Tuple[str, str], List[float]] = {}
        self.compatibility_matrix: Dict[Tuple[str, str], float] = {}
        
    async def compute_transfer_potential(self, 
                                       source_group_id: str, 
                                       target_group_id: str,
                                       group_manager: SequenceGroupManager) -> float:
        """Compute potential for knowledge transfer between groups"""
        
        if source_group_id not in group_manager.groups or target_group_id not in group_manager.groups:
            return 0.0
        
        source_group = group_manager.groups[source_group_id]
        target_group = group_manager.groups[target_group_id]
        
        # Compute compatibility based on multiple factors
        char_similarity = group_manager._compute_similarity(
            source_group.characteristics,
            target_group.characteristics
        )
        
        performance_compatibility = self._compute_performance_compatibility(
            source_group, target_group
        )
        
        domain_compatibility = self._compute_domain_compatibility(
            source_group, target_group, group_manager
        )
        
        # Weighted combination
        transfer_potential = (
            0.4 * char_similarity +
            0.3 * performance_compatibility +
            0.3 * domain_compatibility
        )
        
        return transfer_potential
    
    def _compute_performance_compatibility(self, 
                                         source_group: SequenceGroup, 
                                         target_group: SequenceGroup) -> float:
        """Compute compatibility based on performance characteristics"""
        source_stats = source_group.performance_stats
        target_stats = target_group.performance_stats
        
        if not source_stats or not target_stats:
            return 0.5  # Neutral if no data
        
        # Compare performance distributions
        source_mean = source_stats.get("mean_performance", 0.5)
        target_mean = target_stats.get("mean_performance", 0.5)
        source_std = source_stats.get("std_performance", 0.1)
        target_std = target_stats.get("std_performance", 0.1)
        
        # Compatibility is higher when performance characteristics are similar
        mean_diff = abs(source_mean - target_mean)
        std_diff = abs(source_std - target_std)
        
        compatibility = 1.0 - (mean_diff + std_diff) / 2.0
        return max(0.0, compatibility)
    
    def _compute_domain_compatibility(self, 
                                    source_group: SequenceGroup, 
                                    target_group: SequenceGroup,
                                    group_manager: SequenceGroupManager) -> float:
        """Compute domain-based compatibility"""
        
        # Get representative sequences from each group
        source_domains = set()
        target_domains = set()
        
        for seq_id in source_group.sequences:
            if seq_id in group_manager.sequences:
                source_domains.add(group_manager.sequences[seq_id].domain)
        
        for seq_id in target_group.sequences:
            if seq_id in group_manager.sequences:
                target_domains.add(group_manager.sequences[seq_id].domain)
        
        # Compute domain overlap
        if not source_domains or not target_domains:
            return 0.5
        
        overlap = len(source_domains & target_domains)
        total = len(source_domains | target_domains)
        
        return overlap / total if total > 0 else 0.0
    
    async def transfer_knowledge(self, 
                               source_group_id: str, 
                               target_group_id: str,
                               group_manager: SequenceGroupManager) -> Dict[str, Any]:
        """Transfer knowledge from source group to target group"""
        
        transfer_potential = await self.compute_transfer_potential(
            source_group_id, target_group_id, group_manager
        )
        
        if transfer_potential < self.transfer_threshold:
            return {"transferred": False, "reason": "Low transfer potential"}
        
        source_group = group_manager.groups[source_group_id]
        target_group = group_manager.groups[target_group_id]
        
        # Transfer optimal parameters with adaptation
        transferred_params = self._adapt_parameters(
            source_group.optimal_parameters,
            target_group,
            transfer_potential
        )
        
        # Update target group
        target_group.optimal_parameters.update(transferred_params)
        
        # Record transfer
        transfer_key = (source_group_id, target_group_id)
        if transfer_key not in self.transfer_history:
            self.transfer_history[transfer_key] = []
        self.transfer_history[transfer_key].append(transfer_potential)
        
        return {
            "transferred": True,
            "transfer_potential": transfer_potential,
            "transferred_parameters": transferred_params
        }
    
    def _adapt_parameters(self, 
                         source_params: Dict[str, Any], 
                         target_group: SequenceGroup,
                         transfer_strength: float) -> Dict[str, Any]:
        """Adapt parameters from source to target group"""
        adapted = {}
        
        for key, value in source_params.items():
            if isinstance(value, (int, float)):
                # Adapt numeric parameters based on transfer strength
                if key in target_group.optimal_parameters:
                    target_value = target_group.optimal_parameters[key]
                    # Weighted combination
                    adapted[key] = (transfer_strength * value + 
                                  (1 - transfer_strength) * target_value)
                else:
                    # Scale by transfer strength for new parameters
                    adapted[key] = value * transfer_strength
            else:
                # Keep non-numeric parameters as is
                adapted[key] = value
        
        return adapted


class GroupSequenceParameterOptimizationEngine:
    """Main GSPO engine that orchestrates sequence grouping and parameter optimization"""
    
    def __init__(self,
                 agent: Agent,
                 environment: Environment,
                 reward_function: RewardFunction,
                 grouping_strategy: SequenceGroupingStrategy = SequenceGroupingStrategy.ADAPTIVE,
                 optimization_levels: List[ParameterOptimizationLevel] = None,
                 num_workers: int = 4):
        
        self.agent = agent
        self.environment = environment
        self.reward_function = reward_function
        self.num_workers = num_workers
        
        if optimization_levels is None:
            optimization_levels = [
                ParameterOptimizationLevel.SEQUENCE,
                ParameterOptimizationLevel.INTRA_GROUP,
                ParameterOptimizationLevel.INTER_GROUP
            ]
        self.optimization_levels = optimization_levels
        
        # Initialize components
        self.group_manager = SequenceGroupManager(grouping_strategy)
        self.parameter_optimizers: Dict[str, ParameterSequenceOptimizer] = {}
        self.cross_group_learner = CrossGroupLearningManager()
        
        # Base GRPO engine for computational efficiency
        self.grpo_engine = ComputationalGRPOEngine(
            agent=agent,
            environment=environment,
            reward_function=reward_function,
            num_workers=num_workers
        )
        
        # Metrics
        self.metrics = {
            "total_sequences": 0,
            "total_groups": 0,
            "successful_transfers": 0,
            "optimization_improvements": 0,
            "average_group_performance": 0.0
        }
    
    async def create_sequence_group(self, 
                                  domain: str, 
                                  context: Dict[str, Any],
                                  characteristics: Dict[str, float]) -> str:
        """Create a new sequence group"""
        sequence_id = f"seq_{len(self.group_manager.sequences)}_{uuid.uuid4().hex[:6]}"
        
        group_id = await self.group_manager.add_sequence(
            sequence_id, domain, context, characteristics
        )
        
        # Initialize parameter optimizer for this sequence
        self.parameter_optimizers[sequence_id] = ParameterSequenceOptimizer()
        
        self.metrics["total_sequences"] += 1
        self.metrics["total_groups"] = len(self.group_manager.groups)
        
        return sequence_id
    
    async def optimize_sequence_parameters(self,
                                         sequence_id: str,
                                         current_params: Dict[str, Any],
                                         performance_feedback: float) -> Dict[str, Any]:
        """Optimize parameters for a specific sequence"""
        
        if sequence_id not in self.group_manager.sequences:
            raise ValueError(f"Sequence {sequence_id} not found")
        
        sequence = self.group_manager.sequences[sequence_id]
        
        # Get group context
        primary_group_id = sequence.group_assignments[0] if sequence.group_assignments else ""
        group_context = self.group_manager.get_group_context(primary_group_id)
        
        optimized_params = current_params.copy()
        
        # Level 1: Sequence-level optimization
        if ParameterOptimizationLevel.SEQUENCE in self.optimization_levels:
            if sequence_id in self.parameter_optimizers:
                optimized_params = self.parameter_optimizers[sequence_id].optimize_parameter_sequence(
                    current_params, performance_feedback, group_context
                )
        
        # Level 2: Intra-group optimization
        if ParameterOptimizationLevel.INTRA_GROUP in self.optimization_levels:
            optimized_params = await self._optimize_intra_group(
                sequence_id, optimized_params, performance_feedback
            )
        
        # Level 3: Inter-group optimization
        if ParameterOptimizationLevel.INTER_GROUP in self.optimization_levels:
            optimized_params = await self._optimize_inter_group(
                sequence_id, optimized_params, performance_feedback
            )
        
        # Update group performance
        await self.group_manager.update_group_performance(
            primary_group_id, sequence_id, performance_feedback, optimized_params
        )
        
        return optimized_params
    
    async def _optimize_intra_group(self,
                                  sequence_id: str,
                                  current_params: Dict[str, Any],
                                  performance: float) -> Dict[str, Any]:
        """Optimize parameters using intra-group knowledge"""
        
        sequence = self.group_manager.sequences[sequence_id]
        if not sequence.group_assignments:
            return current_params
        
        primary_group_id = sequence.group_assignments[0]
        group = self.group_manager.groups[primary_group_id]
        
        # If group has optimal parameters, blend with current
        if group.optimal_parameters and group.performance_stats:
            group_performance = group.performance_stats.get("mean_performance", 0.5)
            
            # If group performance is better, move towards group optimum
            if group_performance > performance:
                blend_factor = 0.3  # Conservative blending
                optimized = current_params.copy()
                
                for key, group_value in group.optimal_parameters.items():
                    if key in current_params and isinstance(current_params[key], (int, float)):
                        current_value = current_params[key]
                        optimized[key] = (1 - blend_factor) * current_value + blend_factor * group_value
                
                return optimized
        
        return current_params
    
    async def _optimize_inter_group(self,
                                  sequence_id: str,
                                  current_params: Dict[str, Any],
                                  performance: float) -> Dict[str, Any]:
        """Optimize parameters using inter-group knowledge transfer"""
        
        sequence = self.group_manager.sequences[sequence_id]
        if not sequence.group_assignments:
            return current_params
        
        primary_group_id = sequence.group_assignments[0]
        
        # Find best performing groups for potential transfer
        best_groups = []
        for group_id, group in self.group_manager.groups.items():
            if (group_id != primary_group_id and 
                group.performance_stats and
                group.performance_stats.get("mean_performance", 0) > performance):
                
                best_groups.append((group_id, group.performance_stats["mean_performance"]))
        
        # Sort by performance
        best_groups.sort(key=lambda x: x[1], reverse=True)
        
        # Try transfer from best performing groups
        for source_group_id, _ in best_groups[:3]:  # Top 3 groups
            transfer_result = await self.cross_group_learner.transfer_knowledge(
                source_group_id, primary_group_id, self.group_manager
            )
            
            if transfer_result["transferred"]:
                self.metrics["successful_transfers"] += 1
                
                # Apply transferred parameters
                transferred_params = transfer_result["transferred_parameters"]
                optimized = current_params.copy()
                
                blend_factor = 0.2  # Conservative inter-group blending
                for key, transfer_value in transferred_params.items():
                    if key in current_params and isinstance(current_params[key], (int, float)):
                        current_value = current_params[key]
                        optimized[key] = (1 - blend_factor) * current_value + blend_factor * transfer_value
                
                return optimized
        
        return current_params
    
    async def train_iteration_gspo(self, 
                                 prompts: List[str],
                                 sequence_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a GSPO training iteration combining GRPO with group sequence optimization"""
        
        if len(prompts) != len(sequence_contexts):
            raise ValueError("Number of prompts must match number of sequence contexts")
        
        # Generate trajectories using base GRPO engine
        base_trajectories = await self.grpo_engine.generate_trajectory_batch(prompts)
        
        # Enhance trajectories with GSPO information
        gspo_trajectories = []
        for i, traj in enumerate(base_trajectories):
            context = sequence_contexts[i]
            
            # Create or get sequence
            sequence_id = context.get("sequence_id")
            if not sequence_id:
                # Create new sequence
                sequence_id = await self.create_sequence_group(
                    domain=context.get("domain", "default"),
                    context=context,
                    characteristics=context.get("characteristics", {})
                )
            
            # Create GSPO trajectory
            gspo_traj = GSPOTrajectory(
                trajectory_id=traj.trajectory_id,
                prompt=traj.prompt,
                response=traj.response,
                raw_reward=traj.raw_reward,
                learned_reward=traj.learned_reward,
                computational_cost=traj.computational_cost,
                metadata=traj.metadata,
                sequence_id=sequence_id,
                group_id=self.group_manager.sequences[sequence_id].group_assignments[0] if sequence_id in self.group_manager.sequences else "",
                parameter_sequence=[context.get("current_params", {})],
                sequence_position=context.get("position", 0),
                group_context=self.group_manager.get_group_context(
                    self.group_manager.sequences[sequence_id].group_assignments[0] if sequence_id in self.group_manager.sequences else ""
                )
            )
            
            gspo_trajectories.append(gspo_traj)
        
        # Optimize parameters for each sequence
        optimization_results = []
        for traj in gspo_trajectories:
            if traj.sequence_id and traj.parameter_sequence:
                current_params = traj.parameter_sequence[0]
                optimized_params = await self.optimize_sequence_parameters(
                    traj.sequence_id, current_params, traj.learned_reward
                )
                optimization_results.append({
                    "sequence_id": traj.sequence_id,
                    "original_params": current_params,
                    "optimized_params": optimized_params,
                    "performance": traj.learned_reward
                })
        
        # Update global metrics
        self._update_global_metrics(gspo_trajectories, optimization_results)
        
        return {
            "trajectories": gspo_trajectories,
            "optimization_results": optimization_results,
            "metrics": self.metrics.copy(),
            "group_summary": self._get_group_summary()
        }
    
    def _update_global_metrics(self, 
                             trajectories: List[GSPOTrajectory], 
                             optimization_results: List[Dict[str, Any]]):
        """Update global GSPO metrics"""
        
        if trajectories:
            avg_performance = np.mean([t.learned_reward for t in trajectories])
            self.metrics["average_group_performance"] = avg_performance
        
        # Count optimization improvements
        improvements = 0
        for result in optimization_results:
            original = result["original_params"]
            optimized = result["optimized_params"]
            if original != optimized:
                improvements += 1
        
        self.metrics["optimization_improvements"] += improvements
    
    def _get_group_summary(self) -> Dict[str, Any]:
        """Get summary of all groups"""
        summary = {
            "total_groups": len(self.group_manager.groups),
            "groups": []
        }
        
        for group_id, group in self.group_manager.groups.items():
            group_summary = {
                "group_id": group_id,
                "name": group.name,
                "size": len(group.sequences),
                "performance_stats": group.performance_stats,
                "characteristics": group.characteristics
            }
            summary["groups"].append(group_summary)
        
        return summary
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about the optimization process"""
        
        insights = {
            "total_sequences_processed": self.metrics["total_sequences"],
            "groups_created": self.metrics["total_groups"],
            "successful_knowledge_transfers": self.metrics["successful_transfers"],
            "parameter_optimizations": self.metrics["optimization_improvements"],
            "average_performance": self.metrics["average_group_performance"],
            "group_efficiency": self.metrics["total_sequences"] / max(1, self.metrics["total_groups"]),
            "transfer_success_rate": (self.metrics["successful_transfers"] / 
                                    max(1, self.metrics["optimization_improvements"]))
        }
        
        # Add group-specific insights
        insights["group_insights"] = []
        for group_id, group in self.group_manager.groups.items():
            if group.performance_stats:
                group_insight = {
                    "group_id": group_id,
                    "name": group.name,
                    "sequences": len(group.sequences),
                    "avg_performance": group.performance_stats.get("mean_performance", 0),
                    "performance_trend": group.performance_stats.get("recent_trend", 0),
                    "stability": 1.0 / (group.performance_stats.get("std_performance", 1.0) + 1e-6)
                }
                insights["group_insights"].append(group_insight)
        
        return insights
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.grpo_engine.cleanup()


# Factory functions
def create_gspo_engine(agent: Agent,
                      environment: Environment,
                      reward_function: RewardFunction,
                      grouping_strategy: str = "adaptive",
                      optimization_levels: List[str] = None,
                      num_workers: int = 4) -> GroupSequenceParameterOptimizationEngine:
    """Factory function to create GSPO engine"""
    
    strategy_map = {
        "similarity": SequenceGroupingStrategy.SIMILARITY,
        "performance": SequenceGroupingStrategy.PERFORMANCE,
        "domain": SequenceGroupingStrategy.DOMAIN,
        "temporal": SequenceGroupingStrategy.TEMPORAL,
        "adaptive": SequenceGroupingStrategy.ADAPTIVE,
        "hierarchical": SequenceGroupingStrategy.HIERARCHICAL
    }
    
    level_map = {
        "sequence": ParameterOptimizationLevel.SEQUENCE,
        "intra_group": ParameterOptimizationLevel.INTRA_GROUP,
        "inter_group": ParameterOptimizationLevel.INTER_GROUP,
        "global": ParameterOptimizationLevel.GLOBAL
    }
    
    strategy = strategy_map.get(grouping_strategy, SequenceGroupingStrategy.ADAPTIVE)
    
    levels = None
    if optimization_levels:
        levels = [level_map.get(level, ParameterOptimizationLevel.SEQUENCE) 
                 for level in optimization_levels]
    
    return GroupSequenceParameterOptimizationEngine(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        grouping_strategy=strategy,
        optimization_levels=levels,
        num_workers=num_workers
    )
