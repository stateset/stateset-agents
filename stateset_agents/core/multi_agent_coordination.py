"""
Multi-Agent Coordination System

This module provides primitives for coordinating multiple agents in team-based
scenarios, including communication protocols, role assignment, and collaborative
task execution.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .agent import Agent
from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Predefined agent roles in multi-agent systems"""

    COORDINATOR = "coordinator"  # Orchestrates team activities
    SPECIALIST = "specialist"  # Domain expert
    EVALUATOR = "evaluator"  # Quality assurance
    RESEARCHER = "researcher"  # Information gathering
    EXECUTOR = "executor"  # Task execution
    CUSTOM = "custom"  # User-defined role


class CommunicationProtocol(Enum):
    """Communication patterns between agents"""

    BROADCAST = "broadcast"  # One-to-all communication
    PEER_TO_PEER = "peer_to_peer"  # Direct agent-to-agent
    HIERARCHICAL = "hierarchical"  # Tree-structured communication
    BLACKBOARD = "blackboard"  # Shared memory space
    MESSAGE_QUEUE = "message_queue"  # Asynchronous messaging


class CoordinationStrategy(Enum):
    """Strategy for coordinating agent actions"""

    SEQUENTIAL = "sequential"  # Agents act in sequence
    PARALLEL = "parallel"  # Agents act simultaneously
    CONSENSUS = "consensus"  # Agents must agree
    COMPETITIVE = "competitive"  # Agents compete
    COOPERATIVE = "cooperative"  # Agents collaborate


@dataclass
class AgentMessage:
    """Message passed between agents"""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: Optional[str] = None  # None for broadcast
    content: str = ""
    message_type: str = "info"  # info, request, response, command
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def is_broadcast(self) -> bool:
        return self.receiver_id is None

    def is_for(self, agent_id: str) -> bool:
        return self.receiver_id is None or self.receiver_id == agent_id


@dataclass
class AgentState:
    """State of an agent in multi-agent system"""

    agent_id: str
    role: AgentRole
    status: str = "idle"  # idle, active, waiting, completed
    current_task: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    workload: float = 0.0  # 0.0 to 1.0
    performance_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamState:
    """State of the entire agent team"""

    team_id: str
    agents: Dict[str, AgentState]
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    message_history: List[AgentMessage] = field(default_factory=list)
    task_queue: List[Dict[str, Any]] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    team_reward: float = 0.0


class CommunicationChannel(ABC):
    """Abstract base for agent communication"""

    @abstractmethod
    async def send(self, message: AgentMessage) -> None:
        """Send a message"""
        pass

    @abstractmethod
    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive a message for specific agent"""
        pass

    @abstractmethod
    async def broadcast(self, message: AgentMessage) -> None:
        """Broadcast message to all agents"""
        pass


class BlackboardChannel(CommunicationChannel):
    """Blackboard-style shared memory communication"""

    def __init__(self):
        self.blackboard: Dict[str, Any] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    def _ensure_queue(self, agent_id: str) -> None:
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = asyncio.Queue()

    async def send(self, message: AgentMessage) -> None:
        """Send message to specific agent or broadcast"""
        if message.is_broadcast():
            await self.broadcast(message)
        else:
            self._ensure_queue(message.receiver_id)
            await self.message_queues[message.receiver_id].put(message)

    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive message for agent"""
        self._ensure_queue(agent_id)
        try:
            return await asyncio.wait_for(self.message_queues[agent_id].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def broadcast(self, message: AgentMessage) -> None:
        """Broadcast to all agents"""
        async with self.lock:
            for queue in self.message_queues.values():
                await queue.put(message)

    async def write_shared(self, key: str, value: Any) -> None:
        """Write to shared blackboard"""
        async with self.lock:
            self.blackboard[key] = value

    async def read_shared(self, key: str) -> Optional[Any]:
        """Read from shared blackboard"""
        async with self.lock:
            return self.blackboard.get(key)

    def get_all_shared(self) -> Dict[str, Any]:
        """Get all shared memory"""
        return self.blackboard.copy()


class TaskAllocator(ABC):
    """Abstract base for task allocation strategies"""

    @abstractmethod
    async def allocate_task(
        self,
        task: Dict[str, Any],
        available_agents: List[AgentState],
        team_state: TeamState,
    ) -> Optional[str]:
        """Allocate task to an agent, return agent_id"""
        pass


class CapabilityBasedAllocator(TaskAllocator):
    """Allocate tasks based on agent capabilities"""

    def __init__(self, consider_workload: bool = True):
        self.consider_workload = consider_workload

    async def allocate_task(
        self,
        task: Dict[str, Any],
        available_agents: List[AgentState],
        team_state: TeamState,
    ) -> Optional[str]:
        """Allocate to agent with matching capabilities and lowest workload"""
        required_capabilities = set(task.get("required_capabilities", []))

        # Filter agents with matching capabilities
        capable_agents = [
            agent for agent in available_agents if required_capabilities.issubset(agent.capabilities)
        ]

        if not capable_agents:
            logger.warning(f"No capable agents for task: {task.get('task_id')}")
            return None

        # Select agent with lowest workload
        if self.consider_workload:
            selected = min(capable_agents, key=lambda a: a.workload)
        else:
            # Random selection among capable agents
            selected = np.random.choice(capable_agents)

        return selected.agent_id


class PerformanceBasedAllocator(TaskAllocator):
    """Allocate tasks based on historical performance"""

    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate

    async def allocate_task(
        self,
        task: Dict[str, Any],
        available_agents: List[AgentState],
        team_state: TeamState,
    ) -> Optional[str]:
        """Allocate to highest performing agent with exploration"""
        if not available_agents:
            return None

        # Exploration: random agent
        if np.random.random() < self.exploration_rate:
            return np.random.choice([a.agent_id for a in available_agents])

        # Exploitation: best performing agent
        agents_with_history = [a for a in available_agents if a.performance_history]

        if not agents_with_history:
            return np.random.choice([a.agent_id for a in available_agents])

        best_agent = max(agents_with_history, key=lambda a: np.mean(a.performance_history[-10:]))
        return best_agent.agent_id


class MultiAgentCoordinator:
    """
    Coordinates multiple agents working on collaborative tasks.

    Manages communication, task allocation, and team dynamics.
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        roles: Optional[Dict[str, AgentRole]] = None,
        communication_protocol: CommunicationProtocol = CommunicationProtocol.BLACKBOARD,
        coordination_strategy: CoordinationStrategy = CoordinationStrategy.COOPERATIVE,
        task_allocator: Optional[TaskAllocator] = None,
    ):
        """
        Initialize multi-agent coordinator.

        Args:
            agents: Dictionary of agent_id -> Agent
            roles: Optional role assignments for agents
            communication_protocol: How agents communicate
            coordination_strategy: How agents coordinate actions
            task_allocator: Strategy for assigning tasks
        """
        self.agents = agents
        self.coordination_strategy = coordination_strategy
        self.task_allocator = task_allocator or CapabilityBasedAllocator()

        # Initialize agent states
        self.agent_states: Dict[str, AgentState] = {}
        for agent_id, agent in agents.items():
            role = roles.get(agent_id, AgentRole.EXECUTOR) if roles else AgentRole.EXECUTOR
            self.agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                role=role,
                capabilities=set(getattr(agent, "capabilities", [])),
            )

        # Initialize communication channel
        if communication_protocol == CommunicationProtocol.BLACKBOARD:
            self.comm_channel: CommunicationChannel = BlackboardChannel()
        else:
            raise ValueError(
                f"Unsupported communication_protocol={communication_protocol}. "
                f"Only {CommunicationProtocol.BLACKBOARD.value!r} is currently implemented."
            )

        # Initialize team state
        self.team_state = TeamState(team_id=str(uuid.uuid4()), agents=self.agent_states)

        self.active_tasks: Dict[str, str] = {}  # task_id -> agent_id

    async def execute_collaborative_task(
        self,
        task: Dict[str, Any],
        max_iterations: int = 10,
    ) -> Tuple[MultiTurnTrajectory, Dict[str, Any]]:
        """
        Execute a task requiring multiple agents.

        Args:
            task: Task specification with required capabilities
            max_iterations: Maximum interaction iterations

        Returns:
            Trajectory of agent interactions and final results
        """
        task_id = task.get("task_id", str(uuid.uuid4()))
        trajectory = MultiTurnTrajectory(trajectory_id=task_id, turns=[], total_reward=0.0)

        logger.info(f"Starting collaborative task: {task_id}")

        if self.coordination_strategy == CoordinationStrategy.SEQUENTIAL:
            result = await self._execute_sequential(task, trajectory, max_iterations)
        elif self.coordination_strategy == CoordinationStrategy.PARALLEL:
            result = await self._execute_parallel(task, trajectory, max_iterations)
        elif self.coordination_strategy == CoordinationStrategy.CONSENSUS:
            result = await self._execute_consensus(task, trajectory, max_iterations)
        else:
            supported = [
                CoordinationStrategy.SEQUENTIAL.value,
                CoordinationStrategy.PARALLEL.value,
                CoordinationStrategy.CONSENSUS.value,
            ]
            raise ValueError(
                f"Unsupported coordination_strategy={self.coordination_strategy}. "
                f"Supported strategies: {supported}."
            )

        self.team_state.completed_tasks.append(task_id)
        return trajectory, result

    async def _execute_sequential(
        self,
        task: Dict[str, Any],
        trajectory: MultiTurnTrajectory,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Execute task with agents acting sequentially"""
        context = task.get("context", "")
        results = []

        for iteration in range(max_iterations):
            # Allocate subtask to an agent
            available_agents = [a for a in self.agent_states.values() if a.status == "idle"]

            if not available_agents:
                break

            agent_id = await self.task_allocator.allocate_task(task, available_agents, self.team_state)

            if not agent_id:
                logger.warning(f"No agent available for task iteration {iteration}")
                break

            # Execute agent action
            agent = self.agents[agent_id]
            agent_state = self.agent_states[agent_id]
            agent_state.status = "active"

            # Get agent response
            prompt = f"Task: {task.get('description', '')}\nContext: {context}\nYour role: {agent_state.role.value}"

            try:
                response = await agent.generate_response(prompt)

                turn = ConversationTurn(
                    role=f"agent_{agent_id}",
                    content=response,
                    metadata={
                        "agent_id": agent_id,
                        "agent_role": agent_state.role.value,
                        "iteration": iteration,
                    },
                )
                trajectory.add_turn(turn, reward=0.0)

                results.append({"agent_id": agent_id, "response": response, "iteration": iteration})

                # Update context with agent response
                context += f"\n{agent_state.role.value}: {response}"

                # Check if task is complete
                if "complete" in response.lower() or "done" in response.lower():
                    break

            finally:
                agent_state.status = "idle"

        return {"results": results, "final_context": context}

    async def _execute_parallel(
        self,
        task: Dict[str, Any],
        trajectory: MultiTurnTrajectory,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Execute task with agents acting in parallel"""
        prompt = f"Task: {task.get('description', '')}\nContext: {task.get('context', '')}"

        # Execute all agents in parallel
        agent_tasks = []
        agent_ids = []

        for agent_id, agent in self.agents.items():
            agent_state = self.agent_states[agent_id]
            full_prompt = f"{prompt}\nYour role: {agent_state.role.value}"
            agent_tasks.append(agent.generate_response(full_prompt))
            agent_ids.append(agent_id)

        # Wait for all agents
        responses = await asyncio.gather(*agent_tasks, return_exceptions=True)

        results = []
        for agent_id, response in zip(agent_ids, responses):
            if isinstance(response, Exception):
                logger.error(f"Agent {agent_id} failed: {response}")
                continue

            turn = ConversationTurn(
                role=f"agent_{agent_id}",
                content=str(response),
                metadata={
                    "agent_id": agent_id,
                    "agent_role": self.agent_states[agent_id].role.value,
                },
            )
            trajectory.add_turn(turn, reward=0.0)
            results.append({"agent_id": agent_id, "response": response})

        # Aggregate results
        aggregated = await self._aggregate_responses([r["response"] for r in results])

        return {"results": results, "aggregated_result": aggregated}

    async def _execute_consensus(
        self,
        task: Dict[str, Any],
        trajectory: MultiTurnTrajectory,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Execute task requiring agent consensus"""
        prompt = f"Task: {task.get('description', '')}\nContext: {task.get('context', '')}"

        consensus_reached = False
        iteration = 0
        agent_proposals = {}

        while not consensus_reached and iteration < max_iterations:
            # Each agent proposes a solution
            proposals = []

            for agent_id, agent in self.agents.items():
                agent_state = self.agent_states[agent_id]
                full_prompt = f"{prompt}\n\nPrevious proposals: {agent_proposals}\n\nProvide your proposal:"

                response = await agent.generate_response(full_prompt)
                proposals.append((agent_id, response))
                agent_proposals[agent_id] = response

                turn = ConversationTurn(
                    role=f"agent_{agent_id}",
                    content=response,
                    metadata={
                        "agent_id": agent_id,
                        "iteration": iteration,
                        "proposal": True,
                    },
                )
                trajectory.add_turn(turn, reward=0.0)

            # Check for consensus (simplified: majority agreement)
            consensus_reached = await self._check_consensus(proposals)
            iteration += 1

        final_solution = await self._aggregate_responses([p[1] for p in proposals])

        return {
            "consensus_reached": consensus_reached,
            "iterations": iteration,
            "final_solution": final_solution,
            "proposals": agent_proposals,
        }

    async def _aggregate_responses(self, responses: List[str]) -> str:
        """Aggregate multiple agent responses"""
        # Simple aggregation - can be made more sophisticated
        if len(responses) == 1:
            return responses[0]

        # For now, return the longest response (most detailed)
        return max(responses, key=len)

    async def _check_consensus(self, proposals: List[Tuple[str, str]]) -> bool:
        """Check if agents have reached consensus"""
        if len(proposals) < 2:
            return True

        # Simplified consensus: check if responses are similar
        # In practice, would use semantic similarity
        responses = [p[1] for p in proposals]
        avg_length = np.mean([len(r) for r in responses])
        length_variance = np.var([len(r) for r in responses])

        # Low variance in response length as proxy for consensus
        return length_variance < (avg_length * 0.2) ** 2

    async def send_message(self, message: AgentMessage) -> None:
        """Send message through communication channel"""
        await self.comm_channel.send(message)
        self.team_state.message_history.append(message)

    async def broadcast_message(self, sender_id: str, content: str, message_type: str = "info") -> None:
        """Broadcast message to all agents"""
        message = AgentMessage(sender_id=sender_id, content=content, message_type=message_type)
        await self.comm_channel.broadcast(message)
        self.team_state.message_history.append(message)

    def update_agent_workload(self, agent_id: str, workload: float) -> None:
        """Update agent workload"""
        if agent_id in self.agent_states:
            self.agent_states[agent_id].workload = np.clip(workload, 0.0, 1.0)

    def record_agent_performance(self, agent_id: str, performance: float) -> None:
        """Record agent performance for task allocation"""
        if agent_id in self.agent_states:
            self.agent_states[agent_id].performance_history.append(performance)
            # Keep only recent history
            if len(self.agent_states[agent_id].performance_history) > 100:
                self.agent_states[agent_id].performance_history.pop(0)

    def get_team_statistics(self) -> Dict[str, Any]:
        """Get team performance statistics"""
        return {
            "team_id": self.team_state.team_id,
            "num_agents": len(self.agents),
            "completed_tasks": len(self.team_state.completed_tasks),
            "total_messages": len(self.team_state.message_history),
            "team_reward": self.team_state.team_reward,
            "agent_workloads": {agent_id: state.workload for agent_id, state in self.agent_states.items()},
            "agent_performance": {
                agent_id: np.mean(state.performance_history[-10:]) if state.performance_history else 0.0
                for agent_id, state in self.agent_states.items()
            },
        }


class CooperativeRewardShaping:
    """
    Reward shaping for multi-agent systems.

    Encourages cooperation and team success over individual performance.
    """

    def __init__(
        self,
        team_reward_weight: float = 0.5,
        individual_reward_weight: float = 0.3,
        cooperation_bonus_weight: float = 0.2,
    ):
        self.team_reward_weight = team_reward_weight
        self.individual_reward_weight = individual_reward_weight
        self.cooperation_bonus_weight = cooperation_bonus_weight

    def compute_agent_rewards(
        self,
        team_reward: float,
        individual_contributions: Dict[str, float],
        cooperation_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute rewards for each agent considering team success.

        Args:
            team_reward: Overall team performance
            individual_contributions: Individual agent performance
            cooperation_metrics: Measures of cooperation (message sharing, etc.)

        Returns:
            Dictionary of agent_id -> reward
        """
        agent_rewards = {}

        for agent_id in individual_contributions.keys():
            individual = individual_contributions.get(agent_id, 0.0)
            cooperation = cooperation_metrics.get(agent_id, 0.0)

            # Weighted combination
            total_reward = (
                self.team_reward_weight * team_reward
                + self.individual_reward_weight * individual
                + self.cooperation_bonus_weight * cooperation
            )

            agent_rewards[agent_id] = total_reward

        return agent_rewards
