"""
Tests for Multi-Agent Coordination System
"""

import pytest
import asyncio

from core.multi_agent_coordination import (
    AgentRole,
    AgentMessage,
    AgentState,
    TeamState,
    BlackboardChannel,
    CapabilityBasedAllocator,
    PerformanceBasedAllocator,
    MultiAgentCoordinator,
    CooperativeRewardShaping,
    CoordinationStrategy,
    CommunicationProtocol,
)
from stateset_agents.core.agent import Agent


class MockAgent(Agent):
    """Mock agent for testing"""

    def __init__(self, agent_id: str, response: str = "test response"):
        self.agent_id = agent_id
        self.response = response
        self.capabilities = []

    async def generate_response(self, prompt: str, **kwargs) -> str:
        return self.response


class TestAgentMessage:
    """Test agent message structure"""

    def test_message_creation(self):
        msg = AgentMessage(
            sender_id="agent1",
            receiver_id="agent2",
            content="Hello",
            message_type="info",
        )

        assert msg.sender_id == "agent1"
        assert msg.receiver_id == "agent2"
        assert msg.content == "Hello"
        assert not msg.is_broadcast()
        assert msg.is_for("agent2")

    def test_broadcast_message(self):
        msg = AgentMessage(
            sender_id="agent1",
            receiver_id=None,  # Broadcast
            content="Announcement",
        )

        assert msg.is_broadcast()
        assert msg.is_for("agent2")  # Broadcast is for everyone
        assert msg.is_for("agent3")


class TestAgentState:
    """Test agent state management"""

    def test_state_creation(self):
        state = AgentState(
            agent_id="agent1",
            role=AgentRole.COORDINATOR,
            capabilities={"planning", "coordination"},
        )

        assert state.agent_id == "agent1"
        assert state.role == AgentRole.COORDINATOR
        assert "planning" in state.capabilities
        assert state.status == "idle"
        assert state.workload == 0.0


class TestBlackboardChannel:
    """Test blackboard communication channel"""

    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        channel = BlackboardChannel()

        msg = AgentMessage(
            sender_id="agent1",
            receiver_id="agent2",
            content="Test message",
        )

        await channel.send(msg)
        received = await channel.receive("agent2", timeout=1.0)

        assert received is not None
        assert received.content == "Test message"
        assert received.sender_id == "agent1"

    @pytest.mark.asyncio
    async def test_broadcast(self):
        channel = BlackboardChannel()

        # Register agents
        channel._ensure_queue("agent1")
        channel._ensure_queue("agent2")
        channel._ensure_queue("agent3")

        msg = AgentMessage(
            sender_id="coordinator",
            receiver_id=None,  # Broadcast
            content="Broadcast message",
        )

        await channel.broadcast(msg)

        # All agents should receive
        msg1 = await channel.receive("agent1", timeout=1.0)
        msg2 = await channel.receive("agent2", timeout=1.0)
        msg3 = await channel.receive("agent3", timeout=1.0)

        assert msg1.content == "Broadcast message"
        assert msg2.content == "Broadcast message"
        assert msg3.content == "Broadcast message"

    @pytest.mark.asyncio
    async def test_shared_memory(self):
        channel = BlackboardChannel()

        await channel.write_shared("key1", "value1")
        await channel.write_shared("key2", 42)

        assert await channel.read_shared("key1") == "value1"
        assert await channel.read_shared("key2") == 42
        assert await channel.read_shared("nonexistent") is None

        all_shared = channel.get_all_shared()
        assert "key1" in all_shared
        assert "key2" in all_shared

    @pytest.mark.asyncio
    async def test_receive_timeout(self):
        channel = BlackboardChannel()

        # Should timeout and return None
        result = await channel.receive("agent1", timeout=0.1)
        assert result is None


class TestCapabilityBasedAllocator:
    """Test capability-based task allocation"""

    @pytest.mark.asyncio
    async def test_allocate_matching_capability(self):
        allocator = CapabilityBasedAllocator()

        agents = [
            AgentState(
                agent_id="agent1",
                role=AgentRole.SPECIALIST,
                capabilities={"python", "ml"},
                workload=0.3,
            ),
            AgentState(
                agent_id="agent2",
                role=AgentRole.SPECIALIST,
                capabilities={"javascript", "web"},
                workload=0.5,
            ),
        ]

        task = {
            "task_id": "task1",
            "required_capabilities": ["python", "ml"],
        }

        team_state = TeamState(team_id="team1", agents={})

        agent_id = await allocator.allocate_task(task, agents, team_state)

        assert agent_id == "agent1"  # Only agent1 has required capabilities

    @pytest.mark.asyncio
    async def test_allocate_workload_balancing(self):
        allocator = CapabilityBasedAllocator(consider_workload=True)

        agents = [
            AgentState(
                agent_id="agent1",
                role=AgentRole.EXECUTOR,
                capabilities={"task"},
                workload=0.8,
            ),
            AgentState(
                agent_id="agent2",
                role=AgentRole.EXECUTOR,
                capabilities={"task"},
                workload=0.2,
            ),
        ]

        task = {"task_id": "task1", "required_capabilities": ["task"]}
        team_state = TeamState(team_id="team1", agents={})

        agent_id = await allocator.allocate_task(task, agents, team_state)

        # Should choose agent2 with lower workload
        assert agent_id == "agent2"

    @pytest.mark.asyncio
    async def test_no_capable_agent(self):
        allocator = CapabilityBasedAllocator()

        agents = [
            AgentState(
                agent_id="agent1",
                role=AgentRole.SPECIALIST,
                capabilities={"python"},
            ),
        ]

        task = {"task_id": "task1", "required_capabilities": ["rust"]}
        team_state = TeamState(team_id="team1", agents={})

        agent_id = await allocator.allocate_task(task, agents, team_state)

        assert agent_id is None  # No capable agent


class TestPerformanceBasedAllocator:
    """Test performance-based task allocation"""

    @pytest.mark.asyncio
    async def test_allocate_best_performer(self):
        allocator = PerformanceBasedAllocator(exploration_rate=0.0)  # No exploration

        agents = [
            AgentState(
                agent_id="agent1",
                role=AgentRole.EXECUTOR,
                performance_history=[0.5, 0.6, 0.7],
            ),
            AgentState(
                agent_id="agent2",
                role=AgentRole.EXECUTOR,
                performance_history=[0.8, 0.85, 0.9],
            ),
        ]

        task = {"task_id": "task1"}
        team_state = TeamState(team_id="team1", agents={})

        agent_id = await allocator.allocate_task(task, agents, team_state)

        # Should choose agent2 with better performance
        assert agent_id == "agent2"

    @pytest.mark.asyncio
    async def test_no_history_fallback(self):
        allocator = PerformanceBasedAllocator(exploration_rate=0.0)

        agents = [
            AgentState(agent_id="agent1", role=AgentRole.EXECUTOR),
            AgentState(agent_id="agent2", role=AgentRole.EXECUTOR),
        ]

        task = {"task_id": "task1"}
        team_state = TeamState(team_id="team1", agents={})

        agent_id = await allocator.allocate_task(task, agents, team_state)

        # Should randomly choose one
        assert agent_id in ["agent1", "agent2"]


class TestMultiAgentCoordinator:
    """Test multi-agent coordination"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        agents = {
            "agent1": MockAgent("agent1", "Response 1"),
            "agent2": MockAgent("agent2", "Response 2"),
        }

        roles = {
            "agent1": AgentRole.COORDINATOR,
            "agent2": AgentRole.EXECUTOR,
        }

        coordinator = MultiAgentCoordinator(
            agents=agents,
            roles=roles,
            coordination_strategy=CoordinationStrategy.COOPERATIVE,
        )

        assert len(coordinator.agents) == 2
        assert coordinator.agent_states["agent1"].role == AgentRole.COORDINATOR
        assert coordinator.agent_states["agent2"].role == AgentRole.EXECUTOR

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        agents = {
            "agent1": MockAgent("agent1", "Response from agent1"),
            "agent2": MockAgent("agent2", "Response from agent2"),
        }

        coordinator = MultiAgentCoordinator(
            agents=agents,
            coordination_strategy=CoordinationStrategy.PARALLEL,
        )

        task = {
            "task_id": "test_task",
            "description": "Test parallel execution",
            "context": "Test context",
        }

        trajectory, result = await coordinator.execute_collaborative_task(task)

        assert len(trajectory.turns) == 2
        assert "results" in result
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        agents = {
            "agent1": MockAgent("agent1", "First response"),
            "agent2": MockAgent("agent2", "Second response"),
        }

        # Add capabilities
        for agent in agents.values():
            agent.capabilities = ["general"]

        coordinator = MultiAgentCoordinator(
            agents=agents,
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
        )

        # Create agent states with capabilities
        for agent_id in agents:
            coordinator.agent_states[agent_id].capabilities.add("general")

        task = {
            "task_id": "test_task",
            "description": "Test sequential execution",
            "required_capabilities": ["general"],
        }

        trajectory, result = await coordinator.execute_collaborative_task(task, max_iterations=2)

        assert len(trajectory.turns) >= 1
        assert "results" in result

    @pytest.mark.asyncio
    async def test_send_message(self):
        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }

        coordinator = MultiAgentCoordinator(agents=agents)

        msg = AgentMessage(
            sender_id="agent1",
            receiver_id="agent2",
            content="Test message",
        )

        await coordinator.send_message(msg)

        # Check message history
        assert len(coordinator.team_state.message_history) == 1
        assert coordinator.team_state.message_history[0].content == "Test message"

    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }

        coordinator = MultiAgentCoordinator(agents=agents)

        await coordinator.broadcast_message("coordinator", "Broadcast test")

        assert len(coordinator.team_state.message_history) == 1
        assert coordinator.team_state.message_history[0].is_broadcast()

    def test_update_agent_workload(self):
        agents = {"agent1": MockAgent("agent1")}
        coordinator = MultiAgentCoordinator(agents=agents)

        coordinator.update_agent_workload("agent1", 0.75)
        assert coordinator.agent_states["agent1"].workload == 0.75

        # Test clamping
        coordinator.update_agent_workload("agent1", 1.5)
        assert coordinator.agent_states["agent1"].workload == 1.0

    def test_record_agent_performance(self):
        agents = {"agent1": MockAgent("agent1")}
        coordinator = MultiAgentCoordinator(agents=agents)

        coordinator.record_agent_performance("agent1", 0.8)
        coordinator.record_agent_performance("agent1", 0.9)

        assert len(coordinator.agent_states["agent1"].performance_history) == 2
        assert coordinator.agent_states["agent1"].performance_history[0] == 0.8

    def test_get_team_statistics(self):
        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }
        coordinator = MultiAgentCoordinator(agents=agents)

        coordinator.update_agent_workload("agent1", 0.5)
        coordinator.record_agent_performance("agent1", 0.8)

        stats = coordinator.get_team_statistics()

        assert stats["num_agents"] == 2
        assert "agent_workloads" in stats
        assert "agent_performance" in stats
        assert stats["agent_workloads"]["agent1"] == 0.5


class TestCooperativeRewardShaping:
    """Test cooperative reward shaping"""

    def test_compute_agent_rewards(self):
        reward_shaper = CooperativeRewardShaping(
            team_reward_weight=0.5,
            individual_reward_weight=0.3,
            cooperation_bonus_weight=0.2,
        )

        team_reward = 1.0
        individual_contributions = {
            "agent1": 0.8,
            "agent2": 0.6,
        }
        cooperation_metrics = {
            "agent1": 0.9,
            "agent2": 0.7,
        }

        rewards = reward_shaper.compute_agent_rewards(
            team_reward,
            individual_contributions,
            cooperation_metrics,
        )

        assert "agent1" in rewards
        assert "agent2" in rewards

        # Agent1 should have higher reward (better individual + cooperation)
        assert rewards["agent1"] > rewards["agent2"]

        # Both should benefit from team success
        assert rewards["agent1"] > 0
        assert rewards["agent2"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
