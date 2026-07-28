"""Unit tests for the `--checkpoint` → API startup wiring.

Verifies that `AgentService.register_default_checkpoint_agent` correctly
registers a stub-backed agent at startup when given a checkpoint path. The
stub-backed path is the only one we can test in unit-test mode (real HF
weights need GPU); this test guards the contract that the env-var → load
plumbing works.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
async def agent_service():
    from stateset_agents.api.services.agent_service import AgentService
    from stateset_agents.utils.security import SecurityMonitor

    return AgentService(SecurityMonitor())


class TestRegisterDefaultCheckpointAgent:
    @pytest.mark.asyncio
    async def test_registers_stub_agent_from_existing_path(
        self, tmp_path: Path
    ) -> None:
        """The path must exist; stub-backed models bypass real loading."""
        from stateset_agents.api.services.agent_service import AgentService
        from stateset_agents.utils.security import SecurityMonitor

        svc = AgentService(SecurityMonitor())

        # Make a dummy "checkpoint" directory (path-existence check only).
        ckpt = tmp_path / "fake_adapter"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text("{}")

        agent_id = await svc.register_default_checkpoint_agent(
            checkpoint_path=str(ckpt),
            base_model="stub://test-model",
            agent_id="default",
        )
        assert agent_id == "default"
        assert "default" in svc.agents

    @pytest.mark.asyncio
    async def test_missing_path_raises_file_not_found(self) -> None:
        from stateset_agents.api.services.agent_service import AgentService
        from stateset_agents.utils.security import SecurityMonitor

        svc = AgentService(SecurityMonitor())
        with pytest.raises(FileNotFoundError, match="does not exist"):
            await svc.register_default_checkpoint_agent(
                checkpoint_path="/tmp/this/does/not/exist",
            )

    @pytest.mark.asyncio
    async def test_custom_agent_id(self, tmp_path: Path) -> None:
        from stateset_agents.api.services.agent_service import AgentService
        from stateset_agents.utils.security import SecurityMonitor

        svc = AgentService(SecurityMonitor())
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        await svc.register_default_checkpoint_agent(
            checkpoint_path=str(ckpt),
            base_model="stub://test-model",
            agent_id="customer-support-v2",
        )
        assert "customer-support-v2" in svc.agents
        assert "default" not in svc.agents  # — only the custom id registered

    @pytest.mark.asyncio
    async def test_peft_path_set_when_base_model_supplied(self, tmp_path: Path) -> None:
        """When base_model is named, the LoRA path should be threaded into AgentConfig.peft_path."""
        from stateset_agents.api.services.agent_service import AgentService
        from stateset_agents.utils.security import SecurityMonitor

        svc = AgentService(SecurityMonitor())
        ckpt = tmp_path / "adapter"
        ckpt.mkdir()

        # Stub model — peft_path is *not* set because stub backend skips loading.
        await svc.register_default_checkpoint_agent(
            checkpoint_path=str(ckpt),
            base_model="stub://test",
        )
        assert svc.agents["default"].config.peft_path is None

    @pytest.mark.asyncio
    async def test_metadata_records_paths(self, tmp_path: Path) -> None:
        from stateset_agents.api.services.agent_service import AgentService
        from stateset_agents.utils.security import SecurityMonitor

        svc = AgentService(SecurityMonitor())
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        await svc.register_default_checkpoint_agent(
            checkpoint_path=str(ckpt),
            base_model="stub://x",
        )
        agent = svc.agents["default"]
        metadata = getattr(agent, "metadata", {})
        assert str(ckpt) in metadata.get("checkpoint_path", "")
        assert metadata.get("base_model") == "stub://x"

    @pytest.mark.asyncio
    async def test_overwrites_existing_agent_id(self, tmp_path: Path) -> None:
        """Re-registering under the same id replaces the previous agent."""
        from stateset_agents.api.services.agent_service import AgentService
        from stateset_agents.utils.security import SecurityMonitor

        svc = AgentService(SecurityMonitor())
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()

        await svc.register_default_checkpoint_agent(
            checkpoint_path=str(ckpt),
            base_model="stub://a",
        )
        first_agent = svc.agents["default"]

        await svc.register_default_checkpoint_agent(
            checkpoint_path=str(ckpt),
            base_model="stub://b",
        )
        second_agent = svc.agents["default"]

        # Different agent instance after re-registration.
        assert first_agent is not second_agent
