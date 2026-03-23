import stateset_agents.core.agent as agent_module
from stateset_agents.core import agent_config, agent_factories


def test_agent_config_reexports_match_dedicated_module() -> None:
    assert agent_module.AgentConfig is agent_config.AgentConfig
    assert agent_module.ConfigValidationError is agent_config.ConfigValidationError


def test_agent_factory_reexports_match_dedicated_module() -> None:
    assert agent_module.AGENT_CONFIGS is agent_factories.AGENT_CONFIGS
    assert agent_module._load_agent_configs is agent_factories._load_agent_configs
    assert agent_module.create_agent is agent_factories.create_agent
    assert agent_module.create_peft_agent is agent_factories.create_peft_agent
    assert agent_module.get_preset_config is agent_factories.get_preset_config
    assert (
        agent_module.load_agent_from_checkpoint
        is agent_factories.load_agent_from_checkpoint
    )
    assert agent_module.save_agent_checkpoint is agent_factories.save_agent_checkpoint
