"""Every packaged starter family is built from a StarterSpec and exposes the
same public contract (the add-a-starter checklist, enforced)."""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

from stateset_agents.core.model_presets import PRESETS
from stateset_agents.training import starter_factory

FAMILIES = sorted(
    {p.starter_module for p in PRESETS.values() if p.starter_module is not None}
)
STANDARD_FUNCTIONS = (
    "get_{ix}_system_prompt",
    "get_{ix}_profile_overrides",
    "get_{ix}_profile_description",
    "summarize_{ix}_config",
    "describe_{ix}_starter_profiles",
    "get_{ix}_config",
    "create_{ix}_agent_config",
    "get_{ix}_gspo_overrides",
    "get_{ix}_gspo_config",
    "validate_{ix}_config",
    "create_{ix}_preview",
    "load_{ix}_config_file",
    "write_{ix}_config_file",
)
STANDARD_CONSTANTS = (
    "BASE_MODEL",
    "SUPPORTED_VARIANTS",
    "TASK_CHOICES",
    "STARTER_PROFILE_CHOICES",
    "STARTER_PROFILE_DESCRIPTIONS",
    "DEFAULT_OUTPUT_DIR",
    "LORA_TARGET_MODULES",
    "CONFIG_SUFFIXES",
)


def _module(family: str):
    return importlib.import_module(f"stateset_agents.training.{family}")


@pytest.mark.parametrize("family", FAMILIES)
def test_family_module_is_spec_built(family):
    mod = _module(family)
    assert isinstance(mod.SPEC, starter_factory.StarterSpec)
    assert mod.SPEC.module == mod.__name__
    assert set(starter_factory.starter_all(mod._SYMBOLS)) <= set(mod.__all__)


@pytest.mark.parametrize("family", FAMILIES)
def test_family_exposes_standard_contract(family):
    mod = _module(family)
    spec = mod.SPEC
    for const in STANDARD_CONSTANTS:
        assert hasattr(mod, f"{spec.symbol_prefix}_{const}"), const
    for template in STANDARD_FUNCTIONS:
        name = template.format(ix=spec.fn_infix)
        assert callable(getattr(mod, name)), name
    assert inspect.iscoroutinefunction(getattr(mod, f"run_{spec.run_suffix}_config"))
    assert inspect.iscoroutinefunction(getattr(mod, f"finetune_{spec.run_suffix}"))
    cls = getattr(mod, spec.config_class_name)
    assert dataclasses.is_dataclass(cls) and cls.__module__ == mod.__name__


@pytest.mark.parametrize("family", FAMILIES)
def test_family_config_has_canonical_field_order_and_validation(family):
    mod = _module(family)
    spec = mod.SPEC
    cls = getattr(mod, spec.config_class_name)
    names = [f.name for f in dataclasses.fields(cls)]
    core = ["model_name"] + [n for n, _ in starter_factory._CORE_FIELDS]
    tail = [n for n, _ in starter_factory._TAIL_FIELDS]
    extras = [n for n, _, _ in spec.extra_fields]
    assert names == core + extras + tail
    cfg = cls()
    assert cfg.model_name == spec.base_model
    assert cfg.lora_target_modules == list(spec.lora_target_modules)
    assert cfg.output_dir == spec.default_output_dir
    assert cfg.validate() == getattr(mod, f"validate_{spec.fn_infix}_config")(cfg)
    assert cfg.system_prompt is not None  # mixin filled it from the spec intro
    assert spec.system_prompt_intro in cfg.system_prompt


@pytest.mark.parametrize("family", FAMILIES)
def test_every_profile_resolves(family):
    mod = _module(family)
    spec = mod.SPEC
    get_config = getattr(mod, f"get_{spec.fn_infix}_config")
    for profile in spec.profile_choices:
        cfg = get_config(starter_profile=profile)
        assert cfg.starter_profile == profile
        for key, value in spec.profile_overrides[profile].items():
            assert getattr(cfg, key) == value, (profile, key)


def test_registry_covers_every_family_module():
    for family in FAMILIES:
        _module(family)
    assert len(FAMILIES) >= 12
