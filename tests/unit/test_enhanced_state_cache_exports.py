import importlib


def test_enhanced_state_management_reexports_cache_primitives() -> None:
    state_mod = importlib.import_module("stateset_agents.core.enhanced_state_management")
    cache_mod = importlib.import_module("stateset_agents.core.enhanced_state_cache")

    assert state_mod.CacheEntry is cache_mod.CacheEntry
    assert state_mod.CacheStrategy is cache_mod.CacheStrategy
    assert state_mod.ConsistencyLevel is cache_mod.ConsistencyLevel
    assert state_mod.EvictionPolicy is cache_mod.EvictionPolicy
    assert state_mod.InMemoryCacheBackend is cache_mod.InMemoryCacheBackend
    assert state_mod.MultiCacheBackend is cache_mod.MultiCacheBackend
    assert state_mod.RedisCacheBackend is cache_mod.RedisCacheBackend
