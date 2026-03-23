from stateset_agents.api.services.inference_service import InferenceConfig, _parse_model_map


def test_parse_model_map_handles_json_and_fallback_pairs():
    assert _parse_model_map('{"public": "internal", "x": "y"}') == {
        "public": "internal",
        "x": "y",
    }

    assert _parse_model_map("pub1=internal1, pub2=internal2") == {
        "pub1": "internal1",
        "pub2": "internal2",
    }

    assert _parse_model_map("  pub=internal, badpair, x=y ") == {
        "pub": "internal",
        "x": "y",
    }


def test_parse_model_map_rejects_non_string_values():
    assert _parse_model_map('{"valid": "ok", "bad": 123}') == {
        "valid": "ok"
    }


def test_inference_config_from_env_normalizes_values(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "production")
    monkeypatch.setenv("INFERENCE_BACKEND", "vLLM")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "  moonshotai/Kimi-K2.5  ")
    monkeypatch.setenv("INFERENCE_BACKEND_URL", "  ")
    monkeypatch.setenv("INFERENCE_HEALTH_PATH", "  /status  ")

    config = InferenceConfig.from_env()

    assert config.backend == "vllm"
    assert config.default_model == "moonshotai/Kimi-K2.5"
    assert config.base_url == "http://localhost:8000"
    assert config.health_path == "/status"


def test_inference_config_from_env_invalid_numeric_values_fallback(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "production")
    monkeypatch.setenv("INFERENCE_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("INFERENCE_MAX_RETRIES", "-7")
    monkeypatch.setenv("INFERENCE_DEFAULT_MAX_TOKENS", "bad")

    config = InferenceConfig.from_env()

    assert config.timeout_seconds == 120.0
    assert config.max_retries == 0
    assert config.default_max_tokens == 1


def test_inference_config_unknown_backend_falls_back_safely(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("INFERENCE_BACKEND", "mystery-backend")

    config = InferenceConfig.from_env()

    assert config.backend == "stub"


def test_inference_config_empty_default_model_is_none(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "   ")

    config = InferenceConfig.from_env()

    assert config.default_model is None


def test_inference_config_health_path_is_normalized_to_leading_slash(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_HEALTH_PATH", "health")

    config = InferenceConfig.from_env()

    assert config.health_path == "/health"


def test_inference_config_empty_backend_defaults_per_environment(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("INFERENCE_BACKEND", "   ")

    config = InferenceConfig.from_env()

    assert config.backend == "stub"


def test_inference_config_backend_url_is_trimmed(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("INFERENCE_BACKEND_URL", "  http://example.com/api  ")

    config = InferenceConfig.from_env()

    assert config.base_url == "http://example.com/api"


def test_inference_config_respects_stream_usage_flag_truthy_forms(monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("INFERENCE_STREAM_INCLUDE_USAGE", "YES")

    config = InferenceConfig.from_env()

    assert config.include_stream_usage is True
