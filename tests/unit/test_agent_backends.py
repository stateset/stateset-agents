from core.agent_backends import create_stub_backend


def test_create_stub_backend_uses_configuration_values():
    backend = create_stub_backend(
        stub_responses=["first", "second"],
        max_new_tokens=42,
        temperature=0.5,
        top_p=0.9,
        top_k=10,
        do_sample=False,
        repetition_penalty=1.25,
        pad_token_id=7,
        eos_token_id=8,
    )

    prompt = backend.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=False
    )
    assert prompt.startswith("User: Hello")

    response_one = backend.model.generate("prompt")
    response_two = backend.model.generate("prompt")

    assert response_one.startswith("first")
    assert response_two.startswith("second")

    assert backend.generation_config.max_new_tokens == 42
    assert backend.generation_config.temperature == 0.5
    assert backend.generation_config.pad_token_id == 7
    assert backend.generation_config.eos_token_id == 8
