"""
Tests for GRPO handler context updates and access controls.
"""

import contextlib

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from stateset_agents.api.grpo import service as grpo_service
from stateset_agents.api.grpo.handlers import ConversationHandler, TrainingHandler
from stateset_agents.api.grpo.models import GRPOConversationRequest, GRPOTrainingRequest
from stateset_agents.api.grpo.router_v1 import create_v1_router
from stateset_agents.api.grpo.state import get_state_manager, reset_state_manager
from tests.api.asgi_client import SyncASGIClient


class _RequestContext:
    """Minimal request context used by route dependency injection in tests."""

    def __init__(
        self,
        request_id: str,
        user_id: str,
        roles: list[str],
        api_key: str | None = None,
        client: str = "test-client",
    ) -> None:
        self.request_id = request_id
        self.user_id = user_id
        self.roles = roles
        self.api_key = api_key
        self.client = client


@contextlib.contextmanager
def _grpo_service_test_app():
    """Create a minimal GRPO app with in-memory handlers and dependency overrides."""
    reset_state_manager()

    training_handler = TrainingHandler({})
    multiturn_agent = MagicMock()
    conversation_handler = ConversationHandler({"multiturn_agent": multiturn_agent})

    async def verify_request(request: Request) -> _RequestContext:
        user_id = request.headers.get("x-user-id", "anonymous")
        roles_header = request.headers.get("x-user-roles", "user")
        roles = [role.strip() for role in roles_header.split(",") if role.strip()]
        if not roles:
            roles = ["user"]
        return _RequestContext(
            request_id=request.headers.get("x-request-id", "test-req-id"),
            user_id=user_id,
            roles=roles,
            api_key=request.headers.get("x-api-key"),
        )

    app = FastAPI()

    original_training_handler = grpo_service._training_handler
    original_conversation_handler = grpo_service._conversation_handler
    original_services = grpo_service._services

    grpo_service._training_handler = training_handler
    grpo_service._conversation_handler = conversation_handler
    grpo_service._services = {"multiturn_agent": multiturn_agent}

    app.dependency_overrides[grpo_service.verify_request] = verify_request
    grpo_service._register_routes(app)
    app.include_router(
        create_v1_router(
            training_handler=training_handler,
            conversation_handler=conversation_handler,
            services={},
            verify_request=verify_request,
        )
    )

    try:
        yield {
            "app": app,
            "training_handler": training_handler,
            "conversation_handler": conversation_handler,
            "multiturn_agent": multiturn_agent,
        }
    finally:
        grpo_service._training_handler = original_training_handler
        grpo_service._conversation_handler = original_conversation_handler
        grpo_service._services = original_services
        app.dependency_overrides.pop(grpo_service.verify_request, None)


@pytest.mark.asyncio
async def test_grpo_handler_passes_context_updates():
    reset_state_manager()
    state = get_state_manager()
    conversation_id = "conv_123"
    state.create_conversation(conversation_id=conversation_id, user_id="user_1")

    multiturn_agent = MagicMock()
    multiturn_agent.continue_conversation = AsyncMock(
        return_value=[{"role": "assistant", "content": "ok"}]
    )
    multiturn_agent.get_conversation_summary = MagicMock(
        return_value={"conversation_id": conversation_id}
    )

    handler = ConversationHandler({"multiturn_agent": multiturn_agent})
    request = GRPOConversationRequest(
        message="What should we do next?",
        conversation_id=conversation_id,
        context={"plan_update": {"action": "advance"}},
    )

    response = await handler.handle_message(request, user_id="user_1")

    multiturn_agent.continue_conversation.assert_awaited_once()
    kwargs = multiturn_agent.continue_conversation.call_args.kwargs
    assert kwargs["context_update"] == {"plan_update": {"action": "advance"}}
    assert response.conversation_id == conversation_id


@pytest.mark.asyncio
async def test_conversation_handler_rejects_user_id_spoofing():
    reset_state_manager()
    state = get_state_manager()
    state.create_conversation(conversation_id="conv_123", user_id="owner")

    handler = ConversationHandler({})
    request = GRPOConversationRequest(
        message="Who are you?",
        conversation_id="conv_123",
        user_id="other_user",
    )

    with pytest.raises(PermissionError, match="Conversation user mismatch"):
        await handler.handle_message(request, user_id="owner")


@pytest.mark.asyncio
async def test_conversation_handler_allows_admin_access_to_other_users_conversation():
    reset_state_manager()
    state = get_state_manager()
    state.create_conversation(conversation_id="conv_admin", user_id="owner")

    multiturn_agent = MagicMock()
    multiturn_agent.continue_conversation = AsyncMock(
        return_value=[{"role": "assistant", "content": "ok"}]
    )
    multiturn_agent.get_conversation_summary = MagicMock(
        return_value={"conversation_id": "conv_admin"}
    )

    handler = ConversationHandler({"multiturn_agent": multiturn_agent})
    request = GRPOConversationRequest(
        message="status",
        conversation_id="conv_admin",
        user_id="owner",
    )

    response = await handler.handle_message(
        request,
        user_id="reviewer",
        user_roles=["admin"],
    )

    assert response.conversation_id == "conv_admin"


@pytest.mark.asyncio
async def test_conversation_handler_end_conversation_enforces_owner_or_admin_only():
    reset_state_manager()
    state = get_state_manager()
    state.create_conversation(conversation_id="conv_close", user_id="owner")

    handler = ConversationHandler({})

    with pytest.raises(PermissionError, match="Conversation access denied"):
        handler.end_conversation("conv_close", user_id="intruder")


@pytest.mark.asyncio
async def test_training_handler_respects_job_ownership():
    reset_state_manager()
    handler = TrainingHandler({})
    request = GRPOTrainingRequest(prompts=["optimize", "reward"], num_iterations=1)

    response = await handler.start_training(request, "owner", "req-1")

    assert handler.get_job_status(response.job_id, "intruder") is None
    assert handler.cancel_job(response.job_id, "intruder") is False
    assert handler.get_job_status(response.job_id, "admin", ["admin"]) is not None
    assert handler.cancel_job(response.job_id, "admin", ["admin"]) is True


@pytest.mark.asyncio
async def test_api_endpoints_restrict_job_access_by_user():
    with _grpo_service_test_app() as fixture:
        app = fixture["app"]
        training_handler = fixture["training_handler"]

        request = GRPOTrainingRequest(prompts=["safety"], num_iterations=1)
        train_response = await training_handler.start_training(request, "owner", "req-2")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (
                (await client.get(
                    f"/api/status/{train_response.job_id}",
                    headers={"x-user-id": "intruder", "x-user-roles": "user"},
                )).status_code
                == 404
            )

            assert (
                (await client.get(
                    f"/v1/jobs/{train_response.job_id}",
                    headers={"x-user-id": "intruder", "x-user-roles": "user"},
                )).status_code
                == 404
            )

            delete_response = await client.delete(
                f"/v1/jobs/{train_response.job_id}",
                headers={"x-user-id": "owner", "x-user-roles": "user"},
            )
            assert delete_response.status_code == 200
            assert delete_response.json()["job_id"] == train_response.job_id


@pytest.mark.asyncio
async def test_api_batch_endpoints_restrict_to_job_owner():
    with _grpo_service_test_app() as fixture:
        app = fixture["app"]
        training_handler = fixture["training_handler"]

        owner_job = await training_handler.start_training(
            GRPOTrainingRequest(prompts=["owner prompt"], num_iterations=1),
            "owner",
            "batch-owner",
        )
        intruder_job = await training_handler.start_training(
            GRPOTrainingRequest(prompts=["intruder prompt"], num_iterations=1),
            "intruder",
            "batch-intruder",
        )

        headers = {"x-user-roles": "user"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            status_response = await client.post(
                "/api/batch/status",
                headers={**headers, "x-user-id": "intruder"},
                json={"job_ids": [owner_job.job_id, intruder_job.job_id]},
            )
            assert status_response.status_code == 200
            status_payload = status_response.json()
            assert owner_job.job_id in status_payload["not_found"]
            assert intruder_job.job_id in status_payload["jobs"]
            assert status_payload["jobs"][intruder_job.job_id]["status"] in {
                "starting",
                "running",
                "failed",
                "cancelled",
                "completed",
            }

            status_v1_response = await client.post(
                "/v1/batch/status",
                headers={**headers, "x-user-id": "intruder"},
                json={"job_ids": [owner_job.job_id, intruder_job.job_id]},
            )
            assert status_v1_response.status_code == 200
            status_v1_payload = status_v1_response.json()
            assert owner_job.job_id in status_v1_payload["not_found"]
            assert intruder_job.job_id in status_v1_payload["jobs"]

            cancel_response = await client.post(
                "/api/batch/cancel",
                headers={**headers, "x-user-id": "intruder"},
                json={"job_ids": [owner_job.job_id, intruder_job.job_id]},
            )
            assert cancel_response.status_code == 200
            cancel_payload = cancel_response.json()
            assert owner_job.job_id in cancel_payload["not_found"]
            assert cancel_payload["cancelled"] == [intruder_job.job_id]

            cancel_v1_response = await client.post(
                "/v1/batch/cancel",
                headers={**headers, "x-user-id": "intruder"},
                json={"job_ids": [owner_job.job_id, intruder_job.job_id]},
            )
            assert cancel_v1_response.status_code == 200
            cancel_v1_payload = cancel_v1_response.json()
            assert owner_job.job_id in cancel_v1_payload["not_found"]
            assert cancel_v1_payload["cancelled"] == []
            assert intruder_job.job_id in cancel_v1_payload["already_completed"]


@pytest.mark.asyncio
async def test_api_batch_endpoints_allow_admin_full_visibility():
    with _grpo_service_test_app() as fixture:
        app = fixture["app"]
        training_handler = fixture["training_handler"]

        owner_job = await training_handler.start_training(
            GRPOTrainingRequest(prompts=["owner prompt"], num_iterations=1),
            "owner",
            "batch-owner",
        )
        intruder_job = await training_handler.start_training(
            GRPOTrainingRequest(prompts=["intruder prompt"], num_iterations=1),
            "intruder",
            "batch-intruder",
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            status_response = await client.post(
                "/api/batch/status",
                headers={"x-user-id": "security-admin", "x-user-roles": "admin"},
                json={"job_ids": [owner_job.job_id, intruder_job.job_id]},
            )
            assert status_response.status_code == 200
            status_payload = status_response.json()
            assert owner_job.job_id in status_payload["jobs"]
            assert intruder_job.job_id in status_payload["jobs"]

            status_v1_response = await client.post(
                "/v1/batch/status",
                headers={"x-user-id": "security-admin", "x-user-roles": "admin"},
                json={"job_ids": [owner_job.job_id, intruder_job.job_id]},
            )
            assert status_v1_response.status_code == 200
            status_v1_payload = status_v1_response.json()
            assert owner_job.job_id in status_v1_payload["jobs"]
            assert intruder_job.job_id in status_v1_payload["jobs"]

            cancel_response = await client.post(
                "/api/batch/cancel",
                headers={"x-user-id": "security-admin", "x-user-roles": "admin"},
                json={"job_ids": [owner_job.job_id, intruder_job.job_id]},
            )
            assert cancel_response.status_code == 200
            cancel_payload = cancel_response.json()
            assert owner_job.job_id in cancel_payload["cancelled"]
            assert intruder_job.job_id in cancel_payload["cancelled"]


@pytest.mark.asyncio
async def test_api_batch_cancel_marks_completed_jobs_as_already_completed():
    with _grpo_service_test_app() as fixture:
        app = fixture["app"]
        training_handler = fixture["training_handler"]
        state = get_state_manager()

        active_job = await training_handler.start_training(
            GRPOTrainingRequest(prompts=["active"], num_iterations=1),
            "owner",
            "batch-owner",
        )
        completed_job = await training_handler.start_training(
            GRPOTrainingRequest(prompts=["done"], num_iterations=1),
            "owner",
            "batch-owner",
        )
        state.update_job(completed_job.job_id, status="completed")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            cancel_response = await client.post(
                "/api/batch/cancel",
                headers={"x-user-id": "owner", "x-user-roles": "user"},
                json={"job_ids": [active_job.job_id, completed_job.job_id]},
            )
            assert cancel_response.status_code == 200
            cancel_payload = cancel_response.json()
            assert active_job.job_id in cancel_payload["cancelled"]
            assert completed_job.job_id in cancel_payload["already_completed"]
            assert completed_job.job_id not in cancel_payload["cancelled"]

            cancel_v1_response = await client.post(
                "/v1/batch/cancel",
                headers={"x-user-id": "owner", "x-user-roles": "user"},
                json={"job_ids": [active_job.job_id, completed_job.job_id]},
            )
            assert cancel_v1_response.status_code == 200
            cancel_v1_payload = cancel_v1_response.json()
            assert completed_job.job_id in cancel_v1_payload["already_completed"]


def test_api_endpoints_restrict_conversation_access_by_user():
    with _grpo_service_test_app() as fixture:
        app = fixture["app"]
        multiturn_agent = fixture["multiturn_agent"]
        state = get_state_manager()
        state.create_conversation(conversation_id="conv_owner", user_id="owner")

        multiturn_agent.continue_conversation = AsyncMock(
            return_value=[{"role": "assistant", "content": "ok"}]
        )
        multiturn_agent.get_conversation_summary = MagicMock(
            return_value={"conversation_id": "conv_owner"}
        )

        with SyncASGIClient(app) as client:
            assert (
                client.post(
                    "/api/chat",
                    headers={"x-user-id": "intruder", "x-user-roles": "user"},
                    json={"message": "hello", "conversation_id": "conv_owner"},
                ).status_code
                == 403
            )

            assert (
                client.post(
                    "/v1/chat",
                    headers={"x-user-id": "intruder", "x-user-roles": "user"},
                    json={"message": "hello", "conversation_id": "conv_owner"},
                ).status_code
                == 403
            )

            assert (
                client.delete(
                    "/v1/conversations/conv_owner",
                    headers={"x-user-id": "intruder", "x-user-roles": "user"},
                ).status_code
                == 403
            )


def test_api_conversation_user_id_payload_spoofing_blocked():
    with _grpo_service_test_app() as fixture:
        app = fixture["app"]
        multiturn_agent = fixture["multiturn_agent"]
        state = get_state_manager()
        state.create_conversation(conversation_id="conv_owner", user_id="owner")

        multiturn_agent.continue_conversation = AsyncMock(
            return_value=[{"role": "assistant", "content": "ok"}]
        )
        multiturn_agent.get_conversation_summary = MagicMock(
            return_value={"conversation_id": "conv_owner"}
        )

        with SyncASGIClient(app) as client:
            assert (
                client.post(
                    "/api/chat",
                    headers={"x-user-id": "intruder", "x-user-roles": "user"},
                    json={
                        "message": "hello",
                        "conversation_id": "conv_owner",
                        "user_id": "owner",
                    },
                ).status_code
                == 403
            )

            assert (
                client.post(
                    "/v1/chat",
                    headers={"x-user-id": "intruder", "x-user-roles": "user"},
                    json={
                        "message": "hello",
                        "conversation_id": "conv_owner",
                        "user_id": "owner",
                    },
                ).status_code
                == 403
            )
