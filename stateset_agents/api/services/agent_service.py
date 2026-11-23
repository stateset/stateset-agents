import copy
import logging
import os
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.conversation import (
    ConversationResetError,
    bind_conversation_guard,
)
from stateset_agents.core.multiturn_context import ConversationConflict
from stateset_agents.exceptions import ATTRIBUTE_VALUE_EXCEPTIONS
from stateset_agents.utils.security import SecurityMonitor

from ..conversation_lifecycle import GatewayConversations
from ..schemas import AgentConfigRequest, ConversationRequest, ConversationResponse

logger = logging.getLogger(__name__)

AGENT_SERVICE_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

TOKENIZE_EXCEPTIONS = ATTRIBUTE_VALUE_EXCEPTIONS


class AgentService:
    """Service for managing agents."""

    def __init__(self, security_monitor: SecurityMonitor):
        self.agents: dict[str, MultiTurnAgent] = {}
        self.conversations: dict[str, list[dict[str, Any]]] = {}
        self.security_monitor = security_monitor
        self.conversation_operations = GatewayConversations()
        self._retiring_agents: dict[str, Any] = {}
        self._deleting_conversations: set[tuple[str, str]] = set()

    def _get_or_create_conversation(
        self,
        agent_id: str,
        conversation_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        conversation_list = self.conversations.setdefault(agent_id, [])
        for conversation in conversation_list:
            if conversation.get("id") == conversation_id:
                if not self._user_can_access_conversation(conversation, user_id):
                    raise HTTPException(
                        status_code=404, detail="Conversation not found"
                    )
                if (
                    user_id
                    and conversation.setdefault("metadata", {}).get("user_id") is None
                ):
                    conversation["metadata"]["user_id"] = user_id
                return conversation

        conversation = {
            "id": conversation_id,
            "messages": [],
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "last_message_at": None,
            "total_tokens": 0,
            "metadata": {},
        }
        if user_id:
            conversation["metadata"]["user_id"] = user_id
        conversation_list.append(conversation)
        return conversation

    @staticmethod
    def _user_can_access_conversation(
        conversation: dict[str, Any], user_id: str | None
    ) -> bool:
        if user_id is None:
            return True

        metadata = conversation.get("metadata")
        if not isinstance(metadata, dict):
            return True

        owner = metadata.get("user_id")
        if owner is None:
            return True

        return bool(owner == user_id)

    def _merge_messages(
        self,
        conversation_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge incoming messages into stored conversation history."""
        if not conversation_messages:
            conversation_messages.extend(incoming_messages)
            return conversation_messages

        if (
            len(incoming_messages) >= len(conversation_messages)
            and incoming_messages[: len(conversation_messages)] == conversation_messages
        ):
            conversation_messages[:] = incoming_messages
            return conversation_messages

        if (
            len(conversation_messages) >= len(incoming_messages)
            and conversation_messages[-len(incoming_messages) :] == incoming_messages
        ):
            return conversation_messages

        conversation_messages.extend(incoming_messages)
        return conversation_messages

    def _should_use_stub_model(self, model_name: str) -> bool:
        environment = os.getenv("API_ENVIRONMENT", "production").lower()
        allow_external = (
            os.getenv("API_ALLOW_EXTERNAL_MODELS", "false").lower() == "true"
        )

        if str(model_name).startswith("stub://"):
            return True

        # Default to stub backend in non-production unless explicitly enabled.
        return environment != "production" and not allow_external

    async def create_agent(self, config: AgentConfigRequest) -> str:
        """Create a new agent."""
        agent_id = str(uuid.uuid4())

        agent_config = AgentConfig(
            model_name=config.model_name,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            system_prompt=config.system_prompt,
            use_chat_template=config.use_chat_template,
            enable_planning=config.enable_planning,
            planning_config=config.planning_config,
            use_stub_model=self._should_use_stub_model(config.model_name),
        )

        agent = MultiTurnAgent(agent_config)
        try:
            await agent.initialize()
        except AGENT_SERVICE_EXCEPTIONS as exc:
            environment = os.getenv("API_ENVIRONMENT", "production").lower()
            if environment != "production" and not agent_config.use_stub_model:
                logger.warning(
                    "Agent initialization failed; falling back to stub backend",
                    extra={"model_name": agent_config.model_name, "agent_id": agent_id},
                    exc_info=True,
                )
                agent_config = replace(agent_config, use_stub_model=True)
                agent = MultiTurnAgent(agent_config)
                await agent.initialize()
            else:
                raise exc

        self.agents[agent_id] = agent
        logger.info(f"Created agent {agent_id}")

        return agent_id

    async def register_default_checkpoint_agent(
        self,
        checkpoint_path: str,
        base_model: str | None = None,
        agent_id: str = "default",
    ) -> str:
        """Register a pre-trained checkpoint as a named agent.

        Called at API startup when ``STATESET_DEFAULT_CHECKPOINT`` is set
        (typically by ``stateset-agents serve --checkpoint <path>``). The
        resulting agent is stored under ``agent_id`` (default: ``"default"``)
        and is immediately reachable via ``GET /agents/default``.

        Args:
            checkpoint_path: Filesystem path to a LoRA adapter directory or a
                full model directory.
            base_model: When the checkpoint is a LoRA adapter, the base model
                name to load it on top of. Pass None for a full-model
                checkpoint.
            agent_id: Stable ID under which the agent is registered.

        Returns:
            The registered ``agent_id``.

        Raises:
            FileNotFoundError: if ``checkpoint_path`` does not exist on disk.
            RuntimeError: if agent initialization fails and we're in
                production mode (no stub fallback).
        """
        from pathlib import Path

        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")

        # Use the base model name when supplied (LoRA case); fall back to the
        # checkpoint dir name (full-model case). When base_model is set, treat
        # the checkpoint as a LoRA adapter and load it on top of the base.
        model_name = base_model or str(ckpt)
        is_stub = self._should_use_stub_model(model_name)
        agent_config = AgentConfig(
            model_name=model_name,
            peft_path=str(ckpt) if (base_model and not is_stub) else None,
            use_stub_model=is_stub,
        )
        agent_metadata_path = str(ckpt)

        agent = MultiTurnAgent(agent_config)
        try:
            await agent.initialize()
        except AGENT_SERVICE_EXCEPTIONS as exc:
            environment = os.getenv("API_ENVIRONMENT", "production").lower()
            if environment != "production":
                logger.warning(
                    "Checkpoint-agent initialization failed; falling back to stub backend",
                    extra={"checkpoint": str(ckpt), "base_model": base_model},
                    exc_info=True,
                )
                agent_config = replace(agent_config, use_stub_model=True)
                agent = MultiTurnAgent(agent_config)
                await agent.initialize()
            else:
                raise RuntimeError(
                    f"Failed to register checkpoint agent at {ckpt}: {exc}"
                ) from exc

        # Attach the checkpoint path as metadata so /v1/agents/{id} can
        # surface what was loaded.
        try:
            agent.metadata["checkpoint_path"] = agent_metadata_path
            agent.metadata["base_model"] = base_model
        except Exception:  # noqa: BLE001 — metadata is best-effort
            pass

        self.agents[agent_id] = agent
        logger.info(
            "Registered checkpoint agent",
            extra={
                "agent_id": agent_id,
                "checkpoint": str(ckpt),
                "base_model": base_model,
            },
        )
        return agent_id

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and clear associated conversation state."""
        owner = self.conversation_operations
        with owner.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return False
            self._retiring_agents[agent_id] = agent
            conv_list = list(self.conversations.get(agent_id, []))
            for conv in conv_list:
                owner.retire(conv)
        planning_manager = getattr(agent, "planning_manager", None)
        if planning_manager and hasattr(planning_manager, "clear_conversation"):
            for conv in conv_list:
                conv_id = conv.get("id")
                if conv_id:
                    planning_manager.clear_conversation(conv_id)

        with owner.lock:
            if self.agents.get(agent_id) is agent:
                self.conversations.pop(agent_id, None)
                del self.agents[agent_id]
            if self._retiring_agents.get(agent_id) is agent:
                self._retiring_agents.pop(agent_id)
        return True

    async def get_conversation_response(
        self, agent_id: str, request: ConversationRequest
    ) -> ConversationResponse:
        """Get response from agent for conversation."""
        # Basic validation (request schema already validates structure/content)
        content = request.messages[0].get("content") if request.messages else None
        if not isinstance(content, str) or not content.strip():
            self.security_monitor.log_security_event(
                "input_validation_failure",
                {"agent_id": agent_id, "input_length": len(str(request.messages))},
            )
            raise HTTPException(status_code=400, detail="Invalid input")

        conv_id = request.conversation_id or str(uuid.uuid4())
        principal = request.user_id
        incoming = copy.deepcopy(request.messages)
        request_context = copy.deepcopy(request.context or {})
        has_context = bool(request_context)
        owner = self.conversation_operations
        try:
            with owner.lock:
                agent = self.agents.get(agent_id)
                if agent is None:
                    raise HTTPException(status_code=404, detail="Agent not found")
                if self._retiring_agents.get(agent_id) is agent:
                    raise ConversationConflict("Agent deletion is still pending")
                conversation = self._get_or_create_conversation(
                    agent_id, conv_id, user_id=principal
                )
                authorized_owner = conversation.get("metadata", {}).get("user_id")

            def current() -> bool:
                return (
                    self.agents.get(agent_id) is agent
                    and self._retiring_agents.get(agent_id) is not agent
                    and any(
                        item is conversation
                        for item in self.conversations.get(agent_id, [])
                    )
                    and self._user_can_access_conversation(conversation, principal)
                    and conversation.get("metadata", {}).get("user_id")
                    == authorized_owner
                )

            with owner.operation(
                (agent_id, conv_id), conversation, current
            ) as operation:
                messages = copy.deepcopy(operation.messages)
                self._merge_messages(messages, incoming)
                # Preserve the service's accepted-user-message policy on failure.
                operation.publish({"messages": messages})
                context = request_context
                context["conversation_id"] = conv_id
                if principal:
                    context["user_id"] = principal
                start_time = time.monotonic()
                with bind_conversation_guard(agent, operation.guard):
                    response_text = await operation.run(
                        lambda: agent.generate_response(
                            copy.deepcopy(messages), context=copy.deepcopy(context)
                        )
                    )
                    processing_time = time.monotonic() - start_time
                    plan_payload = None
                    planning_manager = getattr(agent, "planning_manager", None)
                    if planning_manager is not None:
                        operation.check()
                        plan = planning_manager.get_plan(conv_id)
                        if plan is not None:
                            plan_payload = {
                                "goal": plan.goal,
                                "progress": plan.progress(),
                                "summary": plan.summarize(
                                    max_steps=planning_manager.config.max_steps,
                                    keep_completed=planning_manager.config.keep_completed,
                                ),
                            }
                    operation.check()
                    tokenizer = getattr(agent, "tokenizer", None)
                    if tokenizer:
                        try:
                            tokens_used = len(tokenizer.encode(response_text))
                        except TOKENIZE_EXCEPTIONS:
                            tokens_used = int(len(response_text.split()) * 1.3)
                    else:
                        tokens_used = int(len(response_text.split()) * 1.3)
                    metadata = copy.deepcopy(conversation.get("metadata", {}))
                    if has_context:
                        metadata["context"] = copy.deepcopy(context)
                    if plan_payload is not None:
                        metadata["plan"] = plan_payload
                    response = ConversationResponse(
                        response=response_text,
                        conversation_id=conv_id,
                        tokens_used=tokens_used,
                        processing_time=processing_time,
                        metadata={
                            "agent_id": agent_id,
                            **(
                                {"plan": copy.deepcopy(plan_payload)}
                                if plan_payload is not None
                                else {}
                            ),
                        },
                    )
                    operation.publish(
                        {
                            "messages": messages
                            + [{"role": "assistant", "content": response_text}],
                            "last_message_at": datetime.now(timezone.utc).replace(
                                tzinfo=None
                            ),
                            "metadata": metadata,
                            "total_tokens": int(conversation.get("total_tokens", 0))
                            + int(tokens_used),
                        },
                        response=True,
                    )
                    return response
        except (ConversationConflict, ConversationResetError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except AGENT_SERVICE_EXCEPTIONS as e:
            logger.error(f"Agent response failed: {e}")
            raise HTTPException(status_code=500, detail="Agent response failed") from e

    def delete_conversation(
        self, conversation_id: str, user_id: str | None = None
    ) -> bool:
        """Delete a stored conversation and clear related plan state."""
        owner = self.conversation_operations
        with owner.lock:
            found = next(
                (
                    (aid, conv)
                    for aid, entries in self.conversations.items()
                    for conv in entries
                    if conv.get("id") == conversation_id
                ),
                None,
            )
            if found is None or not self._user_can_access_conversation(
                found[1], user_id
            ):
                return False
            agent_id, conversation = found
            key = (agent_id, conversation_id)
            if key in self._deleting_conversations:
                raise HTTPException(
                    status_code=409, detail="Conversation deletion is still pending"
                )
            self._deleting_conversations.add(key)
            owner.retire(conversation)
            agent = self.agents.get(agent_id)
        try:
            planning_manager = getattr(agent, "planning_manager", None)
            if planning_manager and hasattr(planning_manager, "clear_conversation"):
                planning_manager.clear_conversation(conversation_id)
            with owner.lock:
                entries = self.conversations.get(agent_id, [])
                entries[:] = [entry for entry in entries if entry is not conversation]
        finally:
            with owner.lock:
                self._deleting_conversations.discard(key)
        return True

    def get_conversation_status(
        self, conversation_id: str, user_id: str | None = None
    ) -> dict[str, Any]:
        """Inspect request ownership after applying normal conversation access rules."""
        with self.conversation_operations.lock:
            for entries in self.conversations.values():
                for record in entries:
                    if record.get("id") == conversation_id:
                        if self._user_can_access_conversation(record, user_id):
                            return self.conversation_operations.inspect(record)
                        raise HTTPException(
                            status_code=404, detail="Conversation not found"
                        )
        raise HTTPException(status_code=404, detail="Conversation not found")


# Global instance will be created in dependencies or main
