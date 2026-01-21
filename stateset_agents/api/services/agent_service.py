import logging
import os
import time
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from ..schemas import AgentConfigRequest, ConversationRequest, ConversationResponse
from stateset_agents.utils.security import SecurityMonitor

logger = logging.getLogger(__name__)

class AgentService:
    """Service for managing agents."""

    def __init__(self, security_monitor: SecurityMonitor):
        self.agents: Dict[str, MultiTurnAgent] = {}
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.security_monitor = security_monitor

    def _get_or_create_conversation(
        self,
        agent_id: str,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        conversation_list = self.conversations.setdefault(agent_id, [])
        for conversation in conversation_list:
            if conversation.get("id") == conversation_id:
                return conversation

        conversation = {
            "id": conversation_id,
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_message_at": None,
            "total_tokens": 0,
            "metadata": {},
        }
        if user_id:
            conversation["metadata"]["user_id"] = user_id
        conversation_list.append(conversation)
        return conversation

    def _merge_messages(
        self,
        conversation_messages: List[Dict[str, Any]],
        incoming_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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
        allow_external = os.getenv("API_ALLOW_EXTERNAL_MODELS", "false").lower() == "true"

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
        except Exception as exc:
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

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and clear associated conversation state."""
        agent = self.agents.get(agent_id)
        if agent is None:
            return False

        conv_list = self.conversations.get(agent_id, [])
        planning_manager = getattr(agent, "planning_manager", None)
        if planning_manager and hasattr(planning_manager, "clear_conversation"):
            for conv in conv_list:
                conv_id = conv.get("id")
                if conv_id:
                    planning_manager.clear_conversation(conv_id)

        self.conversations.pop(agent_id, None)
        del self.agents[agent_id]
        return True

    async def get_conversation_response(
        self, agent_id: str, request: ConversationRequest
    ) -> ConversationResponse:
        """Get response from agent for conversation."""
        if agent_id not in self.agents:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent = self.agents[agent_id]

        # Basic validation (request schema already validates structure/content)
        content = request.messages[0].get("content") if request.messages else None
        if not isinstance(content, str) or not content.strip():
            self.security_monitor.log_security_event(
                "input_validation_failure",
                {"agent_id": agent_id, "input_length": len(str(request.messages))},
            )
            raise HTTPException(status_code=400, detail="Invalid input")

        # Get or create conversation
        conv_id = request.conversation_id or str(uuid.uuid4())
        conversation = self._get_or_create_conversation(
            agent_id, conv_id, user_id=request.user_id
        )
        conversation_messages = conversation.setdefault("messages", [])
        incoming_messages = [dict(msg) for msg in request.messages]
        self._merge_messages(conversation_messages, incoming_messages)

        # Generate response
        start_time = time.time()

        try:
            context = dict(request.context or {})
            context.setdefault("conversation_id", conv_id)
            if request.user_id:
                context.setdefault("user_id", request.user_id)
            response_text = await agent.generate_response(
                conversation_messages, context=context
            )
            processing_time = time.time() - start_time

            # Add assistant response to conversation
            conversation_messages.append(
                {"role": "assistant", "content": response_text}
            )

            plan_payload = None
            planning_manager = getattr(agent, "planning_manager", None)
            if planning_manager is not None:
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

            # Calculate tokens used
            if hasattr(agent, "tokenizer") and agent.tokenizer:
                try:
                    # accurate count using tokenizer
                    tokens_used = len(agent.tokenizer.encode(response_text))
                except Exception:
                    # Fallback if tokenizer fails or encoding not possible
                    tokens_used = int(len(response_text.split()) * 1.3)
            else:
                # Rough approximation if no tokenizer
                tokens_used = int(len(response_text.split()) * 1.3)

            conversation["last_message_at"] = datetime.utcnow()
            conversation["total_tokens"] = int(conversation.get("total_tokens", 0)) + int(
                tokens_used
            )
            conversation_metadata = conversation.setdefault("metadata", {})
            if request.context:
                conversation_metadata["context"] = dict(request.context)
            if plan_payload is not None:
                conversation_metadata["plan"] = plan_payload

            return ConversationResponse(
                response=response_text,
                conversation_id=conv_id,
                tokens_used=tokens_used,
                processing_time=processing_time,
                metadata={
                    "agent_id": agent_id,
                    **({"plan": plan_payload} if plan_payload is not None else {}),
                },
            )

        except Exception as e:
            logger.error(f"Agent response failed: {e}")
            raise HTTPException(status_code=500, detail="Agent response failed")

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a stored conversation and clear related plan state."""
        for agent_id, conv_list in self.conversations.items():
            for idx, conv in enumerate(conv_list):
                if conv.get("id") == conversation_id:
                    conv_list.pop(idx)
                    agent = self.agents.get(agent_id)
                    planning_manager = getattr(agent, "planning_manager", None)
                    if planning_manager and hasattr(planning_manager, "clear_conversation"):
                        planning_manager.clear_conversation(conversation_id)
                    return True
        return False

# Global instance will be created in dependencies or main
