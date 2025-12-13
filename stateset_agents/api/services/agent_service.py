import uuid
import logging
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional
from fastapi import HTTPException
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from ..schemas import AgentConfigRequest, ConversationRequest, ConversationResponse
from utils.security import SecurityMonitor

logger = logging.getLogger(__name__)

class AgentService:
    """Service for managing agents."""

    def __init__(self, security_monitor: SecurityMonitor):
        self.agents: Dict[str, MultiTurnAgent] = {}
        self.conversations: Dict[str, List[Dict]] = {}
        self.security_monitor = security_monitor

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
        if conv_id not in self.conversations:
            self.conversations[conv_id] = []

        # Add user message to conversation
        self.conversations[conv_id].extend(request.messages)

        # Generate response
        start_time = time.time()

        try:
            response_text = await agent.generate_response(request.messages)
            processing_time = time.time() - start_time

            # Add assistant response to conversation
            self.conversations[conv_id].append(
                {"role": "assistant", "content": response_text}
            )

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

            return ConversationResponse(
                response=response_text,
                conversation_id=conv_id,
                tokens_used=tokens_used,
                processing_time=processing_time,
                metadata={"agent_id": agent_id},
            )

        except Exception as e:
            logger.error(f"Agent response failed: {e}")
            raise HTTPException(status_code=500, detail="Agent response failed")

# Global instance will be created in dependencies or main
