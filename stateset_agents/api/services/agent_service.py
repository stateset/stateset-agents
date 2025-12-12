import uuid
import logging
import time
from typing import Dict, List, Optional
from fastapi import HTTPException
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from ..schemas import AgentConfigRequest, ConversationRequest, ConversationResponse
from utils.security import InputValidator, SecurityMonitor

logger = logging.getLogger(__name__)

class AgentService:
    """Service for managing agents."""

    def __init__(self, security_monitor: SecurityMonitor):
        self.agents: Dict[str, MultiTurnAgent] = {}
        self.conversations: Dict[str, List[Dict]] = {}
        self.security_monitor = security_monitor

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
        )

        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

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

        # Validate input
        if not InputValidator.validate_string(request.messages[0]["content"]):
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
