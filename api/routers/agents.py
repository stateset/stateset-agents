from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict
from ..schemas import AgentConfigRequest, ConversationRequest, ConversationResponse, ErrorResponse
from ..services.agent_service import AgentService
from ..dependencies import get_current_user, get_security_monitor
from utils.security import SecurityMonitor

router = APIRouter(prefix="/agents", tags=["agents"])
conversation_router = APIRouter(prefix="/conversations", tags=["conversations"])

# Singleton for now, could be dependency injected per request if needed
security_monitor = SecurityMonitor()
agent_service = AgentService(security_monitor)

@router.post(
    "",
    summary="Create Agent",
    description="Create a new AI agent with the specified configuration",
    response_model=str,
    responses={
        201: {"description": "Agent created successfully"},
        400: {"description": "Invalid configuration", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_agent(
    config: AgentConfigRequest, 
    user_data: Dict = Depends(get_current_user)
):
    """Create a new agent."""
    try:
        agent_id = await agent_service.create_agent(config)
        return agent_id
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create agent")

@conversation_router.post(
    "",
    response_model=ConversationResponse,
    summary="Chat with Agent",
    description="Send a message to an agent and get a response",
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Agent not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def converse_with_agent(
    request: ConversationRequest,
    background_tasks: BackgroundTasks,
    user_data: Dict = Depends(get_current_user),
    monitor: SecurityMonitor = Depends(get_security_monitor)
):
    """Have a conversation with an agent."""
    try:
        # Use a default agent for demo purposes
        # In production, you'd specify agent_id in the request
        agent_id = getattr(request, "agent_id", "default")

        # For demo, create a temporary agent if none exists
        if agent_id not in agent_service.agents:
            config = AgentConfigRequest(model_name="gpt2")
            agent_id = await agent_service.create_agent(config)

        response = await agent_service.get_conversation_response(agent_id, request)

        # Log the conversation for analytics
        background_tasks.add_task(
            monitor.log_security_event,
            "conversation",
            {
                "agent_id": agent_id,
                "conversation_id": response.conversation_id,
                "tokens_used": response.tokens_used,
                "processing_time": response.processing_time,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Conversation failed")
