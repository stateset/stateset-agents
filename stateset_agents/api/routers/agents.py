"""
Agents Router Module

API endpoints for agent management and conversations.
"""

from datetime import datetime
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ..schemas import (
    AgentConfigRequest,
    ConversationRequest,
    ConversationResponse,
    ErrorResponse,
)
from ..models import PaginatedResponse
from ..services.agent_service import AgentService
from ..dependencies import get_current_user, get_security_monitor, AuthenticatedUser
from ..errors import (
    AgentNotFoundError,
    InvalidAgentConfigError,
    ConversationNotFoundError,
    PromptInjectionError,
    InternalError,
)
from ..security import InputValidator, get_api_security_monitor
from ..constants import MAX_MESSAGE_LENGTH
from utils.security import SecurityMonitor

router = APIRouter(prefix="/agents", tags=["agents"])
conversation_router = APIRouter(prefix="/conversations", tags=["conversations"])

# Singleton for now, could be dependency injected per request if needed
security_monitor = SecurityMonitor()
agent_service = AgentService(security_monitor)


def _agent_config_to_dict(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        return config
    if is_dataclass(config):
        return asdict(config)
    return {}


# ============================================================================
# Response Models
# ============================================================================

class AgentCreatedResponse(BaseModel):
    """Response for agent creation."""
    agent_id: str = Field(..., description="Unique identifier for the created agent")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    config: AgentConfigRequest = Field(..., description="Agent configuration")
    message: str = Field("Agent created successfully", description="Status message")


class AgentDetailResponse(BaseModel):
    """Detailed agent information."""
    agent_id: str = Field(..., description="Agent identifier")
    model_name: str = Field(..., description="Model name")
    created_at: datetime = Field(..., description="Creation timestamp")
    conversation_count: int = Field(0, description="Number of conversations")
    total_tokens_used: int = Field(0, description="Total tokens used")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    status: str = Field("active", description="Agent status")


class AgentListResponse(PaginatedResponse):
    """Paginated list of agents."""
    items: List[AgentDetailResponse] = Field(default_factory=list, description="List of agents")


class ConversationSummary(BaseModel):
    """Summary of a conversation."""
    conversation_id: str = Field(..., description="Conversation identifier")
    agent_id: str = Field(..., description="Associated agent ID")
    message_count: int = Field(0, description="Number of messages")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_message_at: Optional[datetime] = Field(None, description="Last message timestamp")
    total_tokens: int = Field(0, description="Total tokens used")


class ConversationListResponse(PaginatedResponse):
    """Paginated list of conversations."""
    items: List[ConversationSummary] = Field(default_factory=list, description="List of conversations")


# ============================================================================
# Agent Endpoints
# ============================================================================

@router.post(
    "",
    response_model=AgentCreatedResponse,
    status_code=201,
    summary="Create Agent",
    description="Create a new AI agent with the specified configuration.",
    responses={
        201: {"description": "Agent created successfully", "model": AgentCreatedResponse},
        400: {"description": "Invalid configuration", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_agent(
    config: AgentConfigRequest,
    user: AuthenticatedUser = Depends(get_current_user),
) -> AgentCreatedResponse:
    """
    Create a new agent with the specified configuration.

    Args:
        config: Agent configuration including model name, temperature, etc.
        user: Authenticated user making the request.

    Returns:
        AgentCreatedResponse with the created agent's ID and details.

    Raises:
        InvalidAgentConfigError: If the configuration is invalid.
        InternalError: If agent creation fails unexpectedly.
    """
    try:
        # Validate system prompt for injection if provided
        if config.system_prompt:
            try:
                InputValidator.validate_string(
                    config.system_prompt,
                    max_length=10000,
                    field_name="system_prompt",
                    check_injection=True,
                )
            except ValueError as e:
                if "potentially harmful content" in str(e).lower():
                    raise PromptInjectionError("system_prompt")
                raise InvalidAgentConfigError(str(e))

        agent_id = await agent_service.create_agent(config)

        return AgentCreatedResponse(
            agent_id=agent_id,
            created_at=datetime.utcnow(),
            config=config,
            message="Agent created successfully",
        )

    except PromptInjectionError:
        raise
    except ValueError as e:
        raise InvalidAgentConfigError(str(e))
    except Exception as e:
        raise InternalError("Failed to create agent", internal_error=e)


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List Agents",
    description="Get a paginated list of all agents.",
    responses={
        200: {"description": "List of agents", "model": AgentListResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
    },
)
async def list_agents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    user: AuthenticatedUser = Depends(get_current_user),
) -> AgentListResponse:
    """
    Get a paginated list of all agents.

    Args:
        page: Page number (1-indexed).
        page_size: Number of items per page (max 100).
        status: Optional filter by agent status.
        user: Authenticated user.

    Returns:
        Paginated list of agent details.
    """
    # Get all agents from service
    all_agents = list(agent_service.agents.items())

    # Apply status filter if provided
    if status:
        all_agents = [(aid, a) for aid, a in all_agents if getattr(a, "status", "active") == status]

    total = len(all_agents)

    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_agents = all_agents[start_idx:end_idx]

    items = [
        AgentDetailResponse(
            agent_id=agent_id,
            model_name=getattr(agent, "model_name", "unknown"),
            created_at=getattr(agent, "created_at", datetime.utcnow()),
            conversation_count=len(agent_service.conversations.get(agent_id, [])),
            total_tokens_used=getattr(agent, "total_tokens", 0),
            config=_agent_config_to_dict(getattr(agent, "config", {})),
            status=getattr(agent, "status", "active"),
        )
        for agent_id, agent in page_agents
    ]

    return AgentListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end_idx < total,
        has_prev=page > 1,
    )


@router.get(
    "/{agent_id}",
    response_model=AgentDetailResponse,
    summary="Get Agent Details",
    description="Get detailed information about a specific agent.",
    responses={
        200: {"description": "Agent details", "model": AgentDetailResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Agent not found", "model": ErrorResponse},
    },
)
async def get_agent(
    agent_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> AgentDetailResponse:
    """
    Get detailed information about a specific agent.

    Args:
        agent_id: The agent's unique identifier.
        user: Authenticated user.

    Returns:
        Detailed agent information.

    Raises:
        AgentNotFoundError: If the agent doesn't exist.
    """
    if agent_id not in agent_service.agents:
        raise AgentNotFoundError(agent_id)

    agent = agent_service.agents[agent_id]

    return AgentDetailResponse(
        agent_id=agent_id,
        model_name=getattr(agent, "model_name", "unknown"),
        created_at=getattr(agent, "created_at", datetime.utcnow()),
        conversation_count=len(agent_service.conversations.get(agent_id, [])),
        total_tokens_used=getattr(agent, "total_tokens", 0),
        config=_agent_config_to_dict(getattr(agent, "config", {})),
        status=getattr(agent, "status", "active"),
    )


@router.delete(
    "/{agent_id}",
    status_code=204,
    summary="Delete Agent",
    description="Delete an agent and all associated conversations.",
    responses={
        204: {"description": "Agent deleted successfully"},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Agent not found", "model": ErrorResponse},
    },
)
async def delete_agent(
    agent_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> None:
    """
    Delete an agent and all associated conversations.

    Args:
        agent_id: The agent's unique identifier.
        user: Authenticated user.

    Raises:
        AgentNotFoundError: If the agent doesn't exist.
    """
    if agent_id not in agent_service.agents:
        raise AgentNotFoundError(agent_id)

    # Remove agent and associated data
    del agent_service.agents[agent_id]
    agent_service.conversations.pop(agent_id, None)


# ============================================================================
# Conversation Endpoints
# ============================================================================

@conversation_router.post(
    "",
    response_model=ConversationResponse,
    summary="Chat with Agent",
    description="Send a message to an agent and get a response.",
    responses={
        200: {"description": "Successful response", "model": ConversationResponse},
        400: {"description": "Invalid input", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Agent not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def converse_with_agent(
    request: ConversationRequest,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser = Depends(get_current_user),
    monitor: SecurityMonitor = Depends(get_security_monitor),
) -> ConversationResponse:
    """
    Have a conversation with an agent.

    Sends user messages to the agent and returns the generated response.
    Supports both single-turn and multi-turn conversations.

    Args:
        request: Conversation request with messages.
        background_tasks: FastAPI background tasks for async logging.
        user: Authenticated user.
        monitor: Security monitor for event logging.

    Returns:
        ConversationResponse with the agent's reply.

    Raises:
        AgentNotFoundError: If the specified agent doesn't exist.
        PromptInjectionError: If prompt injection is detected.
        InternalError: If response generation fails.
    """
    try:
        # Validate messages for prompt injection
        api_security = get_api_security_monitor()

        for i, msg in enumerate(request.messages):
            content = msg.get("content", "")
            try:
                _, security_event = InputValidator.validate_string(
                    content,
                    max_length=MAX_MESSAGE_LENGTH,
                    field_name=f"messages[{i}].content",
                    check_injection=True,
                )
            except ValueError as e:
                if "potentially harmful content" in str(e).lower():
                    _, _, patterns = InputValidator.detect_prompt_injection(content)
                    api_security.log_prompt_injection_attempt(
                        client_ip="unknown",
                        path="/conversations",
                        content_preview=content[:100],
                        patterns=patterns,
                        user_id=user.user_id if user else None,
                    )
                    raise PromptInjectionError(f"messages[{i}].content")
                raise HTTPException(status_code=400, detail=str(e))

            if security_event:
                api_security.log_event(security_event)

        # Use a default agent for demo purposes
        agent_id = getattr(request, "agent_id", "default")

        # Create a temporary agent if none exists
        if agent_id not in agent_service.agents:
            config = AgentConfigRequest(model_name="gpt2")
            agent_id = await agent_service.create_agent(config)

        response = await agent_service.get_conversation_response(agent_id, request)

        # Log the conversation for analytics
        async def _log_conversation_event(event_type: str, details: Dict[str, Any]) -> None:
            monitor.log_security_event(event_type, details)

        background_tasks.add_task(
            _log_conversation_event,
            "conversation",
            {
                "agent_id": agent_id,
                "conversation_id": response.conversation_id,
                "tokens_used": response.tokens_used,
                "processing_time": response.processing_time,
                "user_id": user.user_id if user else None,
            },
        )

        return response

    except (PromptInjectionError, AgentNotFoundError):
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise InternalError("Conversation failed", internal_error=e)


@conversation_router.get(
    "",
    response_model=ConversationListResponse,
    summary="List Conversations",
    description="Get a paginated list of conversations.",
    responses={
        200: {"description": "List of conversations", "model": ConversationListResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
    },
)
async def list_conversations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationListResponse:
    """
    Get a paginated list of conversations.

    Args:
        page: Page number (1-indexed).
        page_size: Number of items per page (max 100).
        agent_id: Optional filter by agent ID.
        user: Authenticated user.

    Returns:
        Paginated list of conversation summaries.
    """
    # Gather all conversations
    all_convs: List[ConversationSummary] = []

    for aid, conv_list in agent_service.conversations.items():
        if agent_id and aid != agent_id:
            continue

        for conv in conv_list:
            all_convs.append(ConversationSummary(
                conversation_id=conv.get("id", "unknown"),
                agent_id=aid,
                message_count=len(conv.get("messages", [])),
                created_at=conv.get("created_at", datetime.utcnow()),
                last_message_at=conv.get("last_message_at"),
                total_tokens=conv.get("total_tokens", 0),
            ))

    total = len(all_convs)

    # Sort by creation time (newest first)
    all_convs.sort(key=lambda c: c.created_at, reverse=True)

    # Paginate
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_convs = all_convs[start_idx:end_idx]

    return ConversationListResponse(
        items=page_convs,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end_idx < total,
        has_prev=page > 1,
    )


@conversation_router.get(
    "/{conversation_id}",
    summary="Get Conversation",
    description="Get details of a specific conversation.",
    responses={
        200: {"description": "Conversation details"},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Conversation not found", "model": ErrorResponse},
    },
)
async def get_conversation(
    conversation_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get details of a specific conversation.

    Args:
        conversation_id: The conversation's unique identifier.
        user: Authenticated user.

    Returns:
        Full conversation details including message history.

    Raises:
        ConversationNotFoundError: If the conversation doesn't exist.
    """
    # Search for conversation across all agents
    for aid, conv_list in agent_service.conversations.items():
        for conv in conv_list:
            if conv.get("id") == conversation_id:
                return {
                    "conversation_id": conversation_id,
                    "agent_id": aid,
                    "messages": conv.get("messages", []),
                    "created_at": conv.get("created_at"),
                    "last_message_at": conv.get("last_message_at"),
                    "total_tokens": conv.get("total_tokens", 0),
                    "metadata": conv.get("metadata", {}),
                }

    raise ConversationNotFoundError(conversation_id)


@conversation_router.delete(
    "/{conversation_id}",
    status_code=204,
    summary="Delete Conversation",
    description="Delete a specific conversation.",
    responses={
        204: {"description": "Conversation deleted successfully"},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Conversation not found", "model": ErrorResponse},
    },
)
async def delete_conversation(
    conversation_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> None:
    """
    Delete a specific conversation.

    Args:
        conversation_id: The conversation's unique identifier.
        user: Authenticated user.

    Raises:
        ConversationNotFoundError: If the conversation doesn't exist.
    """
    # Search and delete conversation
    for aid, conv_list in agent_service.conversations.items():
        for i, conv in enumerate(conv_list):
            if conv.get("id") == conversation_id:
                conv_list.pop(i)
                return

    raise ConversationNotFoundError(conversation_id)
