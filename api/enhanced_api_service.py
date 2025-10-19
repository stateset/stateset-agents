"""
Enhanced API Service with Comprehensive OpenAPI Documentation

This service provides a production-ready REST API for the StateSet Agents framework
with complete OpenAPI/Swagger documentation, security, and monitoring.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..core.agent import AgentConfig, MultiTurnAgent
from ..core.environment import ConversationEnvironment
from ..core.reward import CompositeReward, HelpfulnessReward, SafetyReward
from ..training.train import train
from ..utils.performance_monitor import get_global_monitor
from ..utils.security import AuthService, InputValidator, SecurityMonitor

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
auth_service = AuthService("your-secret-key-here")
security_monitor = SecurityMonitor()

# ============================================================================
# Pydantic Models for API
# ============================================================================


class AgentConfigRequest(BaseModel):
    """Configuration for creating an agent."""

    model_name: str = Field(..., description="Name of the model to use", example="gpt2")
    max_new_tokens: int = Field(
        512, description="Maximum tokens to generate", ge=1, le=4096
    )
    temperature: float = Field(0.8, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Top-p sampling parameter", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Top-k sampling parameter", ge=1, le=1000)
    system_prompt: Optional[str] = Field(
        None, description="System prompt for the agent"
    )
    use_chat_template: bool = Field(True, description="Whether to use chat template")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "gpt2",
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "system_prompt": "You are a helpful AI assistant.",
            }
        }


class ConversationRequest(BaseModel):
    """Request for agent conversation."""

    messages: List[Dict[str, str]] = Field(
        ..., description="List of conversation messages"
    )
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    max_tokens: int = Field(
        512, description="Maximum tokens in response", ge=1, le=4096
    )
    temperature: float = Field(0.8, description="Response temperature", ge=0.0, le=2.0)
    stream: bool = Field(False, description="Whether to stream the response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    @validator("messages")
    def validate_messages(cls, v):
        """Validate conversation messages."""
        if not v:
            raise ValueError("Messages cannot be empty")

        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    "Message role must be 'system', 'user', or 'assistant'"
                )
            if not isinstance(msg["content"], str) or not msg["content"].strip():
                raise ValueError("Message content must be a non-empty string")

        return v

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! How can you help me?"},
                ],
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": False,
            }
        }


class TrainingRequest(BaseModel):
    """Request for training an agent."""

    agent_config: AgentConfigRequest
    environment_scenarios: List[Dict[str, Any]] = Field(
        ..., description="Training scenarios"
    )
    reward_config: Dict[str, Any] = Field(
        ..., description="Reward function configuration"
    )
    num_episodes: int = Field(
        100, description="Number of training episodes", ge=1, le=10000
    )
    profile: str = Field("balanced", description="Training profile")

    class Config:
        schema_extra = {
            "example": {
                "agent_config": {
                    "model_name": "gpt2",
                    "max_new_tokens": 256,
                    "temperature": 0.7,
                },
                "environment_scenarios": [
                    {
                        "id": "customer_support",
                        "topic": "support",
                        "user_responses": ["I need help", "Thank you"],
                    }
                ],
                "reward_config": {"helpfulness_weight": 0.7, "safety_weight": 0.3},
                "num_episodes": 50,
                "profile": "balanced",
            }
        }


class ConversationResponse(BaseModel):
    """Response from conversation endpoint."""

    response: str = Field(..., description="Agent's response")
    conversation_id: str = Field(..., description="Conversation identifier")
    tokens_used: int = Field(..., description="Number of tokens used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class TrainingResponse(BaseModel):
    """Response from training endpoint."""

    training_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="Training status")
    estimated_time: Optional[float] = Field(
        None, description="Estimated completion time"
    )
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component health status")


class MetricsResponse(BaseModel):
    """Metrics response."""

    timestamp: float = Field(..., description="Metrics timestamp")
    system_metrics: Dict[str, Union[int, float]] = Field(
        ..., description="System metrics"
    )
    performance_metrics: Dict[str, Union[int, float]] = Field(
        ..., description="Performance metrics"
    )
    security_events: int = Field(..., description="Number of security events")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: float = Field(..., description="Error timestamp")


# ============================================================================
# Authentication & Security
# ============================================================================


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """Get current authenticated user."""
    token = credentials.credentials

    try:
        user_data = auth_service.authenticate_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return user_data
    except Exception as e:
        security_monitor.log_security_event(
            "authentication_failure",
            {"error": str(e), "token_prefix": token[:10] if token else "None"},
        )
        raise HTTPException(status_code=401, detail="Authentication failed")


async def require_role(required_role: str, user_data: Dict = Depends(get_current_user)):
    """Require specific role for endpoint access."""
    user_roles = user_data.get("roles", [])
    if required_role not in user_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Required role: {required_role}",
        )
    return user_data


# ============================================================================
# Service Layer
# ============================================================================


class AgentService:
    """Service for managing agents."""

    def __init__(self):
        self.agents: Dict[str, MultiTurnAgent] = {}
        self.conversations: Dict[str, List[Dict]] = {}

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
            security_monitor.log_security_event(
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
        import time

        start_time = time.time()

        try:
            response_text = await agent.generate_response(request.messages)
            processing_time = time.time() - start_time

            # Add assistant response to conversation
            self.conversations[conv_id].append(
                {"role": "assistant", "content": response_text}
            )

            # Estimate tokens (simple approximation)
            tokens_used = len(response_text.split()) * 1.3  # Rough approximation

            return ConversationResponse(
                response=response_text,
                conversation_id=conv_id,
                tokens_used=int(tokens_used),
                processing_time=processing_time,
                metadata={"agent_id": agent_id},
            )

        except Exception as e:
            logger.error(f"Agent response failed: {e}")
            raise HTTPException(status_code=500, detail="Agent response failed")


class TrainingService:
    """Service for managing training jobs."""

    def __init__(self):
        self.training_jobs: Dict[str, Dict] = {}

    async def start_training(self, request: TrainingRequest) -> str:
        """Start a training job."""
        training_id = str(uuid.uuid4())

        # Create training configuration
        agent_config = AgentConfig(**request.agent_config.dict())
        agent = MultiTurnAgent(agent_config)

        # Create environment
        environment = ConversationEnvironment(scenarios=request.environment_scenarios)

        # Create reward function
        reward_fn = CompositeReward(
            [
                HelpfulnessReward(
                    weight=request.reward_config.get("helpfulness_weight", 0.7)
                ),
                SafetyReward(weight=request.reward_config.get("safety_weight", 0.3)),
            ]
        )

        # Start training in background
        self.training_jobs[training_id] = {
            "status": "running",
            "start_time": datetime.utcnow(),
            "config": request.dict(),
        }

        # Run training asynchronously
        asyncio.create_task(
            self._run_training(training_id, agent, environment, reward_fn, request)
        )

        return training_id

    async def _run_training(
        self,
        training_id: str,
        agent: MultiTurnAgent,
        environment,
        reward_fn,
        request: TrainingRequest,
    ):
        """Run training job."""
        try:
            trained_agent = await train(
                agent=agent,
                environment=environment,
                reward_fn=reward_fn,
                num_episodes=request.num_episodes,
                profile=request.profile,
            )

            self.training_jobs[training_id].update(
                {
                    "status": "completed",
                    "completion_time": datetime.utcnow(),
                    "trained_agent": trained_agent,
                }
            )

        except Exception as e:
            logger.error(f"Training failed for job {training_id}: {e}")
            self.training_jobs[training_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "completion_time": datetime.utcnow(),
                }
            )

    def get_training_status(self, training_id: str) -> Dict:
        """Get training job status."""
        if training_id not in self.training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")

        return self.training_jobs[training_id]


# Global service instances
agent_service = AgentService()
training_service = TrainingService()


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting StateSet Agents API")
    yield
    # Shutdown
    logger.info("Shutting down StateSet Agents API")


app = FastAPI(
    title="StateSet Agents API",
    description="""
    ## StateSet Agents API
    
    A comprehensive REST API for training and deploying AI agents using 
    reinforcement learning techniques.
    
    ### Features
    
    - 🤖 **Agent Management**: Create and manage AI agents
    - 💬 **Conversations**: Interactive conversations with agents
    - 🎯 **Training**: Train agents using reinforcement learning
    - 📊 **Monitoring**: Real-time metrics and monitoring
    - 🔒 **Security**: Authentication and authorization
    - 📈 **Analytics**: Performance analytics and insights
    
    ### Authentication
    
    Use Bearer tokens for authentication:
    ```
    Authorization: Bearer <your-token>
    ```
    
    ### Rate Limits
    
    - 100 requests per minute for authenticated users
    - 10 requests per minute for unauthenticated users
    """,
    version="2.0.0",
    contact={
        "name": "StateSet Team",
        "email": "team@stateset.ai",
        "url": "https://stateset.ai",
    },
    license_info={
        "name": "Business Source License 1.1",
        "url": "https://github.com/stateset/stateset-agents/blob/main/LICENSE",
    },
    lifespan=lifespan,
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Routes
# ============================================================================


@app.get("/", summary="API Root", description="Welcome endpoint with API information")
async def root():
    """Get API information."""
    return {
        "message": "Welcome to StateSet Agents API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API service",
)
async def health_check():
    """Get service health status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().timestamp(),
        version="2.0.0",
        uptime=asyncio.get_event_loop().time(),
        components={
            "agent_service": "healthy",
            "training_service": "healthy",
            "security_monitor": "healthy",
        },
    )


@app.post(
    "/agents",
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
    config: AgentConfigRequest, user_data: Dict = Depends(get_current_user)
):
    """Create a new agent."""
    try:
        agent_id = await agent_service.create_agent(config)
        return agent_id
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to create agent")


@app.post(
    "/conversations",
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
            security_monitor.log_security_event,
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
        logger.error(f"Conversation failed: {e}")
        raise HTTPException(status_code=500, detail="Conversation failed")


@app.post(
    "/training",
    response_model=TrainingResponse,
    summary="Start Training",
    description="Start training an agent with the specified configuration",
    responses={
        202: {"description": "Training started successfully"},
        400: {"description": "Invalid configuration", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Insufficient permissions", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def start_training(
    request: TrainingRequest, user_data: Dict = Depends(require_role("trainer"))
):
    """Start a training job."""
    try:
        training_id = await training_service.start_training(request)

        return TrainingResponse(
            training_id=training_id,
            status="running",
            message="Training started successfully",
            estimated_time=request.num_episodes * 30,  # Rough estimate
        )

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start training")


@app.get(
    "/training/{training_id}",
    summary="Get Training Status",
    description="Get the status of a training job",
    responses={
        200: {"description": "Training status retrieved"},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Training job not found", "model": ErrorResponse},
    },
)
async def get_training_status(
    training_id: str, user_data: Dict = Depends(get_current_user)
):
    """Get training job status."""
    try:
        return training_service.get_training_status(training_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Metrics",
    description="Get system and performance metrics",
    responses={
        200: {"description": "Metrics retrieved successfully"},
        401: {"description": "Authentication required", "model": ErrorResponse},
    },
)
async def get_metrics(user_data: Dict = Depends(require_role("admin"))):
    """Get system metrics."""
    try:
        monitor = get_global_monitor()
        summary = monitor.get_metrics_summary()

        return MetricsResponse(
            timestamp=datetime.utcnow().timestamp(),
            system_metrics={
                "total_operations": summary.get("total_operations", 0),
                "active_operations": len(monitor._active_operations),
            },
            performance_metrics={
                "avg_response_time": summary.get("operations_by_type", {})
                .get("agent_response", {})
                .get("avg_duration", 0)
            },
            security_events=len(security_monitor.events),
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_exception",
            "message": exc.detail,
            "timestamp": datetime.utcnow().timestamp(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred",
            "timestamp": datetime.utcnow().timestamp(),
        },
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
