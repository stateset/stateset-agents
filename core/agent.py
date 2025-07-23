"""
Agent base classes for multi-turn RL training

This module defines the core agent interfaces and implementations
for training conversational AI agents using GRPO.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Callable
import asyncio
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None
    logging.warning("PEFT not available. Install with: pip install peft")

from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agent behavior - Compatible with HuggingFace patterns"""
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    system_prompt: Optional[str] = None
    use_chat_template: bool = True
    
    # HuggingFace model configuration
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    attn_implementation: Optional[str] = "flash_attention_2"
    device_map: Optional[str] = "auto"
    use_peft: bool = False
    peft_config: Optional[Dict[str, Any]] = None


class StopOnSpecialTokens(StoppingCriteria):
    """Custom stopping criteria for conversation agents"""
    
    def __init__(self, stop_tokens: List[str], tokenizer):
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        self.stop_token_ids = [
            tokenizer.encode(token, add_special_tokens=False)[0] 
            for token in stop_tokens
        ]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any stop token was generated
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_token_ids


class Agent(ABC):
    """
    Abstract base class for all agents
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.conversation_history = []
        
    @abstractmethod
    async def initialize(self):
        """Initialize the agent (load models, etc.)"""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a response given conversation history"""
        pass
    
    async def reset(self):
        """Reset agent state for new conversation"""
        self.conversation_history = []
    
    def add_to_history(self, turn: ConversationTurn):
        """Add a turn to conversation history"""
        self.conversation_history.append(turn)
    
    def get_history(self) -> List[ConversationTurn]:
        """Get current conversation history"""
        return self.conversation_history.copy()


class MultiTurnAgent(Agent):
    """
    Agent designed for multi-turn conversations
    """
    
    def __init__(
        self,
        config: AgentConfig,
        memory_window: int = 10,
        context_compression: bool = False
    ):
        super().__init__(config)
        self.memory_window = memory_window
        self.context_compression = context_compression
        self.turn_count = 0
        
    async def initialize(self):
        """Initialize the multi-turn agent"""
        logger.info(f"Initializing MultiTurnAgent with model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with proper configuration
        model_kwargs = {
            "trust_remote_code": True
        }
        
        # Apply torch dtype
        if self.config.torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.torch_dtype == "float32":
            model_kwargs["torch_dtype"] = torch.float32
        
        # Apply device mapping
        if self.config.device_map:
            model_kwargs["device_map"] = self.config.device_map
        
        # Apply attention implementation
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Apply PEFT if configured
        if self.config.use_peft and self.config.peft_config and LoraConfig:
            lora_config = LoraConfig(**self.config.peft_config)
            self.model = get_peft_model(self.model, lora_config)
            logger.info("PEFT/LoRA applied to model")
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info("MultiTurnAgent initialized successfully")
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response for multi-turn conversation"""
        
        # Apply memory window
        recent_messages = self._apply_memory_window(messages)
        
        # Add system prompt if configured
        if self.config.system_prompt and (not recent_messages or recent_messages[0]["role"] != "system"):
            recent_messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        
        # Apply chat template if available
        if (self.config.use_chat_template and 
            hasattr(self.tokenizer, 'apply_chat_template') and 
            self.tokenizer.chat_template is not None):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    recent_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback to manual formatting if chat template fails
                prompt = self._format_conversation(recent_messages)
        else:
            prompt = self._format_conversation(recent_messages)
        
        # Generate response
        response = await self._generate_with_model(prompt, context)
        
        self.turn_count += 1
        return response
    
    def _apply_memory_window(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply memory window to conversation history"""
        if self.memory_window <= 0:
            return messages
        
        # Keep system message and last N turns
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        other_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # Keep recent messages within window
        recent_messages = other_messages[-self.memory_window:]
        
        return system_messages + recent_messages
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation for models without chat template"""
        formatted = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"System: {content}\n"
            elif role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        
        formatted += "Assistant: "
        return formatted
    
    async def _generate_with_model(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using the language model"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length - self.config.max_new_tokens
        ).to(self.model.device)
        
        # Setup stopping criteria
        stop_tokens = ["User:", "System:", "\n\n"]
        stopping_criteria = StoppingCriteriaList([
            StopOnSpecialTokens(stop_tokens, self.tokenizer)
        ])
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        response_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response"""
        # Remove common artifacts
        response = response.strip()
        
        # Remove stopping tokens that might have leaked through
        stop_phrases = ["User:", "System:", "Human:", "AI:"]
        for phrase in stop_phrases:
            if phrase in response:
                response = response.split(phrase)[0].strip()
        
        return response
    
    async def process_turn(
        self,
        conversation_history: List[Dict[str, str]],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Process a single conversation turn"""
        
        # Add user input to history
        messages = conversation_history + [{"role": "user", "content": user_input}]
        
        # Generate response
        response = await self.generate_response(messages, context)
        
        # Create conversation turn
        turn = ConversationTurn(
            role="assistant",
            content=response,
            metadata={
                "turn_count": self.turn_count,
                "model_name": self.config.model_name,
                "context": context
            }
        )
        
        return turn


class ToolAgent(MultiTurnAgent):
    """
    Agent that can use tools and function calls
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.tools = tools or []
        self.tool_registry = {tool["name"]: tool for tool in self.tools}
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response with potential tool usage"""
        
        # Check if we need to use tools
        if self._should_use_tools(messages, context):
            return await self._generate_with_tools(messages, context)
        else:
            return await super().generate_response(messages, context)
    
    def _should_use_tools(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if tools should be used for this response"""
        if not self.tools:
            return False
        
        # Simple heuristic: check for tool-related keywords
        last_message = messages[-1]["content"].lower() if messages else ""
        tool_keywords = ["calculate", "search", "look up", "find", "analyze"]
        
        return any(keyword in last_message for keyword in tool_keywords)
    
    async def _generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response that may include tool calls"""
        
        # Add tool descriptions to context
        tool_context = self._format_tool_descriptions()
        enhanced_messages = messages + [
            {"role": "system", "content": f"Available tools: {tool_context}"}
        ]
        
        # Generate response
        response = await super().generate_response(enhanced_messages, context)
        
        # Parse and execute tool calls if present
        response = await self._process_tool_calls(response, context)
        
        return response
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for model context"""
        descriptions = []
        for tool in self.tools:
            desc = f"{tool['name']}: {tool.get('description', 'No description')}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    async def _process_tool_calls(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process any tool calls in the response"""
        # Simple tool call parsing (in practice, this would be more sophisticated)
        if "TOOL_CALL:" in response:
            parts = response.split("TOOL_CALL:")
            if len(parts) > 1:
                tool_call = parts[1].strip().split()[0]
                if tool_call in self.tool_registry:
                    tool_result = await self._execute_tool(tool_call, context)
                    response = parts[0] + f"Tool result: {tool_result}"
        
        return response
    
    async def _execute_tool(
        self,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a tool call"""
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return f"Error: Tool {tool_name} not found"
        
        try:
            # Execute tool function
            if "function" in tool:
                result = await tool["function"](context)
                return str(result)
            else:
                return f"Tool {tool_name} executed (mock result)"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def add_tool(self, tool: Dict[str, Any]):
        """Add a tool to the agent"""
        self.tools.append(tool)
        self.tool_registry[tool["name"]] = tool


# Factory functions
def create_agent(
    agent_type: str,
    model_name: str,
    **kwargs
) -> Agent:
    """Factory function for creating agents"""
    
    config = AgentConfig(model_name=model_name, **kwargs)
    
    if agent_type == "multi_turn":
        return MultiTurnAgent(config)
    elif agent_type == "tool":
        return ToolAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def load_agent_from_checkpoint(
    checkpoint_path: str,
    agent_class: type = MultiTurnAgent
) -> Agent:
    """Load an agent from a saved checkpoint"""
    
    import pickle
    import os
    
    # Load config
    config_path = os.path.join(checkpoint_path, "agent_config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = AgentConfig(**config_dict)
    
    # Create agent
    agent = agent_class(config)
    await agent.initialize()
    
    # Load any additional state
    state_path = os.path.join(checkpoint_path, "agent_state.pkl")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            agent.__dict__.update(state)
    
    return agent


async def save_agent_checkpoint(
    agent: Agent,
    checkpoint_path: str
):
    """Save agent to checkpoint"""
    
    import pickle
    import os
    
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save config
    config_path = os.path.join(checkpoint_path, "agent_config.json")
    with open(config_path, "w") as f:
        json.dump(agent.config.__dict__, f, indent=2)
    
    # Save additional state (excluding model/tokenizer)
    state = {k: v for k, v in agent.__dict__.items() 
             if k not in ["model", "tokenizer", "generation_config"]}
    
    state_path = os.path.join(checkpoint_path, "agent_state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    logger.info(f"Agent checkpoint saved to {checkpoint_path}")


# Pre-defined agent configurations
AGENT_CONFIGS = {
    "helpful_assistant": {
        "system_prompt": "You are a helpful, harmless, and honest AI assistant. Provide clear, accurate, and helpful responses.",
        "temperature": 0.7,
        "max_new_tokens": 512
    },
    
    "customer_service": {
        "system_prompt": "You are a professional customer service representative. Be polite, helpful, and solution-oriented.",
        "temperature": 0.6,
        "max_new_tokens": 256
    },
    
    "tutor": {
        "system_prompt": "You are a patient and encouraging tutor. Break down complex topics and guide students step-by-step.",
        "temperature": 0.8,
        "max_new_tokens": 512
    },
    
    "creative_writer": {
        "system_prompt": "You are a creative writing assistant. Help with storytelling, character development, and creative expression.",
        "temperature": 0.9,
        "max_new_tokens": 1024
    }
}


# Helper functions for agent creation
def create_agent(
    agent_type: str = "multi_turn",
    model_name: str = "gpt2",
    **kwargs
) -> Union[Agent, MultiTurnAgent]:
    """Create an agent of specified type"""
    config = AgentConfig(model_name=model_name, **kwargs)
    
    if agent_type == "multi_turn":
        return MultiTurnAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_peft_agent(
    model_name: str,
    peft_config: Dict[str, Any],
    **kwargs
) -> MultiTurnAgent:
    """Create a PEFT-enabled agent with LoRA"""
    if LoraConfig is None:
        raise ImportError("PEFT library not available. Install with: pip install peft")
    
    config = AgentConfig(
        model_name=model_name,
        use_peft=True,
        peft_config=peft_config,
        **kwargs
    )
    
    return MultiTurnAgent(config)


async def save_agent_checkpoint(
    agent: Agent,
    checkpoint_path: str,
    save_model: bool = True
) -> None:
    """Save agent checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if save_model and agent.model:
        agent.model.save_pretrained(checkpoint_path)
        agent.tokenizer.save_pretrained(checkpoint_path)
    
    # Save agent config
    config_path = checkpoint_path / "agent_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(agent.config), f, indent=2, default=str)


async def load_agent_from_checkpoint(
    checkpoint_path: str,
    load_model: bool = True
) -> MultiTurnAgent:
    """Load agent from checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    # Load config
    config_path = checkpoint_path / "agent_config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = AgentConfig(**config_dict)
    agent = MultiTurnAgent(config)
    
    if load_model:
        await agent.initialize()
    
    return agent


def get_preset_config(preset_name: str, **overrides) -> AgentConfig:
    """Get a preset agent configuration"""
    if preset_name not in AGENT_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(AGENT_CONFIGS.keys())}")
    
    preset = AGENT_CONFIGS[preset_name].copy()
    preset.update(overrides)
    
    return AgentConfig(**preset)