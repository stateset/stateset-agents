#!/usr/bin/env python3
"""
StateSet Agents API Client Example

This example demonstrates how to interact with the StateSet Agents API
to create agents, manage conversations, and retrieve metrics.

Prerequisites:
    pip install httpx python-dotenv

Usage:
    # Start the API server first:
    python -m api.main

    # Or using the CLI:
    stateset-agents serve --host 0.0.0.0 --port 8000

    # Then run this example:
    python examples/api_client_example.py
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_JWT_SECRET = os.getenv("API_JWT_SECRET", "test_secret_for_development_only_not_for_prod")


class StateSetAPIClient:
    """Client for interacting with the StateSet Agents API."""

    def __init__(self, base_url: str = API_BASE_URL, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or API_JWT_SECRET
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    # ============================================================================
    # Health & Status
    # ============================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def get_info(self) -> Dict[str, Any]:
        """Get API information."""
        response = await self.client.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        response = await self.client.get(f"{self.base_url}/metrics/summary")
        response.raise_for_status()
        return response.json()

    # ============================================================================
    # Agent Management
    # ============================================================================

    async def create_agent(
        self,
        model_name: str = "gpt2",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new agent.

        Args:
            model_name: Model to use (e.g., "gpt2", "meta-llama/Llama-2-7b-chat-hf")
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            Agent creation response with agent_id
        """
        payload = {
            "model_name": model_name,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "system_prompt": system_prompt,
            **kwargs
        }

        response = await self.client.post(
            f"{self.base_url}/agents",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    async def list_agents(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all agents with pagination.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            status: Optional status filter

        Returns:
            Paginated list of agents
        """
        params = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status

        response = await self.client.get(
            f"{self.base_url}/agents",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific agent.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            Agent details
        """
        response = await self.client.get(f"{self.base_url}/agents/{agent_id}")
        response.raise_for_status()
        return response.json()

    async def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent and all associated conversations.

        Args:
            agent_id: The agent's unique identifier
        """
        response = await self.client.delete(f"{self.base_url}/agents/{agent_id}")
        response.raise_for_status()

    # ============================================================================
    # Conversations
    # ============================================================================

    async def send_message(
        self,
        messages: List[Dict[str, str]],
        agent_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to an agent and get a response.

        Args:
            messages: List of messages in the format [{"role": "user", "content": "..."}]
            agent_id: Optional agent ID (will create default if not provided)
            conversation_id: Optional conversation ID for multi-turn conversations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            context: Additional context for the conversation

        Returns:
            Conversation response with agent's reply
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if conversation_id:
            payload["conversation_id"] = conversation_id

        if context:
            payload["context"] = context

        response = await self.client.post(
            f"{self.base_url}/conversations",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    async def list_conversations(
        self,
        page: int = 1,
        page_size: int = 20,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List conversations with pagination.

        Args:
            page: Page number
            page_size: Items per page
            agent_id: Optional filter by agent ID

        Returns:
            Paginated list of conversations
        """
        params = {"page": page, "page_size": page_size}
        if agent_id:
            params["agent_id"] = agent_id

        response = await self.client.get(
            f"{self.base_url}/conversations",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get full conversation history.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Complete conversation details
        """
        response = await self.client.get(
            f"{self.base_url}/conversations/{conversation_id}"
        )
        response.raise_for_status()
        return response.json()

    async def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation identifier
        """
        response = await self.client.delete(
            f"{self.base_url}/conversations/{conversation_id}"
        )
        response.raise_for_status()


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Demonstrate API usage with various endpoints."""

    print("=" * 70)
    print("StateSet Agents API Client Example")
    print("=" * 70)
    print()

    try:
        async with StateSetAPIClient() as client:

            # Step 1: Check API health
            print("[1/7] Checking API health...")
            health = await client.health_check()
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Version: {health.get('version', 'unknown')}")
            print()

            # Step 2: Get API information
            print("[2/7] Getting API information...")
            info = await client.get_info()
            print(f"   Message: {info.get('message', 'N/A')}")
            print(f"   Available endpoints: {list(info.get('endpoints', {}).keys())}")
            print()

            # Step 3: Create an agent
            print("[3/7] Creating a customer service agent...")
            agent_response = await client.create_agent(
                model_name="gpt2",
                system_prompt="You are a helpful customer service agent for an e-commerce company. "
                             "Be polite, empathetic, and provide clear solutions to customer issues.",
                temperature=0.7,
                max_new_tokens=256
            )
            agent_id = agent_response["agent_id"]
            print(f"   Agent created with ID: {agent_id}")
            print(f"   Created at: {agent_response.get('created_at', 'N/A')}")
            print()

            # Step 4: Have a conversation
            print("[4/7] Starting a customer service conversation...")
            print("-" * 50)

            # First message
            messages = [
                {"role": "user", "content": "Hi, my order #12345 hasn't arrived yet. It's been 2 weeks!"}
            ]

            print(f"User: {messages[0]['content']}")

            conv_response = await client.send_message(
                messages=messages,
                temperature=0.7,
                max_tokens=256
            )

            agent_reply = conv_response["response"]
            conversation_id = conv_response["conversation_id"]

            print(f"Agent: {agent_reply}")
            print(f"   Tokens used: {conv_response['tokens_used']}")
            print(f"   Processing time: {conv_response['processing_time']:.2f}s")
            print()

            # Continue the conversation
            messages.append({"role": "assistant", "content": agent_reply})
            messages.append({
                "role": "user",
                "content": "I need a refund. This is unacceptable!"
            })

            print(f"User: {messages[-1]['content']}")

            conv_response = await client.send_message(
                messages=messages,
                conversation_id=conversation_id,
                temperature=0.7,
                max_tokens=256
            )

            agent_reply = conv_response["response"]
            print(f"Agent: {agent_reply}")
            print()

            print("-" * 50)
            print()

            # Step 5: List all agents
            print("[5/7] Listing all agents...")
            agents = await client.list_agents(page=1, page_size=10)
            print(f"   Total agents: {agents.get('total', 0)}")
            print(f"   Current page: {agents.get('page', 1)}/{agents.get('total', 0) // agents.get('page_size', 20) + 1}")

            if agents.get("items"):
                for agent in agents["items"][:3]:  # Show first 3
                    print(f"   - Agent ID: {agent['agent_id']}")
                    print(f"     Model: {agent['model_name']}")
                    print(f"     Conversations: {agent['conversation_count']}")
            print()

            # Step 6: List conversations
            print("[6/7] Listing conversations...")
            conversations = await client.list_conversations(page=1, page_size=10)
            print(f"   Total conversations: {conversations.get('total', 0)}")

            if conversations.get("items"):
                for conv in conversations["items"][:3]:  # Show first 3
                    print(f"   - Conversation ID: {conv['conversation_id']}")
                    print(f"     Messages: {conv['message_count']}")
                    print(f"     Tokens: {conv['total_tokens']}")
            print()

            # Step 7: Get system metrics
            print("[7/7] Retrieving system metrics...")
            try:
                metrics = await client.get_metrics()
                print(f"   Timestamp: {metrics.get('timestamp', 'N/A')}")

                if "system" in metrics:
                    print(f"   System metrics: {list(metrics['system'].keys())[:5]}")
                if "api" in metrics:
                    print(f"   API metrics: {list(metrics['api'].keys())[:5]}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print("   Metrics endpoint not available")
                else:
                    raise
            print()

            print("=" * 70)
            print("Example completed successfully!")
            print()
            print("Next steps:")
            print("  - Explore the API documentation at http://localhost:8000/docs")
            print("  - Try the training endpoints in /training")
            print("  - Check out streaming conversations")
            print("  - Monitor system metrics in real-time")
            print("=" * 70)

    except httpx.ConnectError:
        print("\nError: Could not connect to the API server.")
        print("Please ensure the API is running:")
        print("  python -m api.main")
        print("  OR")
        print("  stateset-agents serve --host 0.0.0.0 --port 8000")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"\nHTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
