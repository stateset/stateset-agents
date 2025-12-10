#!/usr/bin/env python3
"""
StateSet Agents API - Simple Synchronous Client Example

A simplified, synchronous version of the API client for quick prototyping
and integration into existing synchronous codebases.

Prerequisites:
    pip install requests python-dotenv

Usage:
    # Start the API server first
    python -m api.main

    # Then run this example
    python examples/api_client_simple.py
"""

import os
import sys
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_JWT_SECRET = os.getenv("API_JWT_SECRET", "test_secret_for_development_only_not_for_prod")


class SimpleAPIClient:
    """Simple synchronous client for StateSet Agents API."""

    def __init__(self, base_url: str = API_BASE_URL, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key or API_JWT_SECRET}",
            "Content-Type": "application/json"
        })
        self.session.timeout = 30

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def create_agent(
        self,
        model_name: str = "gpt2",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """Create a new agent."""
        payload = {
            "model_name": model_name,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        response = self.session.post(f"{self.base_url}/agents", json=payload)
        response.raise_for_status()
        return response.json()

    def list_agents(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """List all agents."""
        params = {"page": page, "page_size": page_size}
        response = self.session.get(f"{self.base_url}/agents", params=params)
        response.raise_for_status()
        return response.json()

    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get agent details."""
        response = self.session.get(f"{self.base_url}/agents/{agent_id}")
        response.raise_for_status()
        return response.json()

    def send_message(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Send a message and get a response."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        response = self.session.post(f"{self.base_url}/conversations", json=payload)
        response.raise_for_status()
        return response.json()

    def list_conversations(
        self,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """List conversations."""
        params = {"page": page, "page_size": page_size}
        response = self.session.get(f"{self.base_url}/conversations", params=params)
        response.raise_for_status()
        return response.json()

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history."""
        response = self.session.get(f"{self.base_url}/conversations/{conversation_id}")
        response.raise_for_status()
        return response.json()


def main():
    """Simple example demonstrating key API operations."""

    print("=" * 60)
    print("StateSet Agents API - Simple Client Example")
    print("=" * 60)
    print()

    try:
        client = SimpleAPIClient()

        # Check health
        print("[1/4] Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print()

        # Create agent
        print("[2/4] Creating an agent...")
        agent = client.create_agent(
            model_name="gpt2",
            system_prompt="You are a helpful AI assistant specialized in customer support.",
            temperature=0.7,
            max_new_tokens=256
        )
        agent_id = agent["agent_id"]
        print(f"   Agent ID: {agent_id}")
        print()

        # Have a conversation
        print("[3/4] Having a conversation...")
        print("-" * 40)

        # First turn
        messages = [
            {"role": "user", "content": "What is reinforcement learning?"}
        ]
        print(f"User: {messages[0]['content']}")

        response = client.send_message(messages, temperature=0.7)
        agent_reply = response["response"]
        conversation_id = response["conversation_id"]

        print(f"Agent: {agent_reply}")
        print(f"   Tokens: {response['tokens_used']}")
        print()

        # Second turn
        messages.append({"role": "assistant", "content": agent_reply})
        messages.append({"role": "user", "content": "Can you give me a simple example?"})

        print(f"User: {messages[-1]['content']}")

        response = client.send_message(
            messages,
            conversation_id=conversation_id,
            temperature=0.7
        )
        agent_reply = response["response"]

        print(f"Agent: {agent_reply}")
        print()
        print("-" * 40)
        print()

        # List conversations
        print("[4/4] Listing conversations...")
        conversations = client.list_conversations()
        print(f"   Total: {conversations['total']}")
        if conversations.get("items"):
            for conv in conversations["items"][:3]:
                print(f"   - {conv['conversation_id']}: {conv['message_count']} messages")
        print()

        print("=" * 60)
        print("Example completed successfully!")
        print()
        print("Try exploring more features:")
        print("  - Training agents: POST /training/jobs")
        print("  - System metrics: GET /metrics/summary")
        print("  - API documentation: http://localhost:8000/docs")
        print("=" * 60)

    except requests.ConnectionError:
        print("\nError: Cannot connect to API server.")
        print("Start the server with: python -m api.main")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"\nHTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
