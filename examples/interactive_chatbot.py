#!/usr/bin/env python3
"""
Interactive Chatbot using StateSet Agents API

A simple command-line chatbot that demonstrates real-world usage
of the StateSet Agents API for building conversational applications.

Prerequisites:
    pip install requests python-dotenv rich

Usage:
    python examples/interactive_chatbot.py
"""

import os
import sys
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for enhanced formatting: pip install rich")

# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_JWT_SECRET = os.getenv("API_JWT_SECRET", "test_secret_for_development_only_not_for_prod")


class Chatbot:
    """Interactive chatbot powered by StateSet Agents API."""

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the chatbot.

        Args:
            system_prompt: Optional system prompt to customize the agent's behavior
        """
        self.base_url = API_BASE_URL.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {API_JWT_SECRET}",
            "Content-Type": "application/json"
        })
        self.session.timeout = 30

        self.messages: List[Dict[str, str]] = []
        self.conversation_id: Optional[str] = None
        self.system_prompt = system_prompt or self._get_default_prompt()
        self.console = Console() if RICH_AVAILABLE else None

    def _get_default_prompt(self) -> str:
        """Get default system prompt."""
        return (
            "You are a helpful, friendly AI assistant. "
            "Provide clear, accurate, and engaging responses. "
            "If you don't know something, say so honestly. "
            "Keep responses concise but informative."
        )

    def initialize(self) -> bool:
        """Initialize the chatbot by checking API health."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            health = response.json()

            if RICH_AVAILABLE:
                self.console.print(f"[green]Connected to API (v{health.get('version', 'unknown')})[/green]")
            else:
                print(f"Connected to API (v{health.get('version', 'unknown')})")

            return True
        except requests.RequestException as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Failed to connect to API: {e}[/red]")
            else:
                print(f"Failed to connect to API: {e}")
            return False

    def send_message(self, user_input: str) -> str:
        """
        Send a message and get the agent's response.

        Args:
            user_input: The user's message

        Returns:
            The agent's response
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})

        # Prepare request payload
        payload = {
            "messages": self.messages,
            "temperature": 0.7,
            "max_tokens": 512,
        }

        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id

        # Send request
        try:
            response = self.session.post(
                f"{self.base_url}/conversations",
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            # Extract response
            agent_reply = result["response"]
            self.conversation_id = result["conversation_id"]

            # Add to message history
            self.messages.append({"role": "assistant", "content": agent_reply})

            return agent_reply

        except requests.HTTPError as e:
            error_msg = f"API Error: {e.response.status_code}"
            if e.response.text:
                try:
                    error_data = e.response.json()
                    error_msg += f" - {error_data.get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {e.response.text[:100]}"
            return error_msg
        except requests.RequestException as e:
            return f"Connection error: {e}"

    def run(self):
        """Run the interactive chatbot."""
        if not self.initialize():
            if RICH_AVAILABLE:
                self.console.print("\n[red]Cannot start chatbot. Please ensure the API server is running:[/red]")
                self.console.print("  python -m api.main")
            else:
                print("\nCannot start chatbot. Please ensure the API server is running:")
                print("  python -m api.main")
            sys.exit(1)

        # Welcome message
        if RICH_AVAILABLE:
            welcome = Panel(
                "[bold cyan]StateSet Agents Chatbot[/bold cyan]\n\n"
                "Ask me anything! Type 'quit', 'exit', or 'bye' to end the conversation.\n"
                "Type '/clear' to start a new conversation.\n"
                "Type '/help' for more commands.",
                title="Welcome",
                border_style="cyan"
            )
            self.console.print(welcome)
        else:
            print("=" * 60)
            print("StateSet Agents Chatbot")
            print("=" * 60)
            print("\nAsk me anything! Type 'quit', 'exit', or 'bye' to end.")
            print("Type '/clear' to start a new conversation.")
            print("Type '/help' for more commands.\n")

        # Main conversation loop
        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
                else:
                    user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    if RICH_AVAILABLE:
                        self.console.print("\n[yellow]Goodbye! Have a great day![/yellow]\n")
                    else:
                        print("\nGoodbye! Have a great day!\n")
                    break

                if user_input.lower() == '/clear':
                    self.messages = []
                    self.conversation_id = None
                    if RICH_AVAILABLE:
                        self.console.print("[yellow]Conversation cleared. Starting fresh![/yellow]")
                    else:
                        print("Conversation cleared. Starting fresh!")
                    continue

                if user_input.lower() == '/help':
                    self._show_help()
                    continue

                if user_input.lower() == '/history':
                    self._show_history()
                    continue

                # Send message and get response
                if RICH_AVAILABLE:
                    with self.console.status("[cyan]Thinking...", spinner="dots"):
                        response = self.send_message(user_input)

                    # Display response with markdown rendering
                    self.console.print("\n[bold green]Agent[/bold green]:")
                    self.console.print(Panel(Markdown(response), border_style="green"))
                else:
                    print("\nThinking...")
                    response = self.send_message(user_input)
                    print(f"\nAgent: {response}")

            except KeyboardInterrupt:
                if RICH_AVAILABLE:
                    self.console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]\n")
                else:
                    print("\n\nInterrupted. Goodbye!\n")
                break
            except Exception as e:
                if RICH_AVAILABLE:
                    self.console.print(f"\n[red]Error: {e}[/red]")
                else:
                    print(f"\nError: {e}")

    def _show_help(self):
        """Display help information."""
        if RICH_AVAILABLE:
            help_text = """
**Available Commands:**

- `/help` - Show this help message
- `/clear` - Clear conversation history and start fresh
- `/history` - Show conversation history
- `quit`, `exit`, `bye` - End the conversation

**Tips:**

- Ask questions naturally, just like talking to a person
- The agent remembers the conversation context
- Use `/clear` if you want to change topics completely
            """
            self.console.print(Panel(Markdown(help_text), title="Help", border_style="yellow"))
        else:
            print("\nAvailable Commands:")
            print("  /help    - Show this help message")
            print("  /clear   - Clear conversation history")
            print("  /history - Show conversation history")
            print("  quit     - End the conversation")
            print("\nTips:")
            print("  - Ask questions naturally")
            print("  - The agent remembers context")
            print("  - Use /clear to change topics\n")

    def _show_history(self):
        """Display conversation history."""
        if not self.messages:
            if RICH_AVAILABLE:
                self.console.print("[yellow]No conversation history yet.[/yellow]")
            else:
                print("No conversation history yet.")
            return

        if RICH_AVAILABLE:
            self.console.print("\n[bold]Conversation History:[/bold]\n")
            for msg in self.messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                color = "blue" if msg["role"] == "user" else "green"
                self.console.print(f"[{color}]{role}:[/{color}] {content}\n")
        else:
            print("\nConversation History:")
            for i, msg in enumerate(self.messages, 1):
                print(f"\n{i}. {msg['role'].capitalize()}: {msg['content']}")
            print()


def main():
    """Main entry point."""

    # Parse command-line arguments for custom system prompt
    system_prompt = None
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python interactive_chatbot.py [system_prompt]")
            print("\nExample:")
            print('  python interactive_chatbot.py "You are a Python expert"')
            sys.exit(0)
        else:
            system_prompt = " ".join(sys.argv[1:])

    # Create and run chatbot
    chatbot = Chatbot(system_prompt=system_prompt)
    chatbot.run()


if __name__ == "__main__":
    main()
