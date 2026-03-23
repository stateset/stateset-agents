StateSet Agents Documentation
=============================

Welcome to the official documentation for **StateSet Agents**, a comprehensive framework for training multi-turn AI agents using reinforcement learning techniques.

🚀 **What is StateSet Agents?**

StateSet Agents is a production-ready framework that enables you to:

- 🤖 **Create and train AI agents** using state-of-the-art reinforcement learning
- 💬 **Build conversational AI** with multi-turn dialogue capabilities
- 🎯 **Implement custom reward functions** for domain-specific training
- 📊 **Monitor and analyze** agent performance in real-time
- 🔒 **Deploy securely** with built-in authentication and authorization
- ☁️ **Scale horizontally** with distributed training support

📖 **Quick Start**

.. code-block:: python

   from stateset_agents import MultiTurnAgent, AgentConfig, train

   # Create an agent
   config = AgentConfig(model_name="gpt2", temperature=0.7)
   agent = MultiTurnAgent(config)

   # Train the agent
   trained_agent = await train(
       agent=agent,
       environment=ConversationEnvironment(scenarios=your_scenarios),
       num_episodes=1000
   )

📚 **Contents**

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   qwen3_5_starter
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   core-concepts
   agents
   environments
   rewards
   training
   monitoring
   security

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/agent
   api/environment
   api/reward
   api/training
   api/monitoring
   api/security
   api/enhanced-api

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   distributed-training
   custom-rewards
   performance-optimization
   deployment
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   architecture
   testing
   changelog
   license

🔗 **Links**

- 📦 `PyPI <https://pypi.org/project/stateset-agents/>`_
- 📚 `GitHub <https://github.com/stateset/stateset-agents>`_
- 🐛 `Issue Tracker <https://github.com/stateset/stateset-agents/issues>`_
- 💬 `Discussions <https://github.com/stateset/stateset-agents/discussions>`_
- 📧 `Security <SECURITY.md>`_

📞 **Support**

If you need help or have questions:

- 📖 Check the :doc:`troubleshooting` guide
- 💬 Join our `Discord community <https://discord.gg/stateset>`_
- 🐛 `Open an issue <https://github.com/stateset/stateset-agents/issues>`_
- 📧 Contact: team@stateset.ai

🤝 **Contributing**

We welcome contributions! See our :doc:`contributing` guide for details on:

- Setting up a development environment
- Coding standards and guidelines
- Submitting pull requests
- Reporting bugs and requesting features

📄 **License**

This project is licensed under the Business Source License 1.1 - see the :doc:`license` file for details.

**© 2024 StateSet. All rights reserved.**
