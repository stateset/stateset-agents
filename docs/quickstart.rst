Quick Start Guide
=================

This guide will get you up and running with StateSet Agents in just a few minutes.

🎯 **What You'll Learn**

- How to install StateSet Agents
- Create your first AI agent
- Train the agent using reinforcement learning
- Have conversations with your trained agent

📦 **Installation**

**Option 1: Install from PyPI (Recommended)**

.. code-block:: bash

   pip install stateset-agents

**Option 2: Install from source**

.. code-block:: bash

   git clone https://github.com/stateset/stateset-agents.git
   cd stateset-agents
   pip install -e ".[dev]"

**Option 3: Using Docker**

.. code-block:: bash

   docker run -p 8000:8000 stateset/agents:latest

🔧 **Verify Installation**

.. code-block:: python

   import stateset_agents
   print(stateset_agents.__version__)

🤖 **Creating Your First Agent**

Let's create a simple conversational agent:

.. code-block:: python

   from stateset_agents import MultiTurnAgent, AgentConfig

   # Configure the agent
   config = AgentConfig(
       model_name="gpt2",           # Use GPT-2 model
       temperature=0.7,             # Sampling temperature
       max_new_tokens=256,          # Maximum response length
       system_prompt="You are a helpful AI assistant."
   )

   # Create the agent
   agent = MultiTurnAgent(config)

   # Initialize the agent (loads the model)
   await agent.initialize()

💬 **Having a Conversation**

Now let's have a conversation with our agent:

.. code-block:: python

   # Prepare conversation messages
   messages = [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Hello! Can you help me learn Python?"}
   ]

   # Get response from agent
   response = await agent.generate_response(messages)
   print(f"Agent: {response}")

🎓 **Training Your Agent**

To make your agent better at specific tasks, you can train it using reinforcement learning:

.. code-block:: python

   from stateset_agents import (
       ConversationEnvironment, 
       CompositeReward, 
       HelpfulnessReward, 
       SafetyReward,
       train
   )

   # Create training scenarios
   scenarios = [
       {
           "id": "learning_python",
           "topic": "education",
           "user_responses": [
               "Hi, I want to learn Python. Where should I start?",
               "That sounds good. What about practical projects?",
               "Great suggestions! How long will it take?",
               "Thank you for all the helpful advice!"
           ]
       }
   ]

   # Create environment
   environment = ConversationEnvironment(scenarios=scenarios)

   # Create reward function
   reward_fn = CompositeReward([
       HelpfulnessReward(weight=0.7),    # Reward helpful responses
       SafetyReward(weight=0.3)           # Reward safe responses
   ])

   # Train the agent
   trained_agent = await train(
       agent=agent,
       environment=environment,
       reward_fn=reward_fn,
       num_episodes=50,          # Number of training episodes
       profile="balanced"        # Training profile
   )

   print("Training completed! 🎉")

📊 **Monitoring Training Progress**

Monitor your training progress with built-in monitoring:

.. code-block:: python

   from stateset_agents import init_wandb

   # Initialize Weights & Biases monitoring
   init_wandb(
       project="my-agent-training",
       name="python-tutor-agent"
   )

   # Training progress will be automatically logged

🔒 **Using the REST API**

Start the REST API server:

.. code-block:: bash

   # Using Python
   python -m stateset_agents.api.enhanced_api_service

   # Using Docker
   docker-compose -f deployment/docker/docker-compose.dev.yml up stateset-agents-dev

The API will be available at http://localhost:8000

**API Documentation**: Visit http://localhost:8000/docs for interactive API documentation.

🧪 **Testing Your Agent**

Test your agent with different scenarios:

.. code-block:: python

   test_scenarios = [
       "How do I install Python?",
       "What are the best Python libraries for data science?",
       "Can you help me debug this Python error?",
       "What's the difference between lists and tuples?"
   ]

   for question in test_scenarios:
       messages = [{"role": "user", "content": question}]
       response = await agent.generate_response(messages)
       print(f"Q: {question}")
       print(f"A: {response}")
       print("-" * 50)

🚀 **Next Steps**

Now that you have a working agent, here are some next steps:

1. **Customize Reward Functions**: Create domain-specific reward functions
2. **Scale Training**: Use distributed training for larger models
3. **Deploy to Production**: Use Docker and Kubernetes for deployment
4. **Add Monitoring**: Set up comprehensive monitoring and alerting
5. **Experiment**: Try different models, training configurations, and scenarios

📚 **Learn More**

- :doc:`examples` - Complete examples and use cases
- :doc:`training` - Advanced training techniques
- :doc:`deployment` - Production deployment guides
- :doc:`api/enhanced-api` - Complete API reference

🆘 **Getting Help**

If you run into issues:

1. Check the :doc:`troubleshooting` guide
2. Search existing `issues <https://github.com/stateset/stateset-agents/issues>`_
3. Ask questions in our `Discord community <https://discord.gg/stateset>`_
4. `Open a new issue <https://github.com/stateset/stateset-agents/issues/new>`_

Happy building! 🚀
