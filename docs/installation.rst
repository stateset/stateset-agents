Installation
============

Core install (lightweight):

.. code-block:: bash

   pip install stateset-agents

Common extras:

.. code-block:: bash

   pip install "stateset-agents[api]"        # FastAPI gateway
   pip install "stateset-agents[training]"   # torch/transformers training extras

From source:

.. code-block:: bash

   git clone https://github.com/stateset/stateset-agents.git
   cd stateset-agents
   pip install -e ".[dev]"

