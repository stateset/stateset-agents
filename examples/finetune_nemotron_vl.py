"""Fine-tune Super VL or Bolt using a StateSet JSON config.

    python examples/finetune_nemotron_vl.py --config bolt.json
    python examples/finetune_nemotron_vl.py --config bolt.json --execute

See docs/NEMOTRON_3_5_VL.md for datasets, environment, and distributed launch.
"""

from stateset_agents.training.nemotron_vl import main

if __name__ == "__main__":
    main()
