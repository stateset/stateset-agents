"""
GRPO Agent Framework

A comprehensive framework for training multi-turn AI agents using 
Group Relative Policy Optimization (GRPO).
"""

__version__ = "0.7.1"
__author__ = "GRPO Framework Team"
__email__ = "team@stateset.ai"

# Re-export everything from the stateset_agents package for backward compatibility
try:
    from stateset_agents import *

    __all__ = getattr(__import__("stateset_agents"), "__all__", [])
except ImportError:
    __all__ = []
