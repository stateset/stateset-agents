# Proxy subpackage mapping to top-level core package with lightweight imports
from importlib import import_module
import sys as _sys

# Map lightweight core submodules explicitly
for _sub in ('trajectory', 'reward', 'environment'):
    try:
        _sys.modules[__name__ + f'.{_sub}'] = import_module(f'core.{_sub}')
    except Exception:
        pass

# Provide names by attempting to import the top-level only after mapping
try:
    _core = import_module('core')
    for _name in dir(_core):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_core, _name)
except Exception:
    # Fallback: expose minimal symbols that tests commonly import
    try:
        from core.environment import Environment
        from core.trajectory import Trajectory, MultiTurnTrajectory, ConversationTurn
        from core.reward import RewardFunction, CompositeReward
        globals().update({
            'Environment': Environment,
            'Trajectory': Trajectory,
            'MultiTurnTrajectory': MultiTurnTrajectory,
            'ConversationTurn': ConversationTurn,
            'RewardFunction': RewardFunction,
            'CompositeReward': CompositeReward,
        })
    except Exception:
        pass