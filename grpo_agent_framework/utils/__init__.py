# Proxy subpackage mapping to top-level utils package
from importlib import import_module
import sys as _sys

_utils = import_module('utils')

for _name in dir(_utils):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_utils, _name)

# Map known submodules if present
for _sub in ('wandb_integration', 'observability', 'monitoring', 'alerts', 'logging', 'cache'):
    try:
        _sys.modules[__name__ + f'.{_sub}'] = import_module(f'utils.{_sub}')
    except Exception:
        pass