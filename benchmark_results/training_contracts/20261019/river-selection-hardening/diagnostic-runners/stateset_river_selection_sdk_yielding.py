import faulthandler
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, '/home/dom/stateset-agents/tests/integration')
sys.path.insert(0, '/home/dom/stateset-agents')
faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=True)
import test_river_native_sdk
original_sample=test_river_native_sdk.ScriptedModel.sample
async def yielding_sample(self, **kwargs):
    import asyncio
    await asyncio.sleep(0)
    return await original_sample(self,**kwargs)
test_river_native_sdk.ScriptedModel.sample=yielding_sample
suite = unittest.defaultTestLoader.loadTestsFromModule(test_river_native_sdk)
log_path=Path('/home/dom/stateset-agents/outputs/river-selection-hardening-20261019/sdk-yielding-checks.txt')
with log_path.open('w') as log, patch('os.fsync', return_value=None):
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'fsync_elided':True,'scripted_transport_yields':True,'scope':'Real River SDK with scripted transport and fsync calls elided; scheduling, selection and recovery logic only, not crash durability or paid remote execution'}
Path('/home/dom/stateset-agents/outputs/river-selection-hardening-20261019/sdk-yielding-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
faulthandler.cancel_dump_traceback_later()
raise SystemExit(not result.wasSuccessful())
