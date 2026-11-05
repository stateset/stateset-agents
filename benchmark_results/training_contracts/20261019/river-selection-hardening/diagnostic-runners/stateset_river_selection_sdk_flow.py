import faulthandler
import asyncio
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
original_schedule=test_river_native_sdk.rl.Schedule
def immediate_schedule(**kwargs):
    kwargs.setdefault('window',0.0)
    return original_schedule(**kwargs)
test_river_native_sdk.rl.Schedule=immediate_schedule
suite = unittest.defaultTestLoader.loadTestsFromName('NativeRiverIntegration.test_native_campaign_trains_selects_replays_and_seals_test_publication',test_river_native_sdk)
log_path=Path('/home/dom/stateset-agents/outputs/river-selection-hardening-20261019/sdk-selection-flow-checks.txt')
with log_path.open('w') as log, patch('os.fsync', return_value=None), patch.object(unittest.IsolatedAsyncioTestCase, '_setupAsyncioRunner', lambda self: setattr(self, '_asyncioRunner', asyncio.Runner(debug=False))):
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'fsync_elided':True,'schedule_window':0.0,'scope':'Real River SDK with scripted transport and fsync calls elided; scheduling, selection and recovery logic only, not crash durability or paid remote execution'}
Path('/home/dom/stateset-agents/outputs/river-selection-hardening-20261019/sdk-selection-flow-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
faulthandler.cancel_dump_traceback_later()
raise SystemExit(not result.wasSuccessful())
