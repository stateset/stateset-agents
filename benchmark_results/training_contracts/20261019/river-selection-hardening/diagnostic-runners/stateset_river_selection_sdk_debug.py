import asyncio,inspect,sys,unittest
from unittest.mock import patch
sys.path[:0]=['/home/dom/stateset-agents/tests/integration','/home/dom/stateset-agents']
import test_river_native_sdk as m
name='test_native_campaign_drains_cancelled_validation_write'
original=getattr(m.NativeRiverIntegration,name)
async def traced(self):
    async def dump():
        await asyncio.sleep(8)
        for task in asyncio.all_tasks():
            print('TASK',task,flush=True)
            coro=task.get_coro()
            for _ in range(25):
                if coro is None:break
                frame=getattr(coro,'cr_frame',None) or getattr(coro,'ag_frame',None)
                if frame is not None:print(frame.f_code.co_filename,frame.f_lineno,frame.f_code.co_name,flush=True)
                coro=getattr(coro,'cr_await',None) or getattr(coro,'ag_await',None)
    watcher=asyncio.create_task(dump())
    try:await original(self)
    finally:watcher.cancel()
setattr(m.NativeRiverIntegration,name,traced)
with patch('os.fsync',return_value=None):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('NativeRiverIntegration.'+name,m))
raise SystemExit(not result.wasSuccessful())
