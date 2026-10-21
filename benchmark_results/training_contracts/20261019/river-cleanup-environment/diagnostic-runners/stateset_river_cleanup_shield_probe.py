import asyncio,faulthandler,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=['/home/dom/stateset-agents/tests/integration','/home/dom/stateset-agents']
log=Path('/home/dom/stateset-agents/outputs/river-cleanup-review-20261019/shield-probe.txt').open('w',buffering=1)
sys.stdout=log;sys.stderr=log
faulthandler.enable(file=log)
faulthandler.dump_traceback_later(60,repeat=True,file=log)
import test_river_native_sdk as m
from river_client.rl.checkpoint import StateStore
original_save=StateStore.save
def save(self,*args,**kwargs):
    started=time.monotonic();print('SAVE START',started,self.path,flush=True)
    try:return original_save(self,*args,**kwargs)
    finally:print('SAVE END',time.monotonic()-started,flush=True)
StateStore.save=save
original_shield=asyncio.shield
def shield(arg):
    outer=original_shield(arg)
    print('SHIELD CREATED',id(arg),id(outer),'inner',arg,'outer',outer,flush=True)
    def done(inner):
        print('SHIELD INNER DONE',id(inner),id(outer),'outer',outer,flush=True)
    arg.add_done_callback(done)
    return outer
asyncio.shield=shield
original_exit=m.rl.RolloutEngine.__aexit__
original_dispatch=m.rl.RolloutEngine._dispatch
async def dispatch(self):
    try:return await original_dispatch(self)
    except BaseException as exc:
        print('DISPATCH EXIT',id(self),type(exc).__name__,flush=True)
        raise
async def exit(self,*args):
    print('ENGINE EXIT START',id(self),'model',type(self.model).__name__,'groups',len(self._groups),'dispatcher',self._dispatcher,flush=True)
    try:return await original_exit(self,*args)
    finally:print('ENGINE EXIT FINISH',id(self),flush=True)
m.rl.RolloutEngine.__aexit__=exit
m.rl.RolloutEngine._dispatch=dispatch
async def dump():
    await asyncio.sleep(12)
    for task in asyncio.all_tasks():
        print('TASK',task,flush=True)
        for frame in task.get_stack():
            print('FRAME',frame.f_code.co_filename,frame.f_lineno,frame.f_code.co_name,flush=True)
            for name in ('generator','iterator','self'):
                obj=frame.f_locals.get(name)
                if hasattr(obj,'ag_frame') and obj.ag_frame:
                    print('GENERATOR',name,obj.ag_frame.f_code.co_filename,obj.ag_frame.f_lineno,flush=True)
                    awaited=obj.ag_await
                    for _ in range(12):
                        if awaited is None:break
                        frame=getattr(awaited,'cr_frame',None)
                        if frame is not None:
                            print('AWAIT',frame.f_code.co_filename,frame.f_lineno,frame.f_code.co_name,flush=True)
                            if 'write' in frame.f_locals:print('WRITE TASK',frame.f_locals['write'],flush=True)
                        awaited=getattr(awaited,'cr_await',None)
                if name=='self' and hasattr(obj,'_dispatcher'):
                    print('ENGINE',id(obj),'dispatcher',obj._dispatcher,'groups',len(obj._groups),flush=True)
async def traced(self):
    watcher=asyncio.create_task(dump())
    try:await original(self)
    finally:watcher.cancel()
name=sys.argv[1] if len(sys.argv)>1 else 'test_real_evaluator_retains_complete_validation_case_evidence'
original=getattr(m.NativeRiverIntegration,name)
setattr(m.NativeRiverIntegration,name,traced)
print('IMPORT COMPLETE',time.monotonic(),flush=True)
with patch('os.fsync', return_value=None):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('NativeRiverIntegration.'+name,m))
print('RESULT',result.wasSuccessful(),flush=True)
faulthandler.cancel_dump_traceback_later()
raise SystemExit(not result.wasSuccessful())
