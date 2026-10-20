import faulthandler
import sys
import unittest
sys.path.insert(0, '/home/dom/stateset-agents/tests/integration')
sys.path.insert(0, '/home/dom/stateset-agents')
faulthandler.enable()
faulthandler.dump_traceback_later(25, repeat=True)
suite = unittest.defaultTestLoader.loadTestsFromName('test_river_native_sdk.NativeRiverIntegration.test_native_campaign_trains_selects_replays_and_seals_test_publication')
result = unittest.TextTestRunner(verbosity=2).run(suite)
faulthandler.cancel_dump_traceback_later()
raise SystemExit(not result.wasSuccessful())
