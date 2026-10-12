import types
import unittest
from tools.replay_weight_cleanup_schedule_prototype import install

events=[]
torch=types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda:events.append('empty')))
memory_utils=types.SimpleNamespace(cleanup_memory=lambda verbos:events.append('cleanup'))


class Owner:
    def free(self):
        self.H = None
        self.Losses = None
        torch.cuda.empty_cache()
        memory_utils.cleanup_memory(verbos=False)


class ScheduleTests(unittest.TestCase):
    def test_preserve_calls_and_drop_owner_before_compaction(self):
        original=Owner.free
        try:
            for enabled in (False,True):
                Owner.free=original
                events.clear()
                impl=types.SimpleNamespace(GPTQPlus=Owner,torch=torch,memory_utils=memory_utils)
                def prepare(layer,index):
                    events.append('compact')
                    return None if not enabled else (None,{},None)
                impl.__dict__['_replay_compact_prepare']=prepare
                pending=install(impl,enabled=enabled)
                owner=Owner();owner.layer_name='mlp.down_proj';owner.layer_idx=25
                owner.H=object();owner.Losses=object()
                owner.free()
                self.assertIsNone(owner.H);self.assertIsNone(owner.Losses)
                ticket=impl.__dict__['_replay_compact_prepare'](None,25)
                self.assertEqual(events,['compact','empty','cleanup'] if enabled else ['empty','cleanup','compact'])
                self.assertEqual(pending,[])
                if enabled:
                    self.assertEqual(ticket[1]['rescheduled_existing_cleanup_calls'],1)
                    with self.assertRaises(RuntimeError):impl.__dict__['_replay_compact_prepare'](None,26)
        finally:Owner.free=original


if __name__=='__main__':unittest.main()
