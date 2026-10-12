import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from tools.timing_completed_prepared_disk import link_completed


class CompletedTimingDiskTests(unittest.TestCase):
    def fixture(self, root):
        ref, target = root/'reference', root/'timed'
        for p in (ref, target):
            folder=p/'static_cache/_prepared_checkpoints/same-id';folder.mkdir(parents=True)
            (folder/'model.safetensors').write_bytes(b'exact prepared model')
        (ref/'status.json').write_text(json.dumps(dict(valid=True,scope='full_pipeline',stage0=dict(returncode=0),
            stage2=dict(returncode=0),metrics=dict(state_sha256='exact'),kl=.3,ppl=25)))
        (target/'status.json').write_text(json.dumps(dict(numerical_reference_validated=True,
            fresh_static_cache_exact_to_reference=True,metrics_by_rank=[dict(state_tensors=507,state_sha256='exact')],
            quality_by_rank=[[dict(kl=.3,ppl=25)]])))
        extra=dict.fromkeys(['valid','all28_outputs_exact','method_records_exact','post_quantization_cache_exact',
                             'outside_speed_timers'],True)
        (target/'independent_complete_method_and_output_audit.json').write_text(json.dumps(extra))
        return ref,target

    def test_completed_link_preserves_names_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref,target=self.fixture(root)
            r=link_completed(target,ref,workspace=root)
            self.assertTrue(r['valid']);self.assertTrue(os.path.samefile(r['reference'],r['destination']))
            self.assertEqual(Path(r['destination']).read_bytes(),b'exact prepared model')
            self.assertFalse(r['fresh_statistics_linked'])
            self.assertTrue(link_completed(target,ref,workspace=root)['already_linked'])
            os.chmod(r['reference'],stat.S_IREAD|stat.S_IWRITE)

    def test_failed_missing_or_changed_artifacts_never_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref,target=self.fixture(root)
            file=target/'static_cache/_prepared_checkpoints/same-id/model.safetensors'
            file.write_bytes(b'different')
            with self.assertRaises(ValueError):link_completed(target,ref,workspace=root)
            self.assertEqual(file.read_bytes(),b'different')
            (target/'independent_complete_method_and_output_audit.json').write_text('{}')
            with self.assertRaises(ValueError):link_completed(target,ref,workspace=root)
            with self.assertRaises(ValueError):link_completed(target,ref,workspace=target)


if __name__=='__main__':unittest.main()
