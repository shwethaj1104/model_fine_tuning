"""Check that final evaluation refuses an altered selected adapter or prompt."""
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.final_test import ROOT, verify
from evaluation.run_model import sha256


class FrozenExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)/'bundle'
        adapter=self.bundle/'training/adapter'
        adapter.mkdir(parents=True)
        self.reference=json.loads((ROOT/'training/final_test_reference.json').read_text())
        identity={k:self.reference[k] for k in ('model_id','resolved_revision','prompt_sha256')}
        identity['smoke']=False
        (adapter/'training_identity.json').write_text(json.dumps(identity))
        (adapter/'adapter_model.safetensors').write_bytes(b'test fixture; not real weights')
        self.reference['adapter_files_sha256']={p.name:sha256(p) for p in adapter.iterdir()}
        self.path=Path(self.temp.name)/'reference.json'
        self.path.write_text(json.dumps(self.reference))
    def test_frozen_inputs_verified_without_model_loading(self):
        _,adapter,rows=verify(self.bundle,self.path)
        self.assertEqual(len(rows),66)
        self.assertEqual(adapter,self.bundle/'training/adapter')
    def test_altered_weights_rejected(self):
        (self.bundle/'training/adapter/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen adapter file changed'):
            verify(self.bundle,self.path)
    def test_changed_prompt_reference_rejected(self):
        self.reference['prompt_sha256']='not the frozen hash'
        self.path.write_text(json.dumps(self.reference))
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):
            verify(self.bundle,self.path)


if __name__=='__main__':unittest.main()
