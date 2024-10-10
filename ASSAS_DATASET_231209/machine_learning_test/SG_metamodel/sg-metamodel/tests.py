
import unittest
import os
import gv_model as gvm

class PipelineTests(unittest.TestCase):

    def test_paths_exist(self):
        import pyastec as pyas
        dir = "/TMPCALCUL/Campagne_ASTEC/WEEKLY_RELEASE/RELEASE_230630/bin_release_int_2023_06_30-11h27/machine_learning/autoencoder_simulator/build_basis/run_0/reference_reduced.bin"
        time = 100
        base = pyas.odloaddir(dir, time)

        metadata = gvm.get_gv_model_metadata(base)

        for path in gvm.generate_paths(metadata.name_mapping):
            with self.subTest(f"Path {path}"):
                self.assertTrue(path.exists_from(base), "Path doesn't exist")
