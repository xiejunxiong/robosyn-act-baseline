import unittest

import numpy as np

from robosyn_act.config import load_config
from robosyn_act.dataset_adapter import adapt_metadata, alias_features, validate_batch
from robosyn_act.smoke import synthetic_batch


class AdapterTests(unittest.TestCase):
    def test_alias_is_nonmutating_and_preserves_padding(self):
        state = np.zeros((2, 14), dtype=np.float32)
        mask = np.array([False, True])
        source = {"observation.qpos": state, "observation.qpos_is_pad": mask}
        result = alias_features(source)
        self.assertIs(result["observation.state"], state)
        self.assertIs(result["observation.state_is_pad"], mask)
        self.assertNotIn("observation.state", source)

    def test_canonical_wins(self):
        source = {"observation.qpos": 1, "observation.state": 2}
        self.assertEqual(alias_features(source)["observation.state"], 2)

    def test_metadata_and_stats_aliases_are_independent(self):
        features = {"observation.qpos": {"shape": [14]}, "cam_high.color": {"shape": [480, 640, 3]}}
        stats = {"observation.qpos": {"mean": [0.0] * 14}}
        mapped, mapped_stats = adapt_metadata(features, stats)
        self.assertEqual(mapped["observation.state"]["shape"], [14])
        self.assertIn("observation.images.cam_high", mapped)
        mapped_stats["observation.state"]["mean"][0] = 1
        self.assertEqual(stats["observation.qpos"]["mean"][0], 0)

    def test_wrong_state_camera_chunk_and_padding_rejected(self):
        config = load_config()
        batch = synthetic_batch(config)
        for key, bad in (("observation.state", np.zeros((1, 7))),
                         ("observation.images.cam_high", np.zeros((1, 480, 640, 3))),
                         ("action", np.zeros((1, 25, 14))),
                         ("action_is_pad", np.zeros((1, 49)))):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_batch({**batch, key: bad}, config)
