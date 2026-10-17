from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from robosyn_act.config import load_config


class ConfigTests(unittest.TestCase):
    def test_historical_training_settings_match_launcher(self):
        config = load_config()
        historical = yaml.safe_load((Path(__file__).parents[1] /
                                    "experiments/click_bell_80k/config.yaml").read_text())
        for key, value in historical["training"].items():
            if key != "seed":
                self.assertEqual(config["training"][key], value, key)
        self.assertEqual(config["evaluation"]["seed"], 0)
        self.assertEqual(config["evaluation"]["episodes"], 10)

    def test_reject_bad_yaml_contract(self):
        base = load_config()
        mutations = [
            lambda x: x.update(extra="typo"),
            lambda x: x["training"].update(batch_size=True),
            lambda x: x["training"].update(n_action_steps=51),
            lambda x: x["training"].update(num_workers=-1),
            lambda x: x["dataset"].update(image_shape=[480, 640, 3]),
            lambda x: x["evaluation"].update(seed=-1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            for mutate in mutations:
                config = deepcopy(base)
                mutate(config)
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_config(path)
            path.write_text("training: [")
            with self.assertRaisesRegex(ValueError, "Cannot load"):
                load_config(path)

    def test_recorded_results_are_consistent_not_claimed_verified(self):
        path = Path(__file__).parents[1] / "experiments/click_bell_80k/results.json"
        result = json.loads(path.read_text())
        episodes, summary = result["episodes"], result["summary"]
        self.assertFalse(result["provenance"]["independently_verified"])
        self.assertFalse(result["official_competition_result"])
        self.assertEqual(len(episodes), 10)
        self.assertEqual(sum(e["success"] for e in episodes), 7)
        self.assertAlmostEqual(sum(e["action_steps"] for e in episodes) / 10,
                               summary["average_action_steps"])
        self.assertEqual(sum(e["model_calls"] for e in episodes), summary["total_model_calls"])
        self.assertEqual(summary["success_rate"], 7 / 10)
