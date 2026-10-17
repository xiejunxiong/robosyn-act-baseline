import contextlib
import importlib.metadata
import io
import json
import unittest
from unittest.mock import patch

from robosyn_act import cli
from robosyn_act.config import load_config
from robosyn_act.smoke import run_smoke


class SmokeTests(unittest.TestCase):
    def test_contract_smoke_does_not_claim_model_forward(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(["smoke"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["mode"], "contract")
        self.assertFalse(report["real_act_forward"])
        self.assertEqual(report["shapes"]["action"], [1, 50, 14])
        for camera in cli.CAMERAS:
            self.assertEqual(report["shapes"][f"observation.images.{camera}"], [1, 3, 480, 640])

    def test_missing_real_model_dependency_fails_explicitly(self):
        with patch("importlib.metadata.distribution", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(ValueError, "Real ACT smoke unavailable"):
                run_smoke(load_config(), act=True)

    def test_config_cli(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["config"]), 0)
        self.assertEqual(json.loads(output.getvalue())["training"]["batch_size"], 24)
