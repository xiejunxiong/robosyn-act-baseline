import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from robosyn_act import cli


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robosyn test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset with spaces"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "data/chunk-000").mkdir(parents=True)
        (self.dataset / "data/chunk-000/episode_000000.parquet").touch()
        (self.dataset / "videos/chunk-000").mkdir(parents=True)
        (self.dataset / "videos/chunk-000/episode_000000.mp4").touch()
        self.info = {
            "codebase_version": "v2.1", "fps": 30, "total_episodes": 1,
            "total_frames": 12, "features": {
                "observation.qpos": {"dtype": "float32", "shape": [14]},
                "action": {"dtype": "float32", "shape": [14]},
                **{f"{camera}.color": {"dtype": "video", "shape": [480, 640, 3]}
                   for camera in cli.CAMERAS},
            },
        }
        self.save_info()
        for name in ("episodes", "tasks", "episodes_stats"):
            (self.dataset / f"meta/{name}.jsonl").touch()
        self.upstream = self.root / "RoboSynChallenge"
        for name in ("policy/act/scripts/train.py", "policy/act/eval.sh", "policy/act/uv.lock"):
            path = self.upstream / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.checkpoint = self.root / "pretrained model"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text('{"type": "act"}')
        (self.checkpoint / "model.safetensors").touch()
        self.output = self.root / "new output"

    def save_info(self):
        (self.dataset / "meta/info.json").write_text(json.dumps(self.info))

    def args(self, *extra):
        return cli.parser().parse_args([
            "train", "--upstream", str(self.upstream), "--python", sys.executable,
            "--dataset-root", str(self.dataset), "--output-dir", str(self.output), *extra,
        ])

    def build(self, args):
        with patch.object(cli, "upstream_root", return_value=self.upstream):
            return cli.build_command(args)

    def test_legacy_and_canonical_data(self):
        self.assertEqual(cli.validate_dataset(self.dataset)["state_dim"], 14)
        features = self.info["features"]
        features["observation.state"] = features.pop("observation.qpos")
        for camera in cli.CAMERAS:
            features[f"observation.images.{camera}"] = features.pop(f"{camera}.color")
        self.save_info()
        self.assertEqual(cli.validate_dataset(self.dataset)["frames"], 12)

    def test_reject_v3(self):
        self.info["codebase_version"] = "v3.0"
        self.save_info()
        with self.assertRaisesRegex(ValueError, "v2.1"):
            cli.validate_dataset(self.dataset)

    def test_missing_camera(self):
        del self.info["features"]["cam_high.color"]
        self.save_info()
        with self.assertRaisesRegex(ValueError, "cam_high"):
            cli.validate_dataset(self.dataset)

    def test_missing_metadata_and_video(self):
        (self.dataset / "meta/episodes_stats.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "episodes_stats"):
            cli.validate_dataset(self.dataset)
        (self.dataset / "videos/chunk-000/episode_000000.mp4").unlink()
        with self.assertRaisesRegex(ValueError, "no videos"):
            cli.validate_dataset(self.dataset)

    def test_bad_shape_and_fps(self):
        for shape in (None, [0], [2, 3], [True]):
            with self.subTest(shape=shape):
                self.info["features"]["action"]["shape"] = shape
                self.save_info()
                with self.assertRaises(ValueError):
                    cli.validate_dataset(self.dataset)
        self.info["features"]["action"]["shape"] = [14]
        self.info["fps"] = float("nan")
        self.save_info()
        with self.assertRaisesRegex(ValueError, "fps"):
            cli.validate_dataset(self.dataset)

    def test_smoke_and_override(self):
        cmd, cwd, env = self.build(self.args("--profile", "smoke", "--batch-size", "2"))
        self.assertEqual(cmd[cmd.index("--steps") + 1], "10")
        self.assertEqual(cmd[cmd.index("--batch-size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--dataset-root") + 1], str(self.dataset.resolve()))
        self.assertEqual(cwd, self.upstream)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("--wandb", cmd)
        self.assertNotIn("--overwrite", cmd)
        self.assertFalse(self.output.exists())

    def test_existing_output_never_deleted(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("experiment")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build(self.args())
        self.assertEqual(marker.read_text(), "experiment")

    def test_nested_output_and_invalid_chunk(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            self.build(self.args("--output-dir", str(self.dataset / "new")))
        with self.assertRaisesRegex(ValueError, "n-action-steps"):
            self.build(self.args("--chunk-size", "4"))

    def test_invalid_checkpoint(self):
        (self.checkpoint / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "model.safetensors"):
            cli.validate_checkpoint(self.checkpoint)

    def test_eval_open_pan_path_and_flags(self):
        config = self.upstream / "configs/other_tasks/open_pan/random/gym_config.json"
        config.parent.mkdir(parents=True)
        config.write_text("{}")
        args = cli.parser().parse_args([
            "eval", "--upstream", str(self.upstream), "--python", sys.executable,
            "--task", "open_pan", "--checkpoint", str(self.checkpoint), "--episodes", "3",
        ])
        cmd, cwd, env = self.build(args)
        self.assertEqual(cmd[2:4], ["other_tasks/open_pan", "random"])
        self.assertEqual(cmd[4], str(self.checkpoint.resolve()))
        self.assertEqual(cmd[cmd.index("--headless") + 1], "True")
        self.assertEqual(cmd[cmd.index("--max_episodes") + 1], "3")
        self.assertEqual(env["PYTHON_BIN"], os.path.abspath(sys.executable))

    def test_revision_mismatch(self):
        with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "other\n")):
            with self.assertRaisesRegex(ValueError, "revision"):
                cli.upstream_root(self.upstream)

    def test_non_object_json(self):
        (self.checkpoint / "config.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            cli.validate_checkpoint(self.checkpoint)

    def test_download_dry_run_does_not_launch(self):
        with patch.object(cli.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main(["download", "checkpoint", "--task", "click_bell",
                             "--destination", str(self.checkpoint), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("RoboSynChallenge/ACT_sim_click_bell", out.getvalue())
        run.assert_not_called()

    def test_real_subprocess_exit_status_and_path_spaces(self):
        # A fake upstream trainer checks argv and environment without importing Torch.
        script = self.upstream / "policy/act/scripts/train.py"
        script.write_text(
            "import os, sys\n"
            "assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'\n"
            "assert 'dataset with spaces' in sys.argv[sys.argv.index('--dataset-root') + 1]\n"
            "sys.exit(7)\n"
        )
        with patch.object(cli, "upstream_root", return_value=self.upstream), \
                contextlib.redirect_stdout(io.StringIO()):
            result = cli.main(["train", "--upstream", str(self.upstream),
                               "--python", sys.executable, "--dataset-root", str(self.dataset),
                               "--output-dir", str(self.output), "--gpu", "2"])
        self.assertEqual(result, 7)


if __name__ == "__main__":
    unittest.main()
