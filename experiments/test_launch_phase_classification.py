#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


def load_launch_module():
    path = Path(__file__).resolve().parents[1] / "challenges" / "launch.py"
    spec = importlib.util.spec_from_file_location("ate_launch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launch = load_launch_module()


class PhaseClassificationTest(unittest.TestCase):
    def classify(self, command: str) -> str:
        return launch.Runner.classify_install_command(command)

    def test_combined_torch_and_transformer_engine_install_is_te_phase(self):
        command = (
            "uv pip install --python .venv/bin/python --torch-backend cu128 "
            "'torch==2.10.0' 'transformer_engine[pytorch]==2.11.0'"
        )
        self.assertEqual(self.classify(command), "megatron+te-install")

    def test_torch_only_install_is_bootstrap_phase(self):
        command = "uv pip install --python .venv/bin/python --torch-backend cu128 'torch==2.10.0'"
        self.assertEqual(self.classify(command), "torch-bootstrap")

    def test_transformer_engine_search_is_inspection(self):
        command = "rg 'import transformer_engine|from transformer_engine' megatron/core -g '*.py'"
        self.assertEqual(self.classify(command), "inspect")

    def test_transformer_engine_package_list_is_inspection(self):
        command = "uv pip list --python .venv/bin/python | rg 'transformer-engine|torch'"
        self.assertEqual(self.classify(command), "inspect")

    def test_smoke_log_read_is_inspection(self):
        command = "sed -n '1,40p' artifacts/install-smoke.log"
        self.assertEqual(self.classify(command), "inspect")

    def test_smoke_test_is_te_phase(self):
        command = "PYTHONPATH=$PWD/Megatron-LM .venv/bin/python - <<'PY' | tee artifacts/install-smoke.log"
        self.assertEqual(self.classify(command), "megatron+te-install")


if __name__ == "__main__":
    unittest.main()
