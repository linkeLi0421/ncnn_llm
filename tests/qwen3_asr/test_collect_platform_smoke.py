#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("collect_platform_smoke.py")


def load_module():
    spec = importlib.util.spec_from_file_location("collect_platform_smoke", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PlatformMetadataTest(unittest.TestCase):
    def test_windows_uses_processor_identifier_without_sysctl(self) -> None:
        module = load_module()
        with (
            mock.patch.object(module.platform, "system", return_value="Windows"),
            mock.patch.object(module.platform, "processor", return_value=""),
            mock.patch.object(module, "cmd_output") as cmd_output,
            mock.patch.dict(os.environ, {"PROCESSOR_IDENTIFIER": "Test Windows CPU"}),
        ):
            metadata = module.platform_metadata()

        self.assertEqual(metadata["system"], "Windows")
        self.assertEqual(metadata["cpu_brand"], "Test Windows CPU")
        self.assertEqual(metadata["hw_logicalcpu"], str(os.cpu_count()))
        cmd_output.assert_not_called()

    def test_linux_reads_cpuinfo_and_does_not_call_sysctl(self) -> None:
        module = load_module()
        with (
            mock.patch.object(module.platform, "system", return_value="Linux"),
            mock.patch.object(module, "proc_cpuinfo_value", side_effect=["Test Linux CPU", "sse avx2"]),
            mock.patch.object(module, "cmd_output") as cmd_output,
        ):
            metadata = module.platform_metadata()

        self.assertEqual(metadata["cpu_brand"], "Test Linux CPU")
        self.assertEqual(metadata["cpu_features"], "sse avx2")
        cmd_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
