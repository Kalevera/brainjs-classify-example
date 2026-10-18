import sys
import types
import unittest
from unittest.mock import patch

from classifier.device import select_device


class DeviceTests(unittest.TestCase):
    def test_invalid_device_rejected(self):
        with self.assertRaisesRegex(ValueError, "cuda.*cpu"):
            select_device("automatic")

    def test_cpu_selection_never_probes_cuda(self):
        fake_torch = types.SimpleNamespace(device=lambda name: name)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(select_device("cpu"), "cpu")

    def test_unavailable_cuda_does_not_fall_back(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                select_device()

    def test_available_cuda_still_checks_kernel_compatibility(self):
        def incompatible(*args, **kwargs):
            raise RuntimeError("no kernel image is available")

        class NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            device=lambda name: name,
            no_grad=NoGrad,
            ones=incompatible,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "no kernel image"):
                select_device()

    def test_real_cuda_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        self.assertEqual(select_device().type, "cuda")


if __name__ == "__main__":
    unittest.main()
