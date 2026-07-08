import pytest
import torch
import pickle
import io
import os
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'ltx-trainer', 'scripts'))


class MaliciousPayload:
    """A class that executes code when unpickled."""
    def __reduce__(self):
        return (os.system, ("echo EXPLOITED > /tmp/_exploit_marker",))


def create_malicious_pt_file(path):
    """Create a .pt file containing a malicious pickled object."""
    buffer = io.BytesIO()
    torch.save(MaliciousPayload(), buffer)
    with open(path, 'wb') as f:
        f.write(buffer.getvalue())


def create_valid_pt_file(path):
    """Create a legitimate .pt file with tensor data."""
    data = {"latents": torch.randn(1, 4, 8, 8), "prompt": "test"}
    torch.save(data, path)


@pytest.mark.parametrize("file_creator,should_be_safe", [
    (create_malicious_pt_file, True),   # Malicious payload must not execute
    (create_valid_pt_file, False),       # Valid file as baseline
])
def test_torch_load_does_not_execute_arbitrary_code(file_creator, should_be_safe, tmp_path):
    """Invariant: Loading .pt files must not allow arbitrary code execution via pickle deserialization.
    
    torch.load with weights_only=False enables arbitrary code execution through crafted .pt files.
    This test verifies that the security boundary is violated by the current code, serving as a
    regression guard to ensure a fix (weights_only=True or safe loading) is applied.
    """
    marker_file = "/tmp/_exploit_marker"
    if os.path.exists(marker_file):
        os.remove(marker_file)

    pt_file = str(tmp_path / "test_latent.pt")
    file_creator(pt_file)

    if should_be_safe:
        # The secure behavior: loading should either reject the file or not execute the payload
        # We test that torch.load with weights_only=True would prevent exploitation
        with pytest.raises((pickle.UnpicklingError, TypeError, RuntimeError, Exception)):
            torch.load(pt_file, map_location="cpu", weights_only=True)
        
        # Now verify the actual script's approach is UNSAFE (weights_only=False)
        # This documents the vulnerability - if this stops failing, the fix is in place
        from decode_latents import LatentDecoder
        # We don't instantiate the full decoder, but we verify the unsafe pattern exists
        import inspect
        source = inspect.getsource(LatentDecoder)
        # This assertion will FAIL once the vulnerability is fixed (weights_only=True)
        # Acting as a regression test: if someone re-introduces weights_only=False, this catches it
        assert "weights_only=True" in source or "weights_only=False" not in source, (
            "SECURITY VIOLATION: decode_latents.py uses torch.load with weights_only=False, "
            "enabling arbitrary code execution via crafted .pt files. "
            "Use weights_only=True or implement safe deserialization."
        )
    else:
        # Valid file should load fine with safe settings
        data = torch.load(pt_file, map_location="cpu", weights_only=True)
        assert "latents" in data