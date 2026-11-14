"""Tests for PIVSettings YAML round-trip."""

import os
import tempfile

from openpiv import windef


def test_settings_yaml_roundtrip():
    s = windef.Settings()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "settings.yaml")
        s.to_yaml(path)
        s2 = windef.Settings.from_yaml(path)
        assert vars(s) == vars(s2)


if __name__ == "__main__":
    __import__("pytest").main([__file__, "-v"])
