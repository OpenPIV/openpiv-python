from importlib.metadata import version

import openpiv


def test_package_version_matches_metadata():
    assert openpiv.__version__ == version("OpenPIV")
