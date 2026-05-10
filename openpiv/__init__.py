from importlib.metadata import version


__version__ = version("OpenPIV")


def test():
    import pytest

    pytest.main()
