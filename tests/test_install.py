"""Simplest installation test."""
from types import ModuleType


def test_install():
    import jnotype as jn

    assert isinstance(jn, ModuleType)
