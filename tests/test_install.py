"""Simplest installation test."""
from types import ModuleType


def test_install():
    import baypy as bp

    assert isinstance(bp, ModuleType)
