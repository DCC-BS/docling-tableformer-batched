"""Pytest configuration for docling_tableformer_batched tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests unless ``TABLEFORMER_E2E`` is set: they download the TableFormer models."""
    if os.environ.get("TABLEFORMER_E2E"):
        return
    skip = pytest.mark.skip(reason="TABLEFORMER_E2E not set")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)
