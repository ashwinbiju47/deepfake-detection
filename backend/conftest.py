"""Shared pytest + Hypothesis configuration for the deepfake-detection-platform backend.

This module is intentionally free of Django and PyTorch imports. Pure-logic
property and unit tests (validation, fusion, labeling, metric math, event
replay) must run without a database, a GPU, or the ML stack. Inference is
represented in logic tests by the deterministic stub fixtures defined below, so
PyTorch is never invoked when exercising business logic.

Defined here:
  * Hypothesis profiles registering and loading a profile with max_examples >= 100
    (design Testing Strategy / Requirement 8.2).
  * ``ml_inference_stub`` / ``deterministic_likelihood`` fixtures producing stable
    values in the inclusive range [0.0, 1.0].
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, Verbosity, settings

# ---------------------------------------------------------------------------
# Django settings module for pytest-django.
#
# The real settings module is created by the project scaffolding task (task 1.1).
# We use setdefault so an explicit DJANGO_SETTINGS_MODULE env var or the value in
# pytest.ini always wins. If task 1.1 uses a different package name, update the
# value in pytest.ini (or export DJANGO_SETTINGS_MODULE) to match.
# ---------------------------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")


# ---------------------------------------------------------------------------
# Hypothesis profiles.
#
# Minimum 100 examples per property test. "default" is loaded unless the
# HYPOTHESIS_PROFILE environment variable selects another profile. The
# function_scoped_fixture health check is suppressed because property tests
# commonly consume function-scoped fixtures (e.g. ml_inference_stub) under @given.
# ---------------------------------------------------------------------------
settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "dev",
    max_examples=100,
    deadline=None,
    verbosity=Verbosity.verbose,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


# ---------------------------------------------------------------------------
# Deterministic ML-inference stub.
# ---------------------------------------------------------------------------
def _deterministic_likelihood(*inputs: Any) -> float:
    """Map arbitrary inputs to a stable pseudo-likelihood in the inclusive range [0.0, 1.0].

    The mapping is a pure function of ``repr(inputs)`` via SHA-256, so identical
    inputs always produce an identical value. This lets logic and property tests
    stand in for Visual_Model / Audio_Model inference without importing or running
    PyTorch, and keeps fusion/labeling determinism tests reproducible.
    """
    key = repr(inputs).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    n = int.from_bytes(digest[:8], "big")
    return n / 0xFFFFFFFFFFFFFFFF  # normalize to [0.0, 1.0] inclusive


class DeterministicInferenceStub:
    """Deterministic stand-in for the Visual_Model and Audio_Model.

    Every method returns a value in the inclusive range [0.0, 1.0] derived purely
    from its inputs. No PyTorch, no I/O, no randomness.
    """

    def infer_face(self, face: Any) -> float:
        """Per-face visual likelihood in [0.0, 1.0] (stands in for Visual_Model.infer_face)."""
        return _deterministic_likelihood("visual_face", face)

    def aggregate(self, per_frame: Any) -> float:
        """Aggregate visual likelihood in [0.0, 1.0] (stands in for Visual_Model.aggregate)."""
        items = tuple(per_frame) if isinstance(per_frame, (list, tuple)) else (per_frame,)
        return _deterministic_likelihood("visual_aggregate", items)

    def infer_audio(self, spectrograms: Any) -> float:
        """Audio likelihood in [0.0, 1.0] (stands in for Audio_Model.infer)."""
        return _deterministic_likelihood("audio", repr(spectrograms))

    def __call__(self, *inputs: Any) -> float:
        """Generic deterministic likelihood for ad-hoc use in tests."""
        return _deterministic_likelihood(*inputs)


@pytest.fixture
def ml_inference_stub() -> DeterministicInferenceStub:
    """Provide a deterministic ML-inference stub returning values in [0.0, 1.0].

    Use in property and unit tests in place of the real PyTorch models so that
    inference is never invoked in logic tests.
    """
    return DeterministicInferenceStub()


@pytest.fixture
def deterministic_likelihood() -> Callable[..., float]:
    """Provide the raw deterministic likelihood function (inputs -> [0.0, 1.0])."""
    return _deterministic_likelihood
