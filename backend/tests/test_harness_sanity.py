"""Self-check for the test harness scaffolding (not a feature/property test).

Verifies that:
  * the Hypothesis profile is loaded with max_examples >= 100, and
  * the deterministic ML-inference stub fixtures return stable values in [0.0, 1.0].

These checks guard the scaffolding configured for property-based testing
(design Testing Strategy / Requirement 8.2); they do not test any product
behavior.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st


def test_hypothesis_profile_runs_at_least_100_examples() -> None:
    assert settings().max_examples >= 100


def test_ml_inference_stub_returns_values_in_unit_interval(ml_inference_stub) -> None:
    for face in (0, "face", (1, 2, 3), 3.14):
        v = ml_inference_stub.infer_face(face)
        assert 0.0 <= v <= 1.0

    agg = ml_inference_stub.aggregate([0.1, 0.2, 0.9])
    assert 0.0 <= agg <= 1.0

    audio = ml_inference_stub.infer_audio(["spec_a", "spec_b"])
    assert 0.0 <= audio <= 1.0


def test_ml_inference_stub_is_deterministic(ml_inference_stub) -> None:
    a = ml_inference_stub.infer_face(("frame", 7))
    b = ml_inference_stub.infer_face(("frame", 7))
    assert a == b


@given(payload=st.one_of(st.integers(), st.text(), st.floats(allow_nan=False, allow_infinity=False)))
@settings(max_examples=100)
def test_deterministic_likelihood_range_and_determinism(deterministic_likelihood, payload) -> None:
    first = deterministic_likelihood(payload)
    second = deterministic_likelihood(payload)
    assert 0.0 <= first <= 1.0
    assert first == second
