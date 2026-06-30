# Backend tests

Test harness for the deepfake-detection-platform Django backend.

## Layout

- `tests/unit/` — example-based unit tests (specific cases and edge cases).
- `tests/property/` — Hypothesis property-based tests (one test per numbered
  correctness property in `design.md`, each tagged with its requirement link).
- `test_harness_sanity.py` — self-check that the shared fixtures and Hypothesis
  profile are wired correctly. Not a feature test.

## Configuration

- `../pytest.ini` configures pytest + pytest-django and test discovery.
- `../conftest.py` registers Hypothesis profiles (default: `max_examples = 100`)
  and provides the deterministic ML-inference stub fixtures.

## Conventions

- Property tests use the **Hypothesis** framework and run **>= 100 examples**.
- Inference is represented by the `ml_inference_stub` / `deterministic_likelihood`
  fixtures, which return deterministic values in `[0.0, 1.0]`. PyTorch is never
  invoked in logic tests.
- Select a Hypothesis profile via `HYPOTHESIS_PROFILE` (`default`, `ci`, `dev`).

## Running

```bash
# From the backend/ directory:
pytest                      # all tests, default profile (100 examples)
HYPOTHESIS_PROFILE=ci pytest  # 200 examples
pytest -m property          # only property tests
pytest -m unit              # only unit tests
```
