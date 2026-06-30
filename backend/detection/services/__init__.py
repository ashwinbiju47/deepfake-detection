"""Service layer for the detection app.

Services encapsulate domain logic (validation pipelines, orchestration helpers)
that sits between the thin DRF view layer and the ORM models. Keeping this logic
out of the views makes it independently unit/property-testable.
"""
