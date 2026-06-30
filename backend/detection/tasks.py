"""Celery tasks for the detection app.

The analysis orchestrator (Task_Queue, Requirement 6) is implemented in
Task 9.1. This module is present so Celery's autodiscovery has a target and the
app wires up cleanly in the skeleton.
"""

from __future__ import annotations

from celery import shared_task


@shared_task(name="detection.ping")
def ping() -> str:
    """Trivial task used to verify broker connectivity in development."""
    return "pong"
