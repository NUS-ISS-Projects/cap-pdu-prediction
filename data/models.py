"""Data models for request and response objects."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class PredictionRequest:
    """Request model for prediction endpoints."""

    time_unit: str
    start_date: str
    prediction_periods: int = None

    def __post_init__(self):
        if self.prediction_periods is None:
            self.prediction_periods = {"day": 24, "week": 2, "month": 4}.get(self.time_unit, 24)


@dataclass
class ChatRequest:
    """Request model for chat endpoints."""

    question: str
    session_id: Optional[str] = None
