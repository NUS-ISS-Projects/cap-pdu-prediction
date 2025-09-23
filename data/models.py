from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class PredictionRequest:
    time_unit: str
    start_date: str
    prediction_periods: int = None

    def __post_init__(self):
        if self.prediction_periods is None:
            self.prediction_periods = {
                'day': 24,
                'week': 2,
                'month': 4
            }.get(self.time_unit, 24)

@dataclass
class ChatRequest:
    question: str
    session_id: Optional[str] = None