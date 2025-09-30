"""Utility functions for prediction and forecast label generation."""

from datetime import datetime, timedelta
import calendar


def generate_forecast_labels(start_date, time_unit, num_periods):
    """Generate forecast labels based on time unit and number of periods."""
    labels = []

    if time_unit == "day":
        current_time = datetime.now()
        current_hour = current_time.hour

        for i in range(num_periods):
            hour = (current_hour + i) % 24
            labels.append(f"{hour:02d}:00")

    elif time_unit == "week":
        current_date = start_date
        for i in range(num_periods):
            forecast_date = current_date + timedelta(days=i + 1)
            labels.append(forecast_date.strftime("%Y-%m-%d"))

    elif time_unit == "month":
        for i in range(1, num_periods + 1):
            labels.append(f"Week {i} {calendar.month_abbr[start_date.month]}")

    return labels
