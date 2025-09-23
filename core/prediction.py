import json
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from .utils import generate_forecast_labels

class PredictionEngine:
    @staticmethod
    def predict_pdu_data(data):
        if not data or len(data) < 2:
            return [0.0] * 24

        x = np.array(range(len(data))).reshape(-1, 1)
        y = np.array(data)

        model = LinearRegression()
        model.fit(x, y)

        future_x = np.array(range(len(data), len(data) + 24)).reshape(-1, 1)
        predictions = model.predict(future_x)

        predictions = np.maximum(predictions, 0)
        return predictions.tolist()

    @staticmethod
    def process_prediction_request(rag_system, time_unit, start_date_str, prediction_periods):
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            start_date = datetime.now()

        if time_unit == 'day':
            training_periods = 24
            end_date = start_date
            start_date_for_training = start_date - timedelta(days=1)
        elif time_unit == 'week':
            training_periods = prediction_periods
            end_date = start_date
            start_date_for_training = start_date - timedelta(weeks=8)
        elif time_unit == 'month':
            training_periods = prediction_periods
            end_date = start_date
            start_date_for_training = start_date - timedelta(days=180)
        else:
            return None

        historical_data = rag_system._fetch_historical_data(
            start_date_for_training.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            time_unit
        )

        if not historical_data or 'buckets' not in historical_data:
            return None

        training_data = [bucket.get('totalPackets', 0) for bucket in historical_data['buckets']]
        training_data = training_data[-training_periods:]

        while len(training_data) < training_periods:
            training_data.insert(0, 0)

        predictions = PredictionEngine.predict_pdu_data(training_data)[:prediction_periods]
        labels = generate_forecast_labels(start_date, time_unit, prediction_periods)

        training_range = {
            "startDate": start_date_for_training.strftime('%Y-%m-%d'),
            "endDate": end_date.strftime('%Y-%m-%d')
        }

        return {
            "predicted_values": predictions,
            "predicted_labels": labels,
            "metadata": {
                "forecastStartDate": start_date_str,
                "timeUnit": time_unit,
                "predictionPeriods": prediction_periods,
                "trainingDataPoints": len(training_data),
                "trainingDataRange": training_range,
                "dataSource": "acquisition_service"
            }
        }