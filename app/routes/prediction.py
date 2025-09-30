"""Prediction routes for PDU forecasting."""

from flask import Blueprint, request, jsonify, current_app
from core.prediction import PredictionEngine
from data.models import PredictionRequest

prediction_bp = Blueprint("prediction", __name__)


def get_rag_system():
    """Get the RAG system from the Flask application context."""
    return current_app.rag_system


@prediction_bp.route("/predict", methods=["POST"])
@prediction_bp.route("/api/prediction", methods=["POST"])
def predict():
    """Handle prediction requests with authentication and data fetching."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        time_unit = data.get("timeUnit", "day")
        start_date = data.get("startDate")

        if not start_date:
            return jsonify({"error": "startDate is required"}), 400

        req = PredictionRequest(
            time_unit=time_unit,
            start_date=start_date,
            prediction_periods=data.get("predictionPeriods"),
        )

        rag = get_rag_system()

        # Extract JWT token from request headers
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            rag.set_jwt_token(token)

        if not rag.jwt_token:
            return (
                jsonify(
                    {
                        "error": "Authentication required",
                        "details": (
                            "Unable to fetch data from acquisition service. "
                            "Check authentication and data availability."
                        ),
                    }
                ),
                401,
            )

        result = PredictionEngine.process_prediction_request(
            rag, req.time_unit, req.start_date, req.prediction_periods
        )

        if result is None:
            return (
                jsonify(
                    {
                        "error": "Unable to generate predictions",
                        "details": (
                            "Unable to fetch data from acquisition service. "
                            "Check authentication and data availability."
                        ),
                    }
                ),
                500,
            )

        return jsonify(result)

    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Bad request: {str(e)}"}), 400
    except Exception as e:  # pylint: disable=broad-exception-caught
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
