from flask import Blueprint, request, jsonify
from core.prediction import PredictionEngine
from data.models import PredictionRequest

prediction_bp = Blueprint('prediction', __name__)

def get_rag_system():
    from app.main import rag_system
    return rag_system

@prediction_bp.route('/predict', methods=['POST'])
@prediction_bp.route('/api/prediction', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        time_unit = data.get('timeUnit', 'day')
        start_date = data.get('startDate')

        if not start_date:
            return jsonify({"error": "startDate is required"}), 400

        req = PredictionRequest(
            time_unit=time_unit,
            start_date=start_date,
            prediction_periods=data.get('predictionPeriods')
        )

        rag = get_rag_system()
        if not rag.jwt_token:
            return jsonify({
                "error": "Authentication required",
                "details": "Unable to fetch data from acquisition service. Check authentication and data availability."
            }), 401

        result = PredictionEngine.process_prediction_request(
            rag, req.time_unit, req.start_date, req.prediction_periods
        )

        if result is None:
            return jsonify({
                "error": "Unable to generate predictions",
                "details": "Unable to fetch data from acquisition service. Check authentication and data availability."
            }), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500