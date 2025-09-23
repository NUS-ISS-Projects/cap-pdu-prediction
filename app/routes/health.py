from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
@health_bp.route('/api/prediction/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "cap-pdu-prediction",
        "timestamp": "2024-01-01T00:00:00Z"
    })