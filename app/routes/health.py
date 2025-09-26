from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)

def get_rag_system():
    from app.main import rag_system
    return rag_system

@health_bp.route('/health', methods=['GET'])
@health_bp.route('/api/prediction/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "cap-pdu-prediction",
        "timestamp": "2024-01-01T00:00:00Z"
    })

@health_bp.route('/api/prediction/chat/health', methods=['GET'])
def chat_health():
    rag = get_rag_system()
    return jsonify({
        "authenticated": bool(rag.jwt_token),
        "cache_summary": rag.cache.get_cache_summary(),
        "vector_store_ready": bool(rag.vector_store.texts)
    })