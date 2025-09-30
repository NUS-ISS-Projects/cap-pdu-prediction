"""Health check routes for the prediction service."""

from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__)


def get_rag_system():
    """Get the RAG system from the Flask application context."""
    return current_app.rag_system


@health_bp.route("/health", methods=["GET"])
@health_bp.route("/api/prediction/health", methods=["GET"])
def health():
    """Health check endpoint for the prediction service."""
    return jsonify(
        {"status": "healthy", "service": "cap-pdu-prediction", "timestamp": "2024-01-01T00:00:00Z"}
    )


@health_bp.route("/api/prediction/chat/health", methods=["GET"])
def chat_health():
    """Health check endpoint with authentication and cache status."""
    rag = get_rag_system()
    return jsonify(
        {
            "authenticated": bool(rag.jwt_token),
            "cache_summary": rag.cache.get_cache_summary(),
            "vector_store_ready": bool(rag.vector_store.texts),
        }
    )
