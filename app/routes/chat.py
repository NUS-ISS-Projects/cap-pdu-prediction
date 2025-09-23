from flask import Blueprint, request, jsonify
from data.models import ChatRequest

chat_bp = Blueprint('chat', __name__)

def get_rag_system():
    from app.main import rag_system
    return rag_system

@chat_bp.route('/chat', methods=['POST'])
@chat_bp.route('/api/prediction/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        question = data.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        req = ChatRequest(
            question=question,
            session_id=data.get('sessionId')
        )

        rag = get_rag_system()
        if not rag.jwt_token:
            return jsonify({
                "error": "Authentication required. Please ensure you have valid access to the data acquisition service."
            }), 401

        answer = rag.answer_question(req.question)
        print("JWT token updated for RAG system authentication")

        return jsonify({
            "answer": answer,
            "question": req.question,
            "sessionId": req.session_id
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@chat_bp.route('/chat/status', methods=['GET'])
@chat_bp.route('/api/prediction/chat/status', methods=['GET'])
def chat_status():
    rag = get_rag_system()
    return jsonify({
        "authenticated": bool(rag.jwt_token),
        "cache_summary": rag.cache.get_cache_summary(),
        "vector_store_ready": bool(rag.vector_store.texts)
    })