from flask import Flask
from app.config import Config
from app.routes.health import health_bp
from app.routes.prediction import prediction_bp
from app.routes.chat import chat_bp
from rag.system import DISDataRAGSystem

rag_system = None

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    global rag_system
    print("Initializing RAG system during application startup...")
    print(f"Using DIS base URL: {Config.DIS_BASE_URL}")

    rag_system = DISDataRAGSystem(base_url=Config.DIS_BASE_URL)
    print("RAG system initialization completed.")

    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(chat_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=(Config.FLASK_ENV != 'production'))