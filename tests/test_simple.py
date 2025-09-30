"""Simple tests that don't require RAG system initialization."""

import pytest
from unittest.mock import patch, MagicMock


def test_basic_imports():
    """Test that basic modules can be imported."""
    from app.routes import health
    from app.routes import prediction

    assert health is not None
    assert prediction is not None


def test_config_import():
    """Test that config can be imported."""
    from app.config import Config

    assert Config is not None


def test_blueprint_registration():
    """Test that blueprints can be registered without errors."""
    from flask import Flask
    from app.routes.health import health_bp

    app = Flask(__name__)
    app.register_blueprint(health_bp)

    # Check that the blueprint was registered
    assert "health" in [bp.name for bp in app.blueprints.values()]


def test_health_endpoints_standalone():
    """Test health endpoints without full app initialization."""
    from flask import Flask
    from app.routes.health import health_bp

    app = Flask(__name__)
    app.register_blueprint(health_bp)

    with app.test_client() as client:
        # Test basic health endpoint
        response = client.get("/health")
        assert response.status_code == 200

        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["service"] == "cap-pdu-prediction"
