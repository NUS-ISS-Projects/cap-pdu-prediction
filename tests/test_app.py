import pytest
from unittest.mock import patch, MagicMock
from app.main import create_app


@pytest.fixture
def app():
    """Create and configure a test application instance."""
    with patch('rag.system.DISDataRAGSystem') as mock_rag_class:
        # Mock the RAG system to prevent initialization issues in tests
        mock_rag_instance = MagicMock()
        mock_rag_instance.jwt_token = None
        mock_rag_class.return_value = mock_rag_instance

        app = create_app()
        app.config['TESTING'] = True
        return app


@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()


def test_app_creation(app):
    """Test that the Flask app is created successfully."""
    assert app is not None
    assert app.config['TESTING'] is True


def test_app_has_blueprints(app):
    """Test that the app has registered the required blueprints."""
    blueprint_names = [bp.name for bp in app.blueprints.values()]
    assert 'health' in blueprint_names
    assert 'prediction' in blueprint_names
    assert 'chat' in blueprint_names


def test_404_error(client):
    """Test that non-existent endpoints return 404."""
    response = client.get('/non-existent-endpoint')
    assert response.status_code == 404