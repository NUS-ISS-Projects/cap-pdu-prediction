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
        mock_rag_instance.cache.get_cache_summary.return_value = {'cached_items': 0}
        mock_rag_instance.vector_store.texts = []
        mock_rag_class.return_value = mock_rag_instance

        app = create_app()
        app.config['TESTING'] = True
        return app


@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()


def test_health_endpoint(client):
    """Test the main health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200

    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'cap-pdu-prediction'
    assert 'timestamp' in data


def test_prediction_health_endpoint(client):
    """Test the prediction service health endpoint."""
    response = client.get('/api/prediction/health')
    assert response.status_code == 200

    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'cap-pdu-prediction'


def test_chat_health_endpoint(client):
    """Test the chat service health endpoint."""
    response = client.get('/api/prediction/chat/health')
    assert response.status_code == 200

    data = response.get_json()
    assert 'authenticated' in data
    assert 'cache_summary' in data
    assert 'vector_store_ready' in data