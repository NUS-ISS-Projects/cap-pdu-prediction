import os

class Config:
    DIS_BASE_URL = os.getenv('DIS_BASE_URL', 'http://kong-gateway-service.default.svc.cluster.local:8000"')
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')

    TRANSFORMERS_CACHE = '/app/.cache/transformers'
    HF_HOME = '/app/.cache/huggingface'

    @classmethod
    def get_endpoints(cls):
        return {
            'acquisition': f"{cls.DIS_BASE_URL}/api/acquisition",
            'ingestion': f"{cls.DIS_BASE_URL}/api/ingestion",
            'processing': f"{cls.DIS_BASE_URL}/api/processing"
        }