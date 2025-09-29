import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from data.client import DataAcquisitionClient
from data.cache import DataCache
from .chunking import DataChunker
from .vector_store import VectorStore

class DISDataRAGSystem:
    def __init__(self, base_url: str):
        self.client = DataAcquisitionClient(base_url)
        self.cache = DataCache()
        self.vector_store = VectorStore()
        self.chunker = DataChunker()
        self.jwt_token = None

    def set_jwt_token(self, token: str):
        self.jwt_token = token
        self.client.set_jwt_token(token)
        print("JWT token set for prediction service")

        if not self.cache.get('realtime'):
            print("First JWT token set - triggering data refresh...")
            self.refresh_data_cache()

    def refresh_data_cache(self):
        print("Refreshing data cache...")

        realtime_data = self.client.fetch_realtime_data()
        print(f"Real-time data fetched: {bool(realtime_data)}, keys: {list(realtime_data.keys()) if realtime_data else 'None'}")

        today = datetime.now().date()
        start_of_today = today.strftime('%Y-%m-%d')
        end_of_today = today.strftime('%Y-%m-%d')

        current_time_ms = int(time.time() * 1000)
        one_hour_ago_ms = current_time_ms - (60 * 60 * 1000)

        cache_updates = {
            'realtime': realtime_data,
            'aggregated_today': self.client.fetch_historical_data(start_of_today, end_of_today, "hour"),
            'aggregated_week': self.client.fetch_historical_data((today - timedelta(days=7)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'), "day"),
            'aggregated_month': self.client.fetch_historical_data((today - timedelta(days=30)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'), "week"),
            'historical_week': self.client.fetch_historical_data((today - timedelta(days=7)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'), "day"),
            'daily_pdu_logs': self.client.fetch_pdu_logs(current_time_ms - (24 * 60 * 60 * 1000), current_time_ms),
            'weekly_pdu_logs': self.client.fetch_pdu_logs(current_time_ms - (7 * 24 * 60 * 60 * 1000), current_time_ms),
            'monthly_pdu_logs': self.client.fetch_pdu_logs(current_time_ms - (30 * 24 * 60 * 60 * 1000), current_time_ms)
        }

        self.cache.update_cache(cache_updates)
        print(f"Cache updated with keys: {list(cache_updates.keys())}")

    def _fetch_historical_data(self, start_date: str, end_date: str, time_unit: str = "day") -> Dict[str, Any]:
        print(f"Fetched {len(self.client.fetch_historical_data(start_date, end_date, time_unit).get('buckets', []))} data points from acquisition service")
        return self.client.fetch_historical_data(start_date, end_date, time_unit)

    def _process_data_to_chunks(self) -> List[str]:
        chunks = []

        realtime_data = self.cache.get('realtime')
        if realtime_data:
            chunks.append(self.chunker.format_realtime_data(realtime_data))

        period_mappings = [
            ('aggregated_today', 'Today'),
            ('aggregated_week', 'This Week'),
            ('aggregated_month', 'This Month'),
            ('historical_week', 'Historical Week')
        ]

        for cache_key, period_name in period_mappings:
            data = self.cache.get(cache_key)
            if data:
                chunks.extend(self.chunker.format_aggregated_data(data, period_name))

        log_mappings = [
            ('daily_pdu_logs', 'Daily'),
            ('weekly_pdu_logs', 'Weekly'),
            ('monthly_pdu_logs', 'Monthly')
        ]

        for cache_key, period_name in log_mappings:
            logs_data = self.cache.get(cache_key)
            if logs_data:
                chunks.extend(self.chunker.format_pdu_logs(logs_data, period_name))

        return [chunk for chunk in chunks if chunk.strip()]

    def _build_vector_store(self):
        print("Processing data to chunks. Cache keys:", list(self.cache.cache.keys()))
        print("Cache contents summary:", self.cache.get_cache_summary())

        chunks = self._process_data_to_chunks()
        self.vector_store.build_index(chunks)

    def answer_question(self, question: str) -> str:
        if self.cache.should_refresh():
            self.refresh_data_cache()

        self._build_vector_store()

        if not self.vector_store.texts:
            return "I don't have enough data to answer your question. Please ensure the data acquisition service is running and try again."

        relevant_chunks = self.vector_store.search(question, top_k=3)

        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."

        context = "\n".join([chunk for chunk, _ in relevant_chunks])

        return f"Based on the available DIS PDU data:\n\n{context}\n\nThis information should help answer your question about: {question}"