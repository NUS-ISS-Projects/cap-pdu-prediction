# rag_system.py

import json
import numpy as np
import torch
import os
import requests
from datetime import datetime, timedelta
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Optional

class DISDataRAGSystem:
    """
    A Retrieval-Augmented Generation system for DIS platform data queries.
    
    This system can fetch real-time data from acquisition, ingestion, and processing services
    and provide natural language responses to user queries about the DIS platform.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initializes the RAG system with service endpoints.
        
        Args:
            base_url (str): Base URL for the DIS platform services
        """
        print("Initializing DIS Data RAG System...")
        self.base_url = base_url
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Service endpoints
        self.endpoints = {
            "acquisition": f"{base_url}/api/acquisition",
            "ingestion": f"{base_url}/api/ingestion", 
            "processing": f"{base_url}/api/processing"
        }
        
        # Initialize models
        self._load_models()
        
        # Initialize data cache
        self.data_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 300  # 5 minutes cache
        
        # Vector store
        self.vector_store = None
        self.source_chunks = []
        
    def _load_models(self):
        """
        Load the retriever and generator models.
        """
        print("Loading Retriever model (for embeddings)...")
        self.retriever_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        print("Loading Generative model (for Q&A)...")
        self.generator_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
        self.generator_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to(self.device)
        print("Models loaded successfully.")
    
    def _fetch_real_time_metrics(self) -> Dict[str, Any]:
        """
        Fetch real-time metrics from the acquisition service.
        """
        try:
            response = requests.get(f"{self.endpoints['acquisition']}/realtime", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching real-time metrics: {e}")
        return {}
    
    def _fetch_aggregated_metrics(self, period: str = "last60minutes") -> Dict[str, Any]:
        """
        Fetch aggregated metrics from the acquisition service.
        """
        try:
            response = requests.get(f"{self.endpoints['acquisition']}/aggregated/{period}", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching aggregated metrics: {e}")
        return {}
    
    def _fetch_historical_data(self, start_date: str, end_date: str, time_unit: str = "day") -> Dict[str, Any]:
        """
        Fetch historical data from the acquisition service.
        """
        try:
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "timeUnit": time_unit,
                "buckets": "true"
            }
            response = requests.get(f"{self.endpoints['acquisition']}/historical", params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching historical data: {e}")
        return {}
    
    def _fetch_pdu_logs(self, start_time: int, end_time: int) -> Dict[str, Any]:
        """
        Fetch PDU logs from the acquisition service.
        """
        try:
            params = {
                "startTime": start_time,
                "endTime": end_time
            }
            response = requests.get(f"{self.endpoints['acquisition']}/realtime/logs", params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching PDU logs: {e}")
        return {}
    
    def _fetch_aggregated_pdu_logs(self, period_type: str = "today") -> Dict[str, Any]:
        """
        Fetch aggregated PDU logs using the built-in aggregation parameters.
        This makes optimal use of the /api/acquisition/realtime/logs endpoint with today=true, week=true, month=true.
        """
        try:
            # Use the built-in aggregation parameters for better performance
            params = {}
            if period_type == "today":
                params["today"] = "true"
            elif period_type == "week":
                params["week"] = "true"
            elif period_type == "month":
                params["month"] = "true"
            
            response = requests.get(f"{self.endpoints['acquisition']}/realtime/logs", params=params, timeout=20)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching aggregated PDU logs for {period_type}: {e}")
        return {}
    
    def _refresh_data_cache(self):
        """
        Refresh the data cache with latest information from all services.
        """
        current_time = datetime.now()
        
        # Check if cache is still valid
        if (self.cache_timestamp and 
            (current_time - self.cache_timestamp).seconds < self.cache_duration):
            return
        
        print("Refreshing data cache...")
        
        # Fetch real-time metrics
        self.data_cache['realtime'] = self._fetch_real_time_metrics()
        
        # Fetch aggregated metrics for different periods
        for period in ['last60minutes', 'last24hours', 'last7days']:
            self.data_cache[f'aggregated_{period}'] = self._fetch_aggregated_metrics(period)
        
        # Fetch recent historical data (last 7 days)
        end_date = current_time.strftime('%Y-%m-%d')
        start_date = (current_time - timedelta(days=7)).strftime('%Y-%m-%d')
        self.data_cache['historical_week'] = self._fetch_historical_data(start_date, end_date, "day")
        
        # Fetch PDU logs for different time periods for comprehensive analysis
        # Recent logs (last hour) for immediate insights
        end_timestamp = int(current_time.timestamp())
        start_timestamp = int((current_time - timedelta(hours=1)).timestamp())
        self.data_cache['recent_logs'] = self._fetch_pdu_logs(start_timestamp, end_timestamp)
        
        # Daily aggregated PDU logs using built-in today=true parameter
        self.data_cache['daily_pdu_logs'] = self._fetch_aggregated_pdu_logs("today")
        
        # Weekly aggregated PDU logs using built-in week=true parameter
        self.data_cache['weekly_pdu_logs'] = self._fetch_aggregated_pdu_logs("week")
        
        # Monthly aggregated PDU logs using built-in month=true parameter
        self.data_cache['monthly_pdu_logs'] = self._fetch_aggregated_pdu_logs("month")
        
        self.cache_timestamp = current_time
        
        # Rebuild vector store with fresh data
        self._build_vector_store()
    
    def _process_data_to_chunks(self) -> List[str]:
        """
        Process cached data into text chunks for the vector store.
        """
        chunks = []
        
        # Process real-time metrics
        if 'realtime' in self.data_cache and self.data_cache['realtime']:
            rt_data = self.data_cache['realtime']
            chunk = f"Real-time system status: {rt_data.get('pdusInLastSixtySeconds', 0)} PDUs received in last 60 seconds. "
            chunk += f"Average rate: {rt_data.get('averagePduRatePerSecondLastSixtySeconds', 0):.2f} PDUs/second. "
            chunk += f"Entity State PDUs: {rt_data.get('entityStatePdusInLastSixtySeconds', 0)}, "
            chunk += f"Fire Event PDUs: {rt_data.get('fireEventPdusInLastSixtySeconds', 0)}, "
            chunk += f"Collision PDUs: {rt_data.get('collisionPdusInLastSixtySeconds', 0)}."
            chunks.append(chunk)
        
        # Process aggregated metrics
        for period_key in ['aggregated_last60minutes', 'aggregated_last24hours', 'aggregated_last7days']:
            if period_key in self.data_cache and self.data_cache[period_key]:
                agg_data = self.data_cache[period_key]
                period_name = period_key.replace('aggregated_', '').replace('last', 'last ')
                chunk = f"Aggregated metrics for {period_name}: "
                chunk += f"Total packets: {agg_data.get('totalPackets', 0)}, "
                chunk += f"Entity State: {agg_data.get('entityStatePackets', 0)}, "
                chunk += f"Fire Events: {agg_data.get('fireEventPackets', 0)}, "
                chunk += f"Collisions: {agg_data.get('collisionPackets', 0)}, "
                chunk += f"Detonations: {agg_data.get('detonationPackets', 0)}. "
                chunk += f"Average rate: {agg_data.get('averagePacketsPerSecond', 0):.2f} packets/second."
                chunks.append(chunk)
        
        # Process historical data
        if 'historical_week' in self.data_cache and self.data_cache['historical_week']:
            hist_data = self.data_cache['historical_week']
            if 'buckets' in hist_data:
                for bucket in hist_data['buckets']:
                    date = bucket.get('date', 'Unknown date')
                    total = bucket.get('totalPackets', 0)
                    if total > 0:
                        chunk = f"On {date}, there were {total} total packets. "
                        chunk += f"Entity State PDUs: {bucket.get('entityStatePduCount', 0)}, "
                        chunk += f"Fire Event PDUs: {bucket.get('fireEventPduCount', 0)}, "
                        chunk += f"Collision PDUs: {bucket.get('collisionPduCount', 0)}, "
                        chunk += f"Detonation PDUs: {bucket.get('detonationPduCount', 0)}."
                        chunks.append(chunk)
                    else:
                        chunks.append(f"On {date}, there was no activity (0 total packets).")
        
        # Process recent PDU logs (last hour)
        if 'recent_logs' in self.data_cache and self.data_cache['recent_logs']:
            logs_data = self.data_cache['recent_logs']
            if 'Pdu_messages' in logs_data:
                pdu_types = {}
                for pdu in logs_data['Pdu_messages']:
                    pdu_type = pdu.get('pduType', 'Unknown')
                    pdu_types[pdu_type] = pdu_types.get(pdu_type, 0) + 1
                
                if pdu_types:
                    chunk = "Recent PDU activity in the last hour: "
                    chunk += ", ".join([f"{count} {pdu_type} PDUs" for pdu_type, count in pdu_types.items()])
                    chunks.append(chunk)
        
        # Process daily aggregated PDU logs (last 24 hours)
        if 'daily_pdu_logs' in self.data_cache and self.data_cache['daily_pdu_logs']:
            self._process_aggregated_logs(self.data_cache['daily_pdu_logs'], "daily (24 hours)", chunks)
        
        # Process weekly aggregated PDU logs (last 7 days)
        if 'weekly_pdu_logs' in self.data_cache and self.data_cache['weekly_pdu_logs']:
            self._process_aggregated_logs(self.data_cache['weekly_pdu_logs'], "weekly (7 days)", chunks)
        
        # Process monthly aggregated PDU logs (last 30 days)
        if 'monthly_pdu_logs' in self.data_cache and self.data_cache['monthly_pdu_logs']:
            self._process_aggregated_logs(self.data_cache['monthly_pdu_logs'], "monthly (30 days)", chunks)
        
        return chunks
    
    def _process_aggregated_logs(self, logs_data: Dict[str, Any], period_name: str, chunks: List[str]):
        """
        Process aggregated PDU logs data for daily, weekly, and monthly analysis.
        This provides comprehensive insights into PDU patterns and trends.
        """
        if not logs_data or 'Pdu_messages' not in logs_data:
            return
        
        pdu_messages = logs_data['Pdu_messages']
        if not pdu_messages:
            chunks.append(f"No PDU activity recorded for {period_name} period.")
            return
        
        # Analyze PDU types and their distribution
        pdu_type_counts = {}
        total_pdus = len(pdu_messages)
        
        # Count PDU types and analyze patterns
        for pdu in pdu_messages:
            pdu_type = pdu.get('pduType', 'Unknown')
            pdu_type_counts[pdu_type] = pdu_type_counts.get(pdu_type, 0) + 1
        
        # Create comprehensive analysis chunk
        if pdu_type_counts:
            chunk = f"PDU analysis for {period_name}: Total {total_pdus} PDUs recorded. "
            
            # Add distribution details
            distribution = []
            for pdu_type, count in sorted(pdu_type_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_pdus) * 100
                distribution.append(f"{pdu_type}: {count} ({percentage:.1f}%)")
            
            chunk += "Distribution: " + ", ".join(distribution[:5])  # Top 5 PDU types
            
            # Add trend insights
            if total_pdus > 100:
                chunk += f". High activity period with {total_pdus/24:.1f} PDUs per hour average" if '24 hours' in period_name else f". Active period with substantial PDU traffic"
            elif total_pdus > 10:
                chunk += ". Moderate activity period"
            else:
                chunk += ". Low activity period"
            
            chunks.append(chunk)
    
    def _build_vector_store(self):
        """
        Build the FAISS vector store from processed data chunks.
        """
        self.source_chunks = self._process_data_to_chunks()
        
        if not self.source_chunks:
            print("No data chunks available for vector store.")
            self.vector_store = None
            return
        
        print(f"Building vector store with {len(self.source_chunks)} chunks...")
        embeddings = self.retriever_model.encode(self.source_chunks, convert_to_tensor=False, show_progress_bar=False)
        
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings))
        
        self.vector_store = index
        print(f"Vector store built with {index.ntotal} vectors.")
    
    def _retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve the top-k most relevant text chunks for a given query.
        """
        if self.vector_store is None or self.vector_store.ntotal == 0:
            return []
        
        query_embedding = self.retriever_model.encode([query])
        distances, indices = self.vector_store.search(np.array(query_embedding), k)
        
        return [self.source_chunks[i] for i in indices[0] if i < len(self.source_chunks)]
    
    def _generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate an answer using the LLM based on the query and retrieved context.
        """
        if not context:
            return "I could not find any relevant information in the current DIS platform data to answer your question. Please try asking about PDU statistics, system metrics, or recent activity."
        
        prompt_template = """
Based on the following DIS platform data, please answer the question.
Provide a direct answer using only the information from the context.
If the context does not contain the answer, state that clearly.

DIS Platform Data:
{context}

Question:
{question}

Answer:
"""
        context_str = "\n".join(f"- {c}" for c in context)
        prompt = prompt_template.format(context=context_str, question=query)
        
        # Tokenize and generate the answer
        inputs = self.generator_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        outputs = self.generator_model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
        answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query and return an answer with context.
        """
        # Refresh data cache
        self._refresh_data_cache()
        
        # Retrieve relevant context
        retrieved_context = self._retrieve(user_query, k=5)
        
        # Generate answer
        answer = self._generate_answer(user_query, retrieved_context)
        
        return {
            "answer": answer,
            "context": retrieved_context,
            "data_sources": list(self.data_cache.keys()),
            "cache_timestamp": self.cache_timestamp.isoformat() if self.cache_timestamp else None
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and data availability.
        """
        self._refresh_data_cache()
        
        status = {
            "rag_system_ready": self.vector_store is not None,
            "data_sources_available": list(self.data_cache.keys()),
            "total_chunks": len(self.source_chunks),
            "cache_timestamp": self.cache_timestamp.isoformat() if self.cache_timestamp else None,
            "device": self.device
        }
        
        return status