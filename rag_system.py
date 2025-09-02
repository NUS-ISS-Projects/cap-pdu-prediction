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

    def __init__(self, base_url: str = "http://kong-gateway-service.default.svc.cluster.local:8000", jwt_token: str = None):
        """
        Initializes the RAG system with service endpoints.
        
        Args:
            base_url (str): Base URL for the DIS platform services
            jwt_token (str): JWT token for authentication (optional)
        """
        print("Initializing DIS Data RAG System...")
        self.base_url = base_url
        self.jwt_token = jwt_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Service endpoints
        self.endpoints = {
            "acquisition": f"{base_url}/api/acquisition",
            "ingestion": f"{base_url}/api/ingestion", 
            "processing": f"{base_url}/api/processing"
        }
        
        # JWT token will be set externally if needed
        
        # Initialize models
        self._load_models()
        
        # Initialize data cache
        self.data_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 2   # 2 seconds cache for real-time data
        
        # Vector store
        self.vector_store = None
        self.source_chunks = []
        

    
    def set_jwt_token(self, jwt_token: str):
        """
        Set or update the JWT token for authentication.
        
        Args:
            jwt_token (str): JWT token for authentication
        """
        was_token_empty = not self.jwt_token
        self.jwt_token = jwt_token
        print("JWT token updated for RAG system authentication")
        
        # If this is the first time setting a token, refresh data cache
        if was_token_empty and jwt_token:
            print("First JWT token set - triggering data refresh...")
            self._refresh_data_cache()
        
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
            headers = {}
            if self.jwt_token:
                headers['Authorization'] = f'Bearer {self.jwt_token}'
            
            url = f"{self.endpoints['acquisition']}/realtime"
            print(f"Fetching real-time metrics from: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            print(f"Real-time metrics response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Real-time metrics data received: {len(str(data))} chars")
                return data
            elif response.status_code == 401:
                print("Authentication required for real-time metrics endpoint")
            else:
                print(f"Real-time metrics request failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error fetching real-time metrics: {e}")
        return {}
    
    def _fetch_aggregated_metrics(self, period_type: str = "today") -> Dict[str, Any]:
        """
        Fetch aggregated metrics from the acquisition service using the correct aggregate endpoint.
        """
        try:
            headers = {}
            if self.jwt_token:
                headers['Authorization'] = f'Bearer {self.jwt_token}'
            
            # Use the correct aggregate endpoint with proper parameters
            current_date = datetime.now().strftime('%Y-%m-%d')
            params = {"startDate": current_date}
            
            if period_type == "week":
                params["week"] = "true"
            elif period_type == "month":
                params["month"] = "true"
            # For "today", just use startDate without additional parameters
            
            response = requests.get(f"{self.endpoints['acquisition']}/aggregate", params=params, headers=headers, timeout=15)
            print(f"Aggregated metrics request for {period_type}: {response.url}, status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print(f"Authentication required for aggregated metrics endpoint: {period_type}")
            else:
                print(f"Aggregated metrics request failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error fetching aggregated metrics for {period_type}: {e}")
        return {}
    
    def _fetch_historical_data(self, start_date: str, end_date: str, time_unit: str = "day") -> Dict[str, Any]:
        """
        Fetch historical data from the acquisition service.
        """
        try:
            headers = {}
            if self.jwt_token:
                headers['Authorization'] = f'Bearer {self.jwt_token}'
            
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "timeUnit": time_unit,
                "buckets": "true"
            }
            response = requests.get(f"{self.endpoints['acquisition']}/historical", params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("Authentication required for historical data endpoint")
        except Exception as e:
            print(f"Error fetching historical data: {e}")
        return {}
    
    def _fetch_pdu_logs(self, start_time: int, end_time: int) -> Dict[str, Any]:
        """
        Fetch PDU logs from the acquisition service.
        """
        try:
            headers = {}
            if self.jwt_token:
                headers['Authorization'] = f'Bearer {self.jwt_token}'
            
            params = {
                "startTime": start_time,
                "endTime": end_time
            }
            response = requests.get(f"{self.endpoints['acquisition']}/realtime/logs", params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("Authentication required for PDU logs endpoint")
        except Exception as e:
            print(f"Error fetching PDU logs: {e}")
        return {}
    
    def _fetch_aggregated_pdu_logs(self, period_type: str = "today") -> Dict[str, Any]:
        """
        Fetch aggregated PDU logs using the /aggregate endpoint with query parameters.
        """
        try:
            headers = {}
            if self.jwt_token:
                headers['Authorization'] = f'Bearer {self.jwt_token}'
            
            # Calculate start date for the query
            current_time = datetime.now()
            start_date = current_time.strftime('%Y-%m-%d')
            
            # Build query parameters based on period type
            params = {
                'startDate': start_date
            }
            
            if period_type == "today":
                params['today'] = 'true'
            elif period_type == "week":
                params['week'] = 'true'
            elif period_type == "month":
                params['month'] = 'true'
            
            url = f"{self.endpoints['acquisition']}/aggregate"
            print(f"Fetching aggregated PDU logs ({period_type}) from: {url} with params: {params}")
            response = requests.get(url, params=params, headers=headers, timeout=20)
            print(f"Aggregated PDU logs ({period_type}) response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Aggregated PDU logs ({period_type}) data received: {len(str(data))} chars")
                return data
            elif response.status_code == 401:
                print(f"Authentication required for aggregated PDU logs endpoint: {period_type}")
            else:
                print(f"Failed to fetch aggregated PDU logs ({period_type}): {response.status_code} - {response.text[:200]}")
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
        realtime_data = self._fetch_real_time_metrics()
        self.data_cache['realtime'] = realtime_data
        print(f"Real-time data fetched: {bool(realtime_data)}, keys: {list(realtime_data.keys()) if realtime_data else 'None'}")
        
        # Fetch aggregated metrics for different periods using the correct endpoint
        for period_type in ['today', 'week', 'month']:
            agg_data = self._fetch_aggregated_metrics(period_type)
            self.data_cache[f'aggregated_{period_type}'] = agg_data
            print(f"Aggregated data for {period_type}: {bool(agg_data)}, keys: {list(agg_data.keys()) if agg_data else 'None'}")
        
        # Fetch recent historical data (last 7 days)
        end_date = current_time.strftime('%Y-%m-%d')
        start_date = (current_time - timedelta(days=7)).strftime('%Y-%m-%d')
        self.data_cache['historical_week'] = self._fetch_historical_data(start_date, end_date, "day")
        
        # Fetch aggregated PDU logs using /aggregate/logs endpoints only
        # Daily aggregated PDU logs using /aggregate/logs/today endpoint
        daily_logs = self._fetch_aggregated_pdu_logs("today")
        self.data_cache['daily_pdu_logs'] = daily_logs
        print(f"Daily PDU logs: {bool(daily_logs)}, keys: {list(daily_logs.keys()) if daily_logs else 'None'}")
        
        # Weekly aggregated PDU logs using /aggregate/logs/week endpoint
        weekly_logs = self._fetch_aggregated_pdu_logs("week")
        self.data_cache['weekly_pdu_logs'] = weekly_logs
        print(f"Weekly PDU logs: {bool(weekly_logs)}, keys: {list(weekly_logs.keys()) if weekly_logs else 'None'}")
        
        # Monthly aggregated PDU logs using /aggregate/logs/month endpoint
        monthly_logs = self._fetch_aggregated_pdu_logs("month")
        self.data_cache['monthly_pdu_logs'] = monthly_logs
        print(f"Monthly PDU logs: {bool(monthly_logs)}, keys: {list(monthly_logs.keys()) if monthly_logs else 'None'}")
        
        self.cache_timestamp = current_time
        
        # Rebuild vector store with fresh data
        self._build_vector_store()
    
    def _process_data_to_chunks(self) -> List[str]:
        """
        Process cached data into text chunks for the vector store.
        """
        chunks = []
        print(f"Processing data to chunks. Cache keys: {list(self.data_cache.keys())}")
        print(f"Cache contents summary: {[(k, bool(v)) for k, v in self.data_cache.items()]}")
        
        # Process real-time metrics
        if 'realtime' in self.data_cache and self.data_cache['realtime']:
            rt_data = self.data_cache['realtime']
            chunk = f"Real-time system status: {rt_data.get('pdusInLastSixtySeconds', 0)} PDUs received in last 60 seconds. "
            chunk += f"Average rate: {rt_data.get('averagePduRatePerSecondLastSixtySeconds', 0):.2f} PDUs/second. "
            
            # Include all available PDU types from real-time data
            pdu_types = [
                ('Entity State PDUs', 'entityStatePdusInLastSixtySeconds'),
                ('Fire Event PDUs', 'fireEventPdusInLastSixtySeconds'),
                ('Collision PDUs', 'collisionPdusInLastSixtySeconds'),
                ('Detonation PDUs', 'detonationPdusInLastSixtySeconds'),
                ('Data PDUs', 'dataPdusInLastSixtySeconds'),
                ('Action Request PDUs', 'actionRequestPdusInLastSixtySeconds'),
                ('Start/Resume PDUs', 'startResumePdusInLastSixtySeconds'),
                ('Set Data PDUs', 'setDataPdusInLastSixtySeconds'),
                ('Designator PDUs', 'designatorPdusInLastSixtySeconds'),
                ('Electromagnetic Emissions PDUs', 'electromagneticEmissionsPdusInLastSixtySeconds')
            ]
            
            pdu_details = []
            for pdu_name, pdu_key in pdu_types:
                count = rt_data.get(pdu_key, 0)
                if count > 0:  # Only include PDU types with activity
                    pdu_details.append(f"{pdu_name}: {count}")
            
            if pdu_details:
                chunk += ", ".join(pdu_details) + "."
            else:
                chunk += "No specific PDU type activity detected."
            
            chunks.append(chunk)
        
        # Process aggregated metrics
        for period_key in ['aggregated_today', 'aggregated_week', 'aggregated_month']:
            if period_key in self.data_cache and self.data_cache[period_key]:
                agg_data = self.data_cache[period_key]
                period_name = period_key.replace('aggregated_', '')
                
                # Process buckets from the aggregate endpoint response
                if 'buckets' in agg_data:
                    time_unit = agg_data.get('timeUnit', 'unknown')
                    total_packets = sum(bucket.get('totalPackets', 0) for bucket in agg_data['buckets'])
                    
                    if total_packets > 0:
                        chunk = f"Aggregated metrics for {period_name} ({time_unit} view): "
                        chunk += f"Total packets: {total_packets}, "
                        
                        # Sum up all available PDU types across all buckets
                        pdu_type_mappings = [
                            ('Entity State PDUs', 'entityStatePduCount'),
                            ('Fire Event PDUs', 'fireEventPduCount'),
                            ('Collision PDUs', 'collisionPduCount'),
                            ('Detonation PDUs', 'detonationPduCount'),
                            ('Data PDUs', 'dataPduCount'),
                            ('Action Request PDUs', 'actionRequestPduCount'),
                            ('Start/Resume PDUs', 'startResumePduCount'),
                            ('Set Data PDUs', 'setDataPduCount'),
                            ('Designator PDUs', 'designatorPduCount'),
                            ('Electromagnetic Emissions PDUs', 'electromagneticEmissionsPduCount')
                        ]
                        
                        pdu_details = []
                        for pdu_name, pdu_key in pdu_type_mappings:
                            count = sum(bucket.get(pdu_key, 0) for bucket in agg_data['buckets'])
                            if count > 0:  # Only include PDU types with activity
                                pdu_details.append(f"{pdu_name}: {count}")
                        
                        if pdu_details:
                            chunk += ", ".join(pdu_details) + "."
                        else:
                            chunk += "No specific PDU type activity detected."
                        chunks.append(chunk)
                    else:
                        chunks.append(f"No activity recorded for {period_name} period.")
        
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
        
        # For PDU statistics queries, provide a structured summary directly from context
        if any(keyword in query.lower() for keyword in ['pdu', 'statistics', 'types', 'count', 'metrics']):
            return self._generate_structured_pdu_summary(context, query)
        
        prompt_template = """
Based on the DIS platform data below, provide a clear and concise answer.
Use only the information provided. Be specific and avoid repetition.

Data:
{context}

Question: {question}

Answer:"""
        context_str = "\n".join(f"- {c}" for c in context)
        prompt = prompt_template.format(context=context_str, question=query)
        
        # Tokenize and generate the answer
        inputs = self.generator_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.generator_model.generate(
            **inputs,
            max_length=150,
            num_beams=3,
            early_stopping=True,
            do_sample=False,
            repetition_penalty=1.2
        )
        answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the answer by removing the prompt
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    
    def _generate_structured_pdu_summary(self, context: List[str], query: str = "") -> str:
        """
        Generate a structured summary for PDU statistics queries.
        """
        # Extract PDU information from context - prioritize aggregated data over real-time
        aggregated_pdu_data = {}
        realtime_pdu_data = {}
        total_packets = 0
        realtime_total = 0
        
        for ctx in context:
            if "Aggregated metrics" in ctx and "Total packets:" in ctx:
                # Extract aggregated data (prioritized)
                parts = ctx.split(", ")
                for part in parts:
                    if "Total packets:" in part:
                        try:
                            # Extract the number after "Total packets: "
                            total_str = part.split("Total packets: ")[1].rstrip('.,;')
                            total_packets = max(total_packets, int(total_str))
                        except (ValueError, IndexError):
                            continue
                    elif " PDUs:" in part:
                        try:
                            pdu_type, count_str = part.split(": ")
                            count_str = count_str.rstrip('.,;')
                            # Use max to get the highest count across different aggregated periods
                            aggregated_pdu_data[pdu_type] = max(aggregated_pdu_data.get(pdu_type, 0), int(count_str))
                        except (ValueError, IndexError):
                            continue
            elif "Real-time system status:" in ctx:
                # Extract real-time data (fallback)
                parts = ctx.split(", ")
                for part in parts:
                    if " PDUs:" in part:
                        try:
                            pdu_type, count_str = part.split(": ")
                            count_str = count_str.rstrip('.,;')
                            realtime_pdu_data[pdu_type] = int(count_str)
                        except (ValueError, IndexError):
                            continue
                    elif "PDUs received" in part:
                        try:
                            realtime_total = int(part.split()[0])
                        except (ValueError, IndexError):
                            continue
        
        # Determine query intent
        query_wants_totals = any(keyword in query.lower() for keyword in ['total', 'today', 'daily']) if query else False
        query_wants_realtime = any(keyword in query.lower() for keyword in ['real-time', 'realtime', 'current', 'live', 'now', 'activity']) if query else False
        
        # Determine response based on query intent and available data
        
        # Prioritize real-time data when query explicitly asks for real-time information
        if query_wants_realtime:
            # Check if we have real-time context (even if no PDUs are active)
            has_realtime_context = any("Real-time system status:" in ctx for ctx in context)
            if has_realtime_context:
                if realtime_total > 0:
                    response = f"Current real-time activity: {realtime_total} PDUs received in the last 60 seconds. "
                    if realtime_pdu_data:
                        pdu_list = [f"{pdu_type}: {count}" for pdu_type, count in sorted(realtime_pdu_data.items())]
                        response += "PDU breakdown: " + ", ".join(pdu_list) + "."
                else:
                    response = "Real-time PDU activity: No PDUs detected in the last 60 seconds. System is currently inactive."
            else:
                # No real-time data available, inform user
                response = "Real-time data is currently unavailable. Please try again later."
        # Prioritize aggregated data when query asks for totals
        elif aggregated_pdu_data and total_packets > 0 and query_wants_totals:
            response = f"Total PDU statistics for today: {total_packets} total packets. "
            pdu_list = [f"{pdu_type}: {count}" for pdu_type, count in sorted(aggregated_pdu_data.items())]
            response += "PDU breakdown: " + ", ".join(pdu_list) + "."
        # Fallback to aggregated data if available and no specific real-time request
        elif aggregated_pdu_data and total_packets > 0:
            response = f"Total PDU statistics: {total_packets} total packets. "
            pdu_list = [f"{pdu_type}: {count}" for pdu_type, count in sorted(aggregated_pdu_data.items())]
            response += "PDU breakdown: " + ", ".join(pdu_list) + "."
        # Fallback to real-time data if no aggregated data
        elif realtime_pdu_data:
            if realtime_total > 0:
                response = f"Current system shows {realtime_total} PDUs received in the last 60 seconds. "
            else:
                response = "Real-time PDU statistics: "
            pdu_list = [f"{pdu_type}: {count}" for pdu_type, count in sorted(realtime_pdu_data.items())]
            response += "PDU breakdown: " + ", ".join(pdu_list) + "."
        else:
            response = "No PDU statistics available."
        
        return response
    
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