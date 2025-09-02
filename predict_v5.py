import json
import numpy as np
import requests
import os
import time
import hashlib
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration for RAG system
DATA_ACQUISITION_URL = os.getenv('DATA_ACQUISITION_URL', 'http://data-acquisition-service:8080')
DATA_INGESTION_URL = os.getenv('DATA_INGESTION_URL', 'http://data-ingestion-service:8080')
DATA_PROCESSING_URL = os.getenv('DATA_PROCESSING_URL', 'http://data-processing-service:8080')
USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
# Optimized model selection for fast inference
MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/DialoGPT-small')  # Lightweight model for speed

# Performance optimization settings
USE_CACHING = os.getenv('USE_CACHING', 'true').lower() == 'true'
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '300'))  # 5 minutes default
MAX_RESPONSE_TIME = float(os.getenv('MAX_RESPONSE_TIME', '3.0'))  # 3 seconds max

def predict_pdu_data(data):
    """
    Processes PDU data, predicts future values, and returns the prediction.

    Args:
        data (dict): The PDU data in a dictionary format.

    Returns:
        dict: A dictionary containing the predicted labels and values, or an error message.
    """
    time_unit = data.get("timeUnit")
    buckets = data.get("buckets", [])

    if not time_unit or not buckets:
        return {"error": "JSON data must include 'timeUnit' and 'buckets'."}

    pdu_fields = [
        "entityStatePduCount", "fireEventPduCount", "collisionPduCount",
        "detonationPduCount", "dataPduCount", "actionRequestPduCount",
        "startResumePduCount", "setDataPduCount", "designatorPduCount",
        "electromagneticEmissionsPduCount"
    ]

    historical_y_values = []
    historical_x_labels = []

    for bucket in buckets:
        total_pdus = sum(bucket.get(field, 0) for field in pdu_fields)
        historical_y_values.append(total_pdus)
        if time_unit == 'hour':
            historical_x_labels.append(bucket.get('hour'))
        elif time_unit == 'day':
            historical_x_labels.append(bucket.get('date'))
        elif time_unit == 'week':
            week_str = bucket.get('week', '')
            if week_str:
                historical_x_labels.append(week_str.split(' ')[0] + " " + week_str.split(' ')[1])

    if not historical_x_labels:
        return {"error": "No data points were found to analyze."}

    num_historical_points = len(historical_x_labels)
    num_to_predict = num_historical_points

    # --- Prediction using Linear Regression ---
    X_historical = np.arange(num_historical_points).reshape(-1, 1)
    y_historical = np.array(historical_y_values)

    model = LinearRegression()
    model.fit(X_historical, y_historical)

    future_x_indices = np.arange(num_historical_points, num_historical_points + num_to_predict).reshape(-1, 1)
    predicted_y = model.predict(future_x_indices)
    
    predicted_x_labels = [f"Pred. Point {i+1}" for i in range(num_to_predict)]

    return {
        "predicted_labels": predicted_x_labels,
        "predicted_values": predicted_y.tolist()
    }

class RAGSystem:
    """Retrieval-Augmented Generation system for accessing DIS platform data"""
    
    def __init__(self):
        self.endpoints = {
            'entity_states': f'{DATA_ACQUISITION_URL}/api/acquisition/entity-states',
            'fire_events': f'{DATA_ACQUISITION_URL}/api/acquisition/fire-events',
            'collision_events': f'{DATA_ACQUISITION_URL}/api/acquisition/collision-events',
            'detonation_events': f'{DATA_ACQUISITION_URL}/api/acquisition/detonation-events',
            'metrics': f'{DATA_ACQUISITION_URL}/api/acquisition/metrics',
            'realtime': f'{DATA_ACQUISITION_URL}/api/acquisition/realtime',
            'aggregate': f'{DATA_ACQUISITION_URL}/api/acquisition/aggregate',
            'monthly': f'{DATA_ACQUISITION_URL}/api/acquisition/monthly',
            'realtime_logs': f'{DATA_ACQUISITION_URL}/api/acquisition/realtime/logs',
            'ingestion_health': f'{DATA_INGESTION_URL}/api/ingestion/health',
            'processing_health': f'{DATA_PROCESSING_URL}/api/processing/health'
        }
    
    def fetch_data(self, endpoint_key, params=None, auth_token=None):
        """Fetch data from specified endpoint"""
        try:
            url = self.endpoints.get(endpoint_key)
            if not url:
                return {"error": f"Unknown endpoint: {endpoint_key}"}
            
            headers = {}
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.RequestException as e:
            logger.error(f"Error fetching data from {endpoint_key}: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}
    
    def get_system_overview(self, auth_token=None):
        """Get comprehensive system overview"""
        overview = {
            "services_health": {},
            "recent_metrics": {},
            "system_status": "operational"
        }
        
        # Check service health
        health_endpoints = ['ingestion_health', 'processing_health']
        for endpoint in health_endpoints:
            health_data = self.fetch_data(endpoint, auth_token=auth_token)
            overview["services_health"][endpoint] = health_data
        
        # Get recent metrics
        metrics_data = self.fetch_data('metrics', auth_token=auth_token)
        overview["recent_metrics"] = metrics_data
        
        return overview
    
    def search_pdu_data(self, query_type, time_range=None, auth_token=None):
        """Search PDU data based on query type"""
        if query_type == "recent_activity":
            return self.fetch_data('realtime', auth_token=auth_token)
        elif query_type == "today_summary":
            params = {'today': 'true'}
            return self.fetch_data('aggregate', params=params, auth_token=auth_token)
        elif query_type == "today_logs":
            # Use realtime/logs for detailed today's data
            current_time = int(datetime.now().timestamp() * 1000)
            start_of_day = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            params = {'startTime': start_of_day, 'endTime': current_time}
            return self.fetch_data('realtime_logs', params=params, auth_token=auth_token)
        elif query_type == "weekly_logs":
            # Use realtime/logs for weekly aggregated data
            current_time = int(datetime.now().timestamp() * 1000)
            week_ago = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            params = {'startTime': week_ago, 'endTime': current_time}
            return self.fetch_data('realtime_logs', params=params, auth_token=auth_token)
        elif query_type == "monthly_logs":
            # Use realtime/logs for monthly aggregated data
            current_time = int(datetime.now().timestamp() * 1000)
            month_ago = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            params = {'startTime': month_ago, 'endTime': current_time}
            return self.fetch_data('realtime_logs', params=params, auth_token=auth_token)
        elif query_type == "entity_states":
            return self.fetch_data('entity_states', auth_token=auth_token)
        elif query_type == "fire_events":
            return self.fetch_data('fire_events', auth_token=auth_token)
        elif query_type == "collision_events":
            return self.fetch_data('collision_events', auth_token=auth_token)
        elif query_type == "detonation_events":
            return self.fetch_data('detonation_events', auth_token=auth_token)
        else:
            return {"error": f"Unknown query type: {query_type}"}

class PerformanceMonitor:
    """Monitor chatbot performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
    
    def record_response_time(self, response_time):
        self.response_times.append(response_time)
        # Keep only last 100 measurements
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_cache_hit(self):
        self.cache_hits += 1
        self.total_requests += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
        self.total_requests += 1
    
    def get_avg_response_time(self):
        return np.mean(self.response_times) if self.response_times else 0
    
    def get_cache_hit_rate(self):
        if self.total_requests == 0:
            return 0
        return (self.cache_hits / self.total_requests) * 100

class SimpleLLMChatbot:
    """Simple rule-based chatbot with LLM-like responses for PDU data queries"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.conversation_context = []
        self.response_cache = {} if USE_CACHING else None
        self.performance_monitor = PerformanceMonitor()
        logger.info(f"Chatbot initialized with caching: {USE_CACHING}")
    
    def _get_cache_key(self, user_query, auth_token=None):
        """Generate cache key for query"""
        # Create cache key from query + user context
        cache_input = f"{user_query.lower().strip()}_{auth_token[:10] if auth_token else 'anon'}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_item):
        """Check if cached response is still valid"""
        if not cached_item:
            return False
        return (time.time() - cached_item['timestamp']) < CACHE_TTL_SECONDS
    
    def _get_cached_response(self, cache_key):
        """Get response from cache if valid"""
        if not USE_CACHING or not self.response_cache:
            return None
        
        cached_item = self.response_cache.get(cache_key)
        if cached_item and self._is_cache_valid(cached_item):
            self.performance_monitor.record_cache_hit()
            logger.info(f"Cache hit for key: {cache_key[:8]}...")
            return cached_item['response']
        
        self.performance_monitor.record_cache_miss()
        return None
    
    def _cache_response(self, cache_key, response):
        """Cache response for future use"""
        if not USE_CACHING or not self.response_cache:
            return
        
        # Limit cache size to prevent memory issues
        if len(self.response_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self.response_cache.keys(), 
                            key=lambda k: self.response_cache[k]['timestamp'])
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        logger.info(f"Cached response for key: {cache_key[:8]}...")
    
    def process_query(self, user_query, auth_token=None):
        """Process user query and generate response with performance optimizations"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(user_query, auth_token)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                response_time = time.time() - start_time
                self.performance_monitor.record_response_time(response_time)
                return cached_response
            
            user_query_lower = user_query.lower()
            
            # Store current query for context-aware processing
            self._current_query = user_query
            
            # Intent classification based on keywords
            if any(keyword in user_query_lower for keyword in ['health', 'status', 'system', 'overview']):
                response = self._handle_system_status_query(auth_token)
            elif any(keyword in user_query_lower for keyword in ['predict', 'prediction', 'forecast']):
                response = self._handle_prediction_query(user_query, auth_token)
            elif any(keyword in user_query_lower for keyword in ['entity', 'entities']):
                response = self._handle_entity_query(auth_token)
            elif any(keyword in user_query_lower for keyword in ['fire', 'firing']):
                response = self._handle_fire_events_query(auth_token)
            elif any(keyword in user_query_lower for keyword in ['collision', 'crash']):
                response = self._handle_collision_query(auth_token)
            elif any(keyword in user_query_lower for keyword in ['detonation', 'explosion']):
                response = self._handle_detonation_query(auth_token)
            elif any(keyword in user_query_lower for keyword in ['week', 'weekly', 'month', 'monthly', 'today', 'current', 'now', 'recent', 'activity', 'logs']):
                response = self._handle_recent_activity_query(auth_token)
            elif any(keyword in user_query_lower for keyword in ['metrics', 'statistics', 'stats']):
                response = self._handle_metrics_query(auth_token)
            elif any(keyword in user_query_lower for keyword in ['help', 'commands', 'what can you do']):
                response = self._handle_help_query()
            else:
                response = self._handle_general_query(user_query, auth_token)
            
            # Cache the response
            self._cache_response(cache_key, response)
            
            # Record performance metrics
            response_time = time.time() - start_time
            self.performance_monitor.record_response_time(response_time)
            
            # Add performance metadata
            response['performance'] = {
                'response_time': round(response_time, 3),
                'cached': False,
                'cache_hit_rate': round(self.performance_monitor.get_cache_hit_rate(), 1)
            }
            
            # Log slow responses
            if response_time > MAX_RESPONSE_TIME:
                logger.warning(f"Slow response: {response_time:.3f}s for query: {user_query[:50]}...")
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self.performance_monitor.record_response_time(response_time)
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": "🚨 **System Error**\n\nI encountered an error processing your request. Please try again or contact support if the issue persists.",
                "data": {"error": str(e)},
                "intent": "error",
                "performance": {
                    "response_time": round(response_time, 3),
                    "cached": False,
                    "error": True
                }
            }
    
    def _handle_system_status_query(self, auth_token):
        """Handle system status queries"""
        overview = self.rag_system.get_system_overview(auth_token)
        
        response = "🔍 **System Status Overview**\n\n"
        
        # Service health summary
        health_status = overview.get('services_health', {})
        healthy_services = 0
        total_services = len(health_status)
        
        for service, health in health_status.items():
            if isinstance(health, dict) and health.get('status') in ['UP', 'healthy']:
                healthy_services += 1
        
        response += f"**Services Health:** {healthy_services}/{total_services} services operational\n"
        
        # Recent metrics summary
        metrics = overview.get('recent_metrics', {})
        if metrics and not metrics.get('error'):
            response += "**Recent Activity:** System is actively processing PDU data\n"
        else:
            response += "**Recent Activity:** Limited data available\n"
        
        response += "\n💡 *Ask me about specific PDU events, predictions, or system metrics for more details.*"
        
        return {
            "response": response,
            "data": overview,
            "intent": "system_status"
        }
    
    def _handle_prediction_query(self, user_query, auth_token):
        """Handle prediction-related queries"""
        response = "🔮 **PDU Prediction Capabilities**\n\n"
        response += "I can help you with PDU predictions! Here's what I can do:\n\n"
        response += "• **Predict future PDU counts** based on historical data\n"
        response += "• **Analyze trends** in entity states, fire events, collisions, and detonations\n"
        response += "• **Forecast system load** for capacity planning\n\n"
        response += "📊 To get predictions, send PDU data in this format:\n"
        response += "```json\n"
        response += "{\n"
        response += '  "timeUnit": "hour",\n'
        response += '  "buckets": [\n'
        response += '    {\n'
        response += '      "hour": "2024-01-01T10:00:00Z",\n'
        response += '      "entityStatePduCount": 150,\n'
        response += '      "fireEventPduCount": 25\n'
        response += '    }\n'
        response += '  ]\n'
        response += "}\n```\n\n"
        response += "💡 *Use the `/api/prediction` endpoint for actual predictions.*"
        
        return {
            "response": response,
            "data": {"prediction_help": True},
            "intent": "prediction_help"
        }
    
    def _handle_entity_query(self, auth_token):
        """Handle entity-related queries"""
        entity_data = self.rag_system.search_pdu_data('entity_states', auth_token=auth_token)
        
        response = "🎯 **Entity State Information**\n\n"
        
        if entity_data.get('error'):
            response += f"⚠️ Unable to retrieve entity data: {entity_data['error']}\n\n"
            response += "This might be due to:\n"
            response += "• Authentication required\n"
            response += "• Service temporarily unavailable\n"
            response += "• No recent entity state data\n"
        else:
            response += "📊 Successfully retrieved entity state data.\n\n"
            if isinstance(entity_data, list):
                response += f"**Total Entities:** {len(entity_data)}\n"
            elif isinstance(entity_data, dict) and 'count' in entity_data:
                response += f"**Entity Count:** {entity_data['count']}\n"
        
        response += "\n💡 *Entity states track the position and status of simulation objects.*"
        
        return {
            "response": response,
            "data": entity_data,
            "intent": "entity_query"
        }
    
    def _handle_fire_events_query(self, auth_token):
        """Handle fire events queries"""
        fire_data = self.rag_system.search_pdu_data('fire_events', auth_token=auth_token)
        
        response = "🔥 **Fire Events Information**\n\n"
        
        if fire_data.get('error'):
            response += f"⚠️ Unable to retrieve fire events: {fire_data['error']}\n"
        else:
            response += "📊 Successfully retrieved fire events data.\n"
            if isinstance(fire_data, list):
                response += f"**Recent Fire Events:** {len(fire_data)}\n"
        
        response += "\n💡 *Fire events represent weapon discharge and munition firing in the simulation.*"
        
        return {
            "response": response,
            "data": fire_data,
            "intent": "fire_events"
        }
    
    def _handle_collision_query(self, auth_token):
        """Handle collision events queries"""
        collision_data = self.rag_system.search_pdu_data('collision_events', auth_token=auth_token)
        
        response = "💥 **Collision Events Information**\n\n"
        
        if collision_data.get('error'):
            response += f"⚠️ Unable to retrieve collision data: {collision_data['error']}\n"
        else:
            response += "📊 Successfully retrieved collision events data.\n"
            if isinstance(collision_data, list):
                response += f"**Recent Collisions:** {len(collision_data)}\n"
        
        response += "\n💡 *Collision events track when entities physically impact each other.*"
        
        return {
            "response": response,
            "data": collision_data,
            "intent": "collision_events"
        }
    
    def _handle_detonation_query(self, auth_token):
        """Handle detonation events queries"""
        detonation_data = self.rag_system.search_pdu_data('detonation_events', auth_token=auth_token)
        
        response = "💣 **Detonation Events Information**\n\n"
        
        if detonation_data.get('error'):
            response += f"⚠️ Unable to retrieve detonation data: {detonation_data['error']}\n"
        else:
            response += "📊 Successfully retrieved detonation events data.\n"
            if isinstance(detonation_data, list):
                response += f"**Recent Detonations:** {len(detonation_data)}\n"
        
        response += "\n💡 *Detonation events represent explosive impacts and munition detonations.*"
        
        return {
            "response": response,
            "data": detonation_data,
            "intent": "detonation_events"
        }
    
    def _handle_recent_activity_query(self, auth_token):
        """Handle recent activity queries"""
        user_query_lower = getattr(self, '_current_query', '').lower()
        
        # Determine time scope based on query keywords
        if any(keyword in user_query_lower for keyword in ['week', 'weekly', 'last week']):
            time_scope = 'weekly_logs'
            title = "📈 **Weekly Activity Summary**"
            period = "the past 7 days"
        elif any(keyword in user_query_lower for keyword in ['month', 'monthly', 'last month']):
            time_scope = 'monthly_logs'
            title = "📈 **Monthly Activity Summary**"
            period = "the past 30 days"
        else:
            time_scope = 'today_logs'
            title = "📈 **Today's Activity Summary**"
            period = "today"
        
        # Try detailed logs first, fallback to aggregate
        activity_data = self.rag_system.search_pdu_data(time_scope, auth_token=auth_token)
        
        response = f"{title}\n\n"
        
        if activity_data.get('error'):
            # Fallback to aggregate endpoint for today
            if time_scope == 'today_logs':
                fallback_data = self.rag_system.search_pdu_data('today_summary', auth_token=auth_token)
                if not fallback_data.get('error'):
                    activity_data = fallback_data
                    response += "📊 Retrieved aggregated summary data.\n\n"
                else:
                    response += f"⚠️ Unable to retrieve activity data: {activity_data['error']}\n"
            else:
                response += f"⚠️ Unable to retrieve {period} data: {activity_data['error']}\n"
        else:
            response += f"📊 Successfully retrieved detailed activity logs for {period}.\n\n"
            
            # Process realtime logs data
            if isinstance(activity_data, dict) and 'pdu_messages' in activity_data:
                pdu_messages = activity_data.get('pdu_messages', [])
                response += f"**Total PDU Messages:** {len(pdu_messages)}\n"
                
                # Analyze PDU types
                pdu_types = {}
                for msg in pdu_messages:
                    pdu_type = msg.get('pdu_type', 'Unknown')
                    pdu_types[pdu_type] = pdu_types.get(pdu_type, 0) + 1
                
                if pdu_types:
                    response += "\n**PDU Type Breakdown:**\n"
                    for pdu_type, count in sorted(pdu_types.items(), key=lambda x: x[1], reverse=True):
                        response += f"• {pdu_type}: {count} messages\n"
            
            # Try to extract meaningful information from other formats
            elif isinstance(activity_data, dict):
                for key, value in activity_data.items():
                    if 'count' in key.lower() or 'total' in key.lower():
                        response += f"**{key.replace('_', ' ').title()}:** {value}\n"
        
        response += f"\n💡 *This shows detailed PDU activity logs for {period}.*"
        
        return {
            "response": response,
            "data": activity_data,
            "intent": "recent_activity",
            "time_scope": time_scope
        }
    
    def _handle_metrics_query(self, auth_token):
        """Handle metrics queries"""
        metrics_data = self.rag_system.search_pdu_data('recent_activity', auth_token=auth_token)
        
        response = "📊 **System Metrics**\n\n"
        
        if metrics_data.get('error'):
            response += f"⚠️ Unable to retrieve metrics: {metrics_data['error']}\n"
        else:
            response += "📈 Successfully retrieved system metrics.\n\n"
            
            # Extract key metrics
            if isinstance(metrics_data, dict):
                response += "**Key Performance Indicators:**\n"
                for key, value in metrics_data.items():
                    if isinstance(value, (int, float)):
                        response += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        response += "\n💡 *Metrics help monitor system performance and PDU processing rates.*"
        
        return {
            "response": response,
            "data": metrics_data,
            "intent": "metrics"
        }
    
    def _handle_help_query(self):
        """Handle help queries"""
        response = "🤖 **DIS Platform Chatbot Help**\n\n"
        response += "I'm your intelligent assistant for the DIS (Distributed Interactive Simulation) platform. Here's what I can help you with:\n\n"
        response += "**📊 Data Queries:**\n"
        response += "• System status and health checks\n"
        response += "• Entity state information\n"
        response += "• Fire events and weapon discharge data\n"
        response += "• Collision and impact events\n"
        response += "• Detonation and explosive events\n"
        response += "• Real-time metrics and statistics\n"
        response += "• **Detailed activity logs** (today, weekly, monthly)\n\n"
        response += "**🔮 Predictions:**\n"
        response += "• PDU count forecasting\n"
        response += "• Trend analysis\n"
        response += "• System load predictions\n\n"
        response += "**📈 Time-based Queries:**\n"
        response += "• \"What happened today?\" - Detailed today's PDU logs\n"
        response += "• \"Show me this week's activity\" - 7-day aggregated data\n"
        response += "• \"Monthly summary\" - 30-day activity overview\n"
        response += "• \"Recent logs\" - Latest PDU message details\n\n"
        response += "**💬 Example Queries:**\n"
        response += "• \"What's the system status?\"\n"
        response += "• \"Show me recent fire events\"\n"
        response += "• \"How many entities are active?\"\n"
        response += "• \"What happened this week?\"\n"
        response += "• \"Monthly PDU activity summary\"\n"
        response += "• \"Help me with predictions\"\n\n"
        response += "💡 *I can access detailed PDU logs for comprehensive time-based analysis!*"
        
        return {
            "response": response,
            "data": {"help": True},
            "intent": "help"
        }
    
    def _handle_general_query(self, user_query, auth_token):
        """Handle general queries"""
        response = f"🤔 **Understanding Your Query**\n\n"
        response += f"You asked: \"{user_query}\"\n\n"
        response += "I'm not sure exactly what you're looking for, but I can help you with:\n\n"
        response += "• **System information** - Ask about system status or health\n"
        response += "• **PDU data** - Ask about entities, fire events, collisions, or detonations\n"
        response += "• **Predictions** - Ask for forecasts or trend analysis\n"
        response += "• **Recent activity** - Ask what happened today or recently\n\n"
        response += "💡 *Try rephrasing your question or ask \"help\" for more guidance.*"
        
        return {
            "response": response,
            "data": {"query": user_query, "understood": False},
            "intent": "general"
        }

# Initialize chatbot
chatbot = SimpleLLMChatbot()

@app.route('/health', methods=['GET'])
@app.route('/api/prediction/health', methods=['GET'])
def health():
    """
    Health check endpoint for Kubernetes and Docker.
    """
    return jsonify({"status": "healthy", "service": "cap-pdu-prediction"}), 200

@app.route('/predict', methods=['POST'])
@app.route('/api/prediction', methods=['POST'])
def predict():
    """
    API endpoint to receive PDU JSON data and return predictions.
    """
    try:
        # Try to get JSON data, this will raise an exception for invalid JSON
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Request must be JSON"}), 400
        prediction_result = predict_pdu_data(data)
        if "error" in prediction_result:
            return jsonify(prediction_result), 400
        return jsonify(prediction_result)
    except (ValueError, TypeError, UnicodeDecodeError) as e:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        # Check if it's a JSON-related error
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['json', 'decode', 'parse', 'invalid']):
            return jsonify({"error": "Invalid JSON format"}), 400
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
@app.route('/api/prediction/chat', methods=['POST'])
def chat():
    """
    Chatbot endpoint for natural language queries about DIS platform data.
    Supports RAG (Retrieval-Augmented Generation) for accessing real-time data.
    """
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Request must be JSON"}), 400
        
        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({"error": "Query field is required"}), 400
        
        # Extract auth token from headers or request data
        auth_token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            auth_token = auth_header.split(' ')[1]
        elif data.get('auth_token'):
            auth_token = data.get('auth_token')
        
        # Process query with chatbot
        logger.info(f"Processing chat query: {user_query[:100]}...")
        response = chatbot.process_query(user_query, auth_token=auth_token)
        
        # Add metadata
        response['timestamp'] = datetime.now().isoformat()
        response['query'] = user_query
        response['service'] = 'cap-pdu-prediction-chatbot'
        
        return jsonify(response), 200
        
    except (ValueError, TypeError, UnicodeDecodeError) as e:
        logger.error(f"JSON parsing error in chat endpoint: {str(e)}")
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/chat/health', methods=['GET'])
@app.route('/api/prediction/chat/health', methods=['GET'])
def chat_health():
    """
    Health check endpoint specifically for chatbot functionality.
    """
    try:
        # Test basic chatbot functionality
        test_response = chatbot.process_query("help")
        
        health_status = {
            "status": "healthy",
            "service": "cap-pdu-prediction-chatbot",
            "chatbot_ready": bool(test_response.get('response')),
            "rag_system_ready": bool(chatbot.rag_system),
            "endpoints_configured": len(chatbot.rag_system.endpoints),
            "caching_enabled": USE_CACHING,
            "performance": {
                "avg_response_time": round(chatbot.performance_monitor.get_avg_response_time(), 3),
                "cache_hit_rate": round(chatbot.performance_monitor.get_cache_hit_rate(), 1),
                "total_requests": chatbot.performance_monitor.total_requests,
                "cache_size": len(chatbot.response_cache) if chatbot.response_cache else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Chatbot health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "service": "cap-pdu-prediction-chatbot",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/performance', methods=['GET'])
@app.route('/api/prediction/chat/performance', methods=['GET'])
def chat_performance():
    """
    Performance metrics endpoint for monitoring and alerting.
    """
    try:
        monitor = chatbot.performance_monitor
        
        performance_data = {
            "service": "cap-pdu-prediction-chatbot",
            "metrics": {
                "avg_response_time_seconds": round(monitor.get_avg_response_time(), 3),
                "max_response_time_seconds": max(monitor.response_times) if monitor.response_times else 0,
                "min_response_time_seconds": min(monitor.response_times) if monitor.response_times else 0,
                "cache_hit_rate_percent": round(monitor.get_cache_hit_rate(), 1),
                "total_requests": monitor.total_requests,
                "cache_hits": monitor.cache_hits,
                "cache_misses": monitor.cache_misses,
                "cache_size": len(chatbot.response_cache) if chatbot.response_cache else 0,
                "recent_response_times": monitor.response_times[-10:] if monitor.response_times else []
            },
            "configuration": {
                "caching_enabled": USE_CACHING,
                "cache_ttl_seconds": CACHE_TTL_SECONDS,
                "max_response_time_seconds": MAX_RESPONSE_TIME,
                "model_name": MODEL_NAME,
                "gpu_enabled": USE_GPU
            },
            "health_indicators": {
                "response_time_ok": monitor.get_avg_response_time() < MAX_RESPONSE_TIME,
                "cache_performance_ok": monitor.get_cache_hit_rate() > 20.0 if monitor.total_requests > 10 else True,
                "memory_usage_ok": len(chatbot.response_cache) < 1000 if chatbot.response_cache else True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(performance_data), 200
        
    except Exception as e:
        logger.error(f"Performance metrics failed: {str(e)}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    logger.info("Starting CAP PDU Prediction Service with Chatbot")
    logger.info(f"RAG System configured with {len(chatbot.rag_system.endpoints)} endpoints")
    logger.info(f"GPU support: {USE_GPU}")
    app.run(debug=True, host='0.0.0.0', port=5001)
