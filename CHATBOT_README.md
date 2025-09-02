# DIS Platform Chatbot Integration

## Overview

The CAP PDU Prediction service now includes an intelligent chatbot that provides natural language access to DIS (Distributed Interactive Simulation) platform data. The chatbot uses RAG (Retrieval-Augmented Generation) techniques to access real-time data from various services and provide intelligent responses to user queries.

## Features

### 🤖 Intelligent Query Processing
- Natural language understanding for DIS platform queries
- Intent classification and context-aware responses
- Real-time data retrieval from multiple services
- Structured responses with relevant data and insights

### 📊 Data Access Capabilities
- **System Status**: Health checks and service monitoring
- **Entity States**: Simulation object tracking and status
- **Fire Events**: Weapon discharge and munition firing data
- **Collision Events**: Physical impact and collision tracking
- **Detonation Events**: Explosive impacts and munition detonations
- **Real-time Metrics**: Performance indicators and statistics
- **Aggregated Data**: Daily, weekly, and monthly summaries

### 🔮 Prediction Integration
- Seamless integration with existing PDU prediction capabilities
- Guidance on using prediction endpoints
- Trend analysis and forecasting insights

## API Endpoints

### Chat Endpoint
```
POST /api/prediction/chat
```

**Request Format:**
```json
{
  "query": "What's the system status?"
}
```

**Response Format:**
```json
{
  "response": "🔍 **System Status Overview**\n\n**Services Health:** 2/2 services operational\n**Recent Activity:** System is actively processing PDU data\n\n💡 *Ask me about specific PDU events, predictions, or system metrics for more details.*",
  "data": {
    "services_health": {...},
    "recent_metrics": {...}
  },
  "intent": "system_status",
  "timestamp": "2024-01-20T10:30:00Z",
  "query": "What's the system status?",
  "service": "cap-pdu-prediction-chatbot"
}
```

### Chatbot Health Check
```
GET /api/prediction/chat/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "cap-pdu-prediction-chatbot",
  "chatbot_ready": true,
  "rag_system_ready": true,
  "endpoints_configured": 11,
  "timestamp": "2024-01-20T10:30:00Z"
}
```

## Authentication

The chatbot endpoints require Firebase JWT authentication, consistent with other protected endpoints in the DIS platform.

**Header:**
```
Authorization: Bearer <your_jwt_token>
```

## Example Queries

### System Information
- "What's the system status?"
- "Are all services healthy?"
- "Show me system overview"

### Entity Data
- "How many entities are active?"
- "Show me entity states"
- "What entities are in the simulation?"

### Event Data
- "Show me recent fire events"
- "Any collisions today?"
- "What detonations occurred?"
- "Tell me about recent activity"

### Metrics and Statistics
- "Show me system metrics"
- "What are the performance indicators?"
- "Give me today's statistics"

### Predictions
- "Help me with predictions"
- "How do I forecast PDU data?"
- "Explain prediction capabilities"

### General Help
- "Help"
- "What can you do?"
- "Show me available commands"

## cURL Examples

### Basic Chat Query
```bash
curl -X POST "http://localhost:5001/api/prediction/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"query": "What is the system status?"}'
```

### Through Kong Gateway
```bash
curl -X POST "http://dis.local:32080/api/prediction/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"query": "Show me recent fire events"}'
```

### Health Check
```bash
curl "http://dis.local:32080/api/prediction/chat/health"
```

## Architecture

### RAG System Components

1. **Data Retrieval**: Connects to multiple DIS platform services
   - Data Acquisition Service
   - Data Ingestion Service
   - Data Processing Service

2. **Intent Classification**: Keyword-based intent recognition
   - System status queries
   - Entity information requests
   - Event data queries
   - Metrics and statistics
   - Prediction assistance

3. **Response Generation**: Context-aware response formatting
   - Structured markdown responses
   - Relevant data inclusion
   - Error handling and fallbacks

### Service Integration

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Chatbot API    │───▶│   RAG System    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Formatted     │◀───│   Response       │◀───│   Data Services │
│   Response      │    │   Generator      │    │   (Acquisition, │
└─────────────────┘    └──────────────────┘    │   Ingestion,    │
                                                │   Processing)   │
                                                └─────────────────┘
```

## Configuration

### Environment Variables

- `DATA_ACQUISITION_URL`: URL for data acquisition service (default: `http://data-acquisition-service:8080`)
- `DATA_INGESTION_URL`: URL for data ingestion service (default: `http://data-ingestion-service:8080`)
- `DATA_PROCESSING_URL`: URL for data processing service (default: `http://data-processing-service:8080`)
- `USE_GPU`: Enable GPU support for advanced LLM features (default: `false`)
- `MODEL_NAME`: LLM model name for advanced features (default: `microsoft/DialoGPT-medium`)

### Resource Requirements

**Standard Configuration:**
- CPU: 500m request, 2000m limit
- Memory: 2Gi request, 4Gi limit

**GPU Configuration (Optional):**
- GPU: 1x NVIDIA GPU
- Memory: 4Gi+ recommended
- CPU: 1000m+ recommended

## Deployment

### Kubernetes Deployment

The chatbot is integrated into the existing `cap-pdu-prediction-service` deployment with enhanced resource allocation and environment configuration.

### Kong Gateway Integration

New routes are automatically configured:
- `/api/prediction/chat` (Protected)
- `/api/prediction/chat/health` (Public)

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure valid JWT token is provided
   - Check token expiration
   - Verify Firebase configuration

2. **Service Connectivity**
   - Check if backend services are running
   - Verify network connectivity between services
   - Review service URLs in environment variables

3. **Resource Constraints**
   - Monitor memory usage (chatbot requires more memory)
   - Check CPU utilization
   - Consider enabling GPU support for better performance

### Health Monitoring

```bash
# Check chatbot health
curl "http://dis.local:32080/api/prediction/chat/health"

# Check overall service health
curl "http://dis.local:32080/api/prediction/health"

# Monitor logs
kubectl logs -f deployment/cap-pdu-prediction-service
```

## Future Enhancements

### Planned Features
- Advanced LLM integration with 8B models
- Vector database for improved RAG performance
- Conversation memory and context retention
- Multi-language support
- Voice interface capabilities

### GPU Support
To enable GPU support for advanced LLM features:

1. Uncomment GPU resource limits in deployment
2. Set `USE_GPU=true` environment variable
3. Ensure GPU nodes are available in cluster
4. Update model configuration for GPU-optimized models

## Support

For issues or questions regarding the chatbot functionality:
1. Check the health endpoints
2. Review application logs
3. Verify service connectivity
4. Consult the main DIS platform documentation

---

**Note**: This chatbot provides intelligent access to DIS platform data through natural language queries, making the system more accessible and user-friendly while maintaining security and performance standards.