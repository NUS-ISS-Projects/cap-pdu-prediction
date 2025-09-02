# LLM Performance Optimization for GKE Deployment

## Overview

This document outlines performance optimization strategies for the DIS Platform Chatbot to ensure fast inference speeds in Google Kubernetes Engine (GKE) production environments.

## Current Architecture Considerations

### Performance Challenges
- **Model Loading Time**: Initial model loading can take 10-30 seconds
- **Inference Latency**: Response generation may take 2-5 seconds per query
- **Memory Usage**: LLM models require significant RAM (2-8GB depending on size)
- **Cold Start Issues**: First request after pod restart has high latency
- **Concurrent Users**: Multiple simultaneous requests can cause resource contention

## Optimization Strategies

### 1. Intelligent Model Selection

#### Current Implementation
```python
# Default lightweight model for fast inference
MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/DialoGPT-medium')  # ~350MB
```

#### Recommended Models by Use Case

**Production (Speed Priority):**
- `microsoft/DialoGPT-small` (117MB) - Sub-second inference
- `distilbert-base-uncased` (268MB) - Fast text understanding
- `google/flan-t5-small` (242MB) - Efficient text generation

**Balanced (Speed + Quality):**
- `microsoft/DialoGPT-medium` (345MB) - Current default
- `google/flan-t5-base` (990MB) - Better responses

**High Quality (GPU Required):**
- `microsoft/DialoGPT-large` (1.4GB) - Best responses
- `google/flan-t5-large` (2.8GB) - Advanced reasoning

### 2. Response Caching Strategy

#### Implementation Plan
```python
# Add to SimpleLLMChatbot class
class SimpleLLMChatbot:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.conversation_context = []
        self.response_cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
    
    def _get_cache_key(self, query, auth_token=None):
        """Generate cache key for query"""
        # Hash query + user context for caching
        import hashlib
        cache_input = f"{query.lower().strip()}_{auth_token[:10] if auth_token else 'anon'}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp):
        """Check if cached response is still valid"""
        return (datetime.now().timestamp() - timestamp) < self.cache_ttl
```

#### Cache Strategy
- **System Status Queries**: Cache for 2 minutes (data changes slowly)
- **Entity/Event Data**: Cache for 30 seconds (more dynamic)
- **Help/Static Content**: Cache for 1 hour (rarely changes)
- **Prediction Guidance**: Cache for 10 minutes (static content)

### 3. Model Optimization Techniques

#### Model Quantization
```python
# Add to environment configuration
USE_QUANTIZATION = os.getenv('USE_QUANTIZATION', 'true').lower() == 'true'
QUANTIZATION_BITS = int(os.getenv('QUANTIZATION_BITS', '8'))  # 8-bit quantization

# Model loading with quantization
if USE_QUANTIZATION:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
```

#### Model Compilation
```python
# TorchScript compilation for faster inference
if USE_GPU:
    model = torch.jit.script(model)  # Compile for GPU
else:
    model = torch.jit.optimize_for_inference(model)  # Optimize for CPU
```

### 4. Kubernetes Resource Optimization

#### Current Configuration
```yaml
resources:
  limits:
    cpu: "2000m"
    memory: "4Gi"
  requests:
    cpu: "500m"
    memory: "2Gi"
```

#### Optimized Configuration
```yaml
# For CPU-only deployment (fast startup)
resources:
  limits:
    cpu: "1000m"      # Reduced CPU for faster scheduling
    memory: "3Gi"      # Sufficient for medium models
  requests:
    cpu: "300m"       # Lower request for better pod density
    memory: "1.5Gi"    # Minimum for model loading

# For GPU deployment (maximum performance)
resources:
  limits:
    cpu: "2000m"
    memory: "6Gi"
    nvidia.com/gpu: 1
  requests:
    cpu: "500m"
    memory: "3Gi"
    nvidia.com/gpu: 1
```

#### Node Pool Configuration
```yaml
# Dedicated node pool for AI workloads
apiVersion: container.cnrm.cloud.google.com/v1beta1
kind: ContainerNodePool
metadata:
  name: ai-workload-pool
spec:
  nodeConfig:
    machineType: "n1-standard-4"  # 4 vCPU, 15GB RAM
    # For GPU: "n1-standard-4" + nvidia-tesla-t4
    accelerators:
    - acceleratorCount: 1
      acceleratorType: "nvidia-tesla-t4"
  initialNodeCount: 1
  autoscaling:
    enabled: true
    minNodeCount: 1
    maxNodeCount: 3
```

### 5. Application-Level Optimizations

#### Lazy Model Loading
```python
class SimpleLLMChatbot:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.conversation_context = []
        self._model = None  # Lazy loading
        self._tokenizer = None
    
    @property
    def model(self):
        """Lazy load model only when needed"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load model with optimizations"""
        logger.info(f"Loading model: {MODEL_NAME}")
        start_time = time.time()
        
        # Load with optimizations
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if USE_GPU else torch.float32,
            device_map="auto" if USE_GPU else None,
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
```

#### Response Streaming
```python
@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint for real-time responses"""
    def generate_response():
        # Yield partial responses as they're generated
        for chunk in chatbot.stream_response(user_query, auth_token):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return Response(generate_response(), mimetype='text/plain')
```

### 6. Monitoring and Performance Metrics

#### Key Metrics to Track
```python
# Add performance monitoring
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'memory_usage': [],
            'cache_hit_rate': 0,
            'model_load_time': 0
        }
    
    def track_response_time(self, start_time):
        response_time = time.time() - start_time
        self.metrics['response_times'].append(response_time)
        return response_time
    
    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB
```

#### Health Check with Performance Data
```python
@app.route('/chat/performance', methods=['GET'])
def chat_performance():
    """Performance metrics endpoint"""
    return jsonify({
        'avg_response_time': np.mean(monitor.metrics['response_times'][-100:]),
        'memory_usage_mb': monitor.get_memory_usage(),
        'cache_hit_rate': monitor.metrics['cache_hit_rate'],
        'model_loaded': chatbot._model is not None,
        'gpu_available': torch.cuda.is_available() if USE_GPU else False
    })
```

## Deployment Strategies

### 1. Multi-Tier Deployment

#### Fast Response Tier (CPU-only)
```yaml
# Lightweight deployment for common queries
replicas: 3
resources:
  requests: { cpu: "200m", memory: "1Gi" }
  limits: { cpu: "500m", memory: "2Gi" }
env:
  - name: MODEL_NAME
    value: "microsoft/DialoGPT-small"
  - name: USE_GPU
    value: "false"
```

#### High-Quality Tier (GPU-enabled)
```yaml
# GPU deployment for complex queries
replicas: 1
resources:
  requests: { cpu: "500m", memory: "3Gi", nvidia.com/gpu: 1 }
  limits: { cpu: "2000m", memory: "6Gi", nvidia.com/gpu: 1 }
env:
  - name: MODEL_NAME
    value: "google/flan-t5-base"
  - name: USE_GPU
    value: "true"
```

### 2. Load Balancing Strategy

#### Kong Gateway Configuration
```yaml
# Route simple queries to fast tier
- name: chatbot-fast
  paths: ['/api/prediction/chat']
  plugins:
  - name: request-transformer
    config:
      add:
        headers: ['X-Tier: fast']
  upstream:
    targets:
    - target: cap-pdu-prediction-fast:5001
      weight: 80
    - target: cap-pdu-prediction-gpu:5001
      weight: 20
```

### 3. Auto-scaling Configuration

#### HPA with Custom Metrics
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chatbot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cap-pdu-prediction-service
  minReplicas: 2
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
```

## Performance Benchmarks

### Target Performance Goals

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Response Time | <1s | <3s | >5s |
| Model Load Time | <10s | <30s | >60s |
| Memory Usage | <2GB | <4GB | >6GB |
| Cache Hit Rate | >60% | >40% | <20% |
| Concurrent Users | 50+ | 20+ | <10 |

### Testing Strategy

#### Load Testing Script
```bash
#!/bin/bash
# Load test the chatbot endpoint

echo "Starting chatbot load test..."

# Test with different query types
QUERIES=(
  "What is the system status?"
  "Show me today's activity"
  "How many entities are active?"
  "Help me with predictions"
)

for i in {1..100}; do
  QUERY=${QUERIES[$((RANDOM % ${#QUERIES[@]}))]} 
  curl -X POST "http://dis.local:32080/api/prediction/chat" \
    -H "Authorization: Bearer $JWT_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$QUERY\"}" \
    -w "Response time: %{time_total}s\n" &
done

wait
echo "Load test completed"
```

## Implementation Checklist

### Phase 1: Basic Optimizations
- [ ] Implement response caching
- [ ] Add performance monitoring
- [ ] Optimize model selection
- [ ] Configure resource limits

### Phase 2: Advanced Optimizations
- [ ] Implement model quantization
- [ ] Add lazy model loading
- [ ] Configure auto-scaling
- [ ] Set up performance dashboards

### Phase 3: Production Deployment
- [ ] Deploy multi-tier architecture
- [ ] Configure load balancing
- [ ] Implement monitoring alerts
- [ ] Conduct load testing

## Troubleshooting Guide

### Common Performance Issues

1. **High Response Times**
   - Check model size and complexity
   - Verify cache hit rates
   - Monitor CPU/memory usage
   - Consider GPU acceleration

2. **Memory Issues**
   - Reduce model size
   - Enable model quantization
   - Implement garbage collection
   - Increase pod memory limits

3. **Cold Start Problems**
   - Implement model pre-loading
   - Use readiness probes with longer delays
   - Consider keeping warm replicas

4. **Scaling Issues**
   - Adjust HPA thresholds
   - Monitor node capacity
   - Check resource requests/limits
   - Verify cluster auto-scaling

---

**Note**: This optimization guide ensures the DIS Platform Chatbot delivers fast, reliable responses in production GKE environments while maintaining cost efficiency and scalability.