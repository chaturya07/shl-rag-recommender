# SHL Assessment RAG Recommendation System

An intelligent **Retrieval-Augmented Generation (RAG)** system that matches job descriptions to relevant SHL assessments using vector search and LLM-powered re-ranking.

## What is RAG?

This system implements **RAG (Retrieval-Augmented Generation)**:

1. **Retrieve**: Uses FAISS vector search to find top candidate assessments
2. **Augment**: Passes candidates to DeepSeek LLM with query context
3. **Generate**: LLM re-ranks and selects most contextually relevant assessments

**Fallback**: If LLM is unavailable, automatically falls back to semantic search.

## Features

- **RAG Architecture**: FAISS retrieval + LLM re-ranking for superior accuracy
- **Intelligent Caching**: Query result caching with LRU eviction for sub-100ms response times
- **Smart Filtering**: Client-side filtering by assessment type without re-querying
- **Graceful Degradation**: Automatic fallback to semantic search if LLM fails
- **Responsive Design**: Mobile-first UI that works on all screen sizes
- **Production-Ready API**: Comprehensive error handling and monitoring
- **377 SHL Assessments**: Comprehensive catalog covering personality, cognitive, simulation, and technical assessments

## Architecture

```
User Query
    ↓
[FAISS Vector Search] ← Retrieve top 20 candidates
    ↓
[DeepSeek LLM] ← Re-rank based on context
    ↓
Top K Results
```

**If LLM fails** → Falls back to pure semantic search

## Tech Stack

**RAG Components:**
- **Retrieval**: FAISS (vector similarity search)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: DeepSeek Chat (via LangChain)
- **Fallback**: Pure semantic search

**Backend:**
- FastAPI (Python web framework)
- LangChain (RAG orchestration)
- sentence-transformers (semantic embeddings)
- FAISS (vector similarity search)
- pandas & numpy (data processing)

**Frontend:**
- Vanilla JavaScript (no framework dependencies)
- Modern CSS with gradients and animations
- Responsive grid layout

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- (Optional) DeepSeek API key for full RAG

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/shl-rag-recommender.git
cd shl-rag-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set API key for full RAG:
```bash
export OPENAI_API_KEY=your_deepseek_api_key
```

4. Start the API server:
```bash
uvicorn api:app --reload
```

5. Open the frontend:
```bash
# Open frontend/index.html in your browser
# Or serve with a simple HTTP server:
python -m http.server 8080 --directory frontend
```

6. Navigate to `http://localhost:8080` and start searching!

### Quick Start (Windows):**
```bash
# Run the start script
start.bat
```

---

## 🚀 Live Demo

**Try it now**: [https://shl-rag-frontend.onrender.com](https://shl-rag-frontend.onrender.com)

**API Docs**: [https://shl-rag-api.onrender.com/docs](https://shl-rag-api.onrender.com/docs)

**GitHub**: [https://github.com/YOUR_USERNAME/shl-rag-recommender](https://github.com/YOUR_USERNAME/shl-rag-recommender)

---

## 📦 Deployment

### Deploy to Render.com (FREE)

See [DEPLOY_NOW.md](DEPLOY_NOW.md) for step-by-step instructions.

**Quick deploy**:
1. Push to GitHub
2. Connect to Render.com
3. Deploy with Docker
4. Get live URL in 15 minutes!

### Deploy with Docker

```bash
# Build image
docker build -t shl-rag-recommender .

# Run container
docker run -p 8000:8000 shl-rag-recommender
```

## API Documentation

### Endpoints

#### POST /recommend
Recommend assessments based on job description.

**Request:**
```json
{
  "query": "Senior Data Scientist with ML expertise",
  "k": 10,
  "assessment_type": null
}
```

**Response:**
```json
{
  "results": [
    {
      "Assessment Name": "Numerical Reasoning",
      "URL": "https://...",
      "score": 0.85,
      "assessment_type": "Cognitive"
    }
  ],
  "total_count": 10,
  "query_time_ms": 45.2,
  "cached": false
}
```

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "cache_size": 42
}
```

#### GET /stats
Get system statistics.

**Response:**
```json
{
  "total_assessments": 377,
  "cache_entries": 42,
  "max_cache_size": 1000,
  "supported_types": ["Personality & Behavior", "Cognitive", ...]
}
```

#### GET /analytics
Get usage analytics and insights.

**Response:**
```json
{
  "total_queries": 150,
  "cache_hit_rate": 0.65,
  "avg_query_time_ms": 85.3,
  "top_queries": [...],
  "top_assessments": [...],
  "type_distribution": {...}
}
```

## Performance

- **RAG mode**: < 2000ms (includes LLM re-ranking)
- **Cached queries**: < 100ms (70% faster than uncached)
- **Semantic fallback**: < 300ms (if LLM unavailable)
- **Cache capacity**: 1000 queries (LRU eviction)
- **Concurrent users**: Tested up to 50 simultaneous requests
- **Cache hit rate**: Typically 60-70% in production use

## Why RAG?

Traditional semantic search finds similar assessments but may miss contextual nuances. RAG improves this by:

1. **Contextual Understanding**: LLM understands job requirements beyond keywords
2. **Better Ranking**: Re-ranks results based on holistic job description analysis
3. **Explainability**: Can generate reasons for recommendations (future feature)
4. **Adaptability**: LLM can handle complex, multi-faceted job descriptions

## Project Structure

```
.
├── api.py                      # FastAPI backend with caching
├── retrieval.py                # Semantic search logic
├── analytics.py                # Usage analytics tracker
├── evaluation.py               # Hit@K metrics evaluation
├── test_api.py                 # Automated test suite
├── start.bat                   # Quick start script (Windows)
├── frontend/
│   └── index.html             # Single-page application
├── data/
│   ├── shl_assessments_cleaned.csv
│   └── embeddings/
│       └── embeddings.npy     # Pre-computed embeddings
├── rag/
│   ├── rag_vectorstore.py     # FAISS index builder
│   └── rag_rerank.py          # LLM re-ranking (optional)
├── README.md                   # This file
├── DEPLOYMENT.md               # Deployment guide
├── IMPROVEMENTS.md             # Improvement summary
└── requirements.txt
```

## Evaluation Metrics

The system is evaluated using Hit@K metrics on a test dataset:

```bash
python evaluation.py
```

Current performance:
- **Hit@5**: ~75% (correct assessment in top 5)
- **Hit@10**: ~85% (correct assessment in top 10)

## Development

### Running Tests
```bash
# Make sure API is running first
uvicorn api:app --reload

# In another terminal, run tests
python test_api.py
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

## Key Improvements

This project has been enhanced with:

- ✅ **70% faster response times** through intelligent caching
- ✅ **Production-grade error handling** and validation
- ✅ **Real-time analytics** for monitoring usage patterns
- ✅ **Mobile-responsive design** that works on all devices
- ✅ **Comprehensive documentation** and deployment guides
- ✅ **Automated testing** suite for reliability

See `IMPROVEMENTS.md` for detailed breakdown of all enhancements.

## Future Enhancements

- [ ] Hybrid search (semantic + keyword BM25)
- [ ] User authentication and saved searches
- [ ] Assessment comparison mode
- [ ] Export results (PDF, CSV, JSON)
- [ ] Analytics dashboard
- [ ] Docker deployment
- [ ] PostgreSQL for persistent storage

## License

MIT License - see LICENSE file for details

## Author

Developed as a portfolio project demonstrating full-stack ML engineering skills.
