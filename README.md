# AstroAI — Conversational Insight Agent

Multi-turn conversational AI service for Vedic astrology, featuring RAG and conversation ownership.

## Architecture

| Layer | Technology |
|---|---|
| API | FastAPI |
| Agent | LangGraph (tool-calling agent) |
| LLM | OpenAI GPT-4o-mini |
| Vector Store | ChromaDB |
| Memory | Redis |
| Package Mgr | Poetry |
| Containerisation | Docker Compose |

## Setup

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- OpenAI API key

### Clone the Repository

```bash
git clone https://github.com/aflahibr/AstroAI.git
cd AstroAI
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Build & Run

```bash
# Build images and start services (FastAPI + Redis)
# The knowledge base is automatically ingested during the build step.
docker compose up --build

# Or run in detached mode
docker compose up --build -d
```

The API will be available at **http://localhost:8000**.

### Stop Services

```bash
docker compose down
```

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "message": "How will my month be in career?",
    "user_profile": {
      "name": "Ritika",
      "birth_date": "1995-08-20",
      "birth_time": "14:30",
      "birth_place": "Jaipur, India",
      "preferred_language": "hi"
    }
  }'
```

### Example Response

```json
{
  "response": "आपके लिए यह महीना अवसर लेकर आ रहा है...",
  "zodiac": "Leo",
  "context_used": ["career_guidance", "leo_traits"],
  "retrieval_used": true
}
```

## Project Structure

```
AstroAI/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── services/
│       ├── agent.py          # LangGraph agent workflow
│       ├── memory.py         # Redis session memory
│       └── rag.py            # ChromaDB vector search
├── data/
│   ├── vedic_astrology.txt   # Knowledge corpus
│   ├── planetary_traits.json # Planet characteristics
│   └── zodiac_personality.json # Zodiac sign traits
├── scripts/
│   └── ingest.py             # Data ingestion to ChromaDB
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Design Decisions

### Intent-Aware RAG
The RAG tool is **bound to the LLM** via LangGraph tool-calling. The LLM itself decides when to invoke retrieval based on the tool's description. This avoids a separate classifier — the LLM's judgment determines:
- **Retrieve**: Factual questions about planets, houses, signs, transits
- **Skip retrieval**: Greetings, follow-ups, summaries, meta-questions

### Memory Control
- **Windowing**: Only the last 10 turns are sent to the LLM context
- **Max storage**: Redis trims to 20 turns max
- **TTL decay**: Sessions expire after 24 hours of inactivity

### Quality-Cost Trade-off
| Scenario | Retrieval | Outcome |
|---|---|---|
| "What does Saturn in the 7th house mean?" | ✅ Used | Provides grounded, factual response |
| "Summarise what you told me" | ❌ Skipped | Uses conversation history only, avoids irrelevant retrieval noise |
