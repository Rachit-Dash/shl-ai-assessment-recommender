# SHL AI Assessment Recommender

Production-ready FastAPI solution for the SHL AI Intern Assignment. The API is stateless, catalog-grounded, and returns strict JSON responses for SHL assessment recommendations, refinements, comparisons, clarification, and refusal cases.

## Features

- `GET /health`
- `POST /chat`
- Stateless conversation handling from full message history
- SHL catalog-only recommendations
- Clarifies vague hiring requests before recommending
- Supports recommendation refinement
- Supports grounded assessment comparisons
- Refuses off-topic and prompt-injection requests
- FAISS semantic retrieval with deterministic keyword fallback
- Gemini integration with safe fallback when the API key is missing

## Project Structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── scraper.py
├── build_index.py
├── render.yaml
├── .env.example
├── catalog/
│   └── shl_catalog.json
├── embeddings/
│   ├── faiss.index
│   └── metadata.pkl
├── services/
│   ├── retrieval.py
│   ├── recommender.py
│   ├── conversation.py
│   └── llm_service.py
├── prompts/
│   └── system_prompt.txt
└── utils/
    └── helpers.py
```

## Requirements

- Python 3.11
- Gemini API key for LLM wording, optional for local fallback behavior

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file or set environment variables in your shell:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

`GOOGLE_API_KEY` is optional. If it is missing, the API still works with deterministic fallback replies.

## Refresh Catalog

The repository includes a small SHL catalog seed so the API can run immediately. To refresh from the live SHL Individual Test Solutions catalog:

```bash
python scraper.py
```

The scraper uses `requests` and `beautifulsoup4`, handles pagination with `type=2`, follows official catalog item pages, and writes `catalog/shl_catalog.json`.

## Build Embeddings

After installing dependencies and refreshing the catalog if needed:

```bash
python build_index.py
```

This creates:

- `embeddings/faiss.index`
- `embeddings/metadata.pkl`

If the index is missing or cannot be loaded, the API falls back to deterministic keyword retrieval from `catalog/shl_catalog.json`.

## Run Locally

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/health
```

## API Examples

### Health

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

### Vague Query

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"I need an assessment\"}]}"
```

Response shape:

```json
{
  "reply": "To recommend the right SHL assessments, what role are you hiring for and do you need technical, cognitive, personality, or behavioral testing?",
  "recommendations": [],
  "end_of_conversation": false
}
```

### Recommendation

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hiring a Java backend developer. Need technical and cognitive tests.\"}]}"
```

Response shape:

```json
{
  "reply": "I found SHL catalog-backed assessment matches for this hiring need.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K - Knowledge & Skills"
    }
  ],
  "end_of_conversation": true
}
```

### Refinement

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hiring a Java backend developer\"},{\"role\":\"assistant\",\"content\":\"I found relevant SHL assessments.\"},{\"role\":\"user\",\"content\":\"Also include personality tests\"}]}"
```

### Comparison

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Difference between OPQ and GSA\"}]}"
```

## Render Deployment

1. Push the project to GitHub.
2. Create a new Render Web Service.
3. Use the included `render.yaml`, or configure manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Python version: `3.11.9`
4. Add environment variable:
   - `GOOGLE_API_KEY`
   - Optional: `GEMINI_MODEL=gemini-1.5-flash`

For best recommendation quality, run `python scraper.py` and `python build_index.py` before deployment and commit the generated catalog and embedding artifacts.

## Strict Response Schema

All `/chat` responses return only:

```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "assessment name",
      "url": "official shl url",
      "test_type": "type"
    }
  ],
  "end_of_conversation": false
}
```

No markdown and no additional fields are returned by the API.
