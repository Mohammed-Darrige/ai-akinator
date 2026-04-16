# AI Akinator

FastAPI backend for an LLM-native animal guessing game.

## Features

- Server-side session state
- Constraint ledger for confirmed facts
- Streaming SSE responses
- Validation for both generated questions and final guesses
- OpenAI-compatible provider support via environment variables

## Environment

Copy `.env.example` to `.env` and fill in your provider credentials:

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.z.ai/api/coding/paas/v4
LLM_MODEL_NAME=glm-5-turbo
```

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is exposed under `/api/v1`.

## Notes

- `.env` is local-only and ignored by git.
- Default examples target Z.ai's OpenAI-compatible endpoint, but other compatible providers can be used.
