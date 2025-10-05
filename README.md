# Simple Mistral Chatbot (Gradio)

This is a small Gradio UI that calls the Mistral API `POST /v1/chat/completions` endpoint.

Files:
- `app.py` - the Gradio app. Enter your Mistral API key and model id in the UI.
- `requirements.txt` - minimal dependencies.

Usage:

1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the app:

```powershell
python app.py
```

Notes and assumptions:
- The app sends `Authorization: ApiKey <KEY>` in the header as required by Mistral docs.
- This app uses non-streaming completions and a simple `messages` array with dicts of `role` and `content`.
- For production use, don't keep API keys in plain text; use environment variables or a secrets manager.

Edge cases handled:
- Missing API key: the UI will not call the API and returns empty assistant response.
- HTTP errors: show server response in assistant bubble.
- Unexpected response shape: the code attempts to extract text robustly and falls back to JSON.

Next steps (optional):
- Add streaming support (SSE) for token-by-token updates.
- Add local conversation history persistence.
- Add model list lookup using `GET /v1/models`.
"# simplechat" 
