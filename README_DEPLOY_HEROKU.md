This repository contains a Gradio + FastAPI app (`chat10fixed6.py`) adapted to run as a single ASGI app. To run locally:

1. Create virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run locally with uvicorn:

```powershell
$env:PORT=7860
.\.venv\Scripts\python.exe -m uvicorn chat10fixed6:root_app --host 0.0.0.0 --port $env:PORT
```

3. For Heroku deploy:
- Ensure Procfile exists (web: uvicorn chat10fixed6:root_app --host 0.0.0.0 --port $PORT)
- Set config vars (MISTRAL_API_KEY, IMAGEROUTER_API_KEY, FACEPP_API_KEY, FACEPP_API_SECRET)
- git push heroku main

Notes:
- Heavy dependencies like OpenCV and Pillow can make slug size large. Consider migrating heavy processing to a separate service if Heroku slug limits are exceeded.
- Heroku's acceptable use applies; NSFW content may violate policies.
