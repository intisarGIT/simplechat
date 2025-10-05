# Dockerfile for chat10fixed6 app
# Uses python slim, installs requirements and runs the script which launches Gradio
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Ensure upload/output directories exist
RUN mkdir -p /app/uploads /app/output /app/models

# Expose Gradio default port (configurable via GRADIO_PORT env)
EXPOSE 7860

# Default environment variables (can be overridden by host)
ENV GRADIO_PORT=7860

# Run the app
CMD ["python", "chat10fixed6.py"]
