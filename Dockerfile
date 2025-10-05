# Dockerfile for Modern XAI Toolkit
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs logs examples

# Expose ports for web interfaces
EXPOSE 7860 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "gradio" ]; then\n\
    python xai_web_app.py\n\
elif [ "$1" = "streamlit" ]; then\n\
    streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501\n\
elif [ "$1" = "cli" ]; then\n\
    python modern_xai.py\n\
else\n\
    echo "Usage: docker run <image> [gradio|streamlit|cli]"\n\
    echo "  gradio    - Launch Gradio web interface"\n\
    echo "  streamlit - Launch Streamlit dashboard"\n\
    echo "  cli       - Run command-line demo"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["gradio"]
