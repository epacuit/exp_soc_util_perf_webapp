FROM python:3.12-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    pkg-config \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up Streamlit configuration
RUN mkdir -p ~/.streamlit/ && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "port = 8501" >> ~/.streamlit/config.toml && \
    echo "enableCORS = false" >> ~/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

# Command to start the application
CMD ["streamlit", "run", "Expected_Social_Utility_Performance.py"]
