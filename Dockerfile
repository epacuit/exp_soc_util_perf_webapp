# Use the official Python 3.12 base image
FROM python:3.12-slim

# Install system dependencies required for cffi and other libraries
RUN apt-get update && apt-get install -y \
    libffi-dev \
    pkg-config \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy application files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up Streamlit configuration
RUN mkdir -p ~/.streamlit/ && \
    echo "[server]\nheadless = true\nport = \$PORT\nenableCORS = false" > ~/.streamlit/config.toml

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "Expected_Social_Utility_Performance.py"]
