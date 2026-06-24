# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS dependencies (if any required for OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./

# Expose the default Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit app
CMD ["streamlit", "run", "krishi_mitra.py", "--server.port", "8501", "--server.enableCORS", "false"]
