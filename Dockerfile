FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY app.py .
COPY src ./src
COPY models ./models

# Expose Flask port
EXPOSE 8000

# Run Flask app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
