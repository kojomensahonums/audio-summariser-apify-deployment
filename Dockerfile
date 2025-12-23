# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (for Streamlit if needed)
EXPOSE 8080

# Run the Python script
CMD ["python", "main.py"]
