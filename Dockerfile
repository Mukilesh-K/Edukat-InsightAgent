# Use the official Python image as the builder stage
FROM python:3.8-slim AS builder

# Set the working directory for the builder stage
WORKDIR /app

# Install system dependencies required for psycopg2 and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the builder stage
COPY requirements.txt .

# Install Python dependencies (including build dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the builder stage
COPY . .

# Final production stage
FROM python:3.8-slim

# Set the working directory for the final image
WORKDIR /app

# Copy only the necessary files (installed packages and app) from the builder stage
COPY --from=builder /app /app

# Install only production dependencies (if there are any additional production dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port Uvicorn will run on
EXPOSE 8000

# Set the command to run the Uvicorn server
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]