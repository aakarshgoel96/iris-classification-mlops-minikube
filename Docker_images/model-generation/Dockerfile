
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY model_generation.py .
COPY iris.csv .

# Set environment variables for MinIO
ENV MINIO_ACCESS_KEY=YVVWLh1LWl4iT4U34dlg
ENV MINIO_SECRET_KEY=jlAkCk89zV51o0EvHwIA9kGRnmNtn6PtkypJ3aFy

# Command to run the application
CMD ["python", "model_generation.py"]
