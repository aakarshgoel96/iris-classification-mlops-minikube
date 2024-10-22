# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install the necessary dependencies
RUN pip install prometheus-client mlflow requests

# Expose the port for Prometheus scraping
EXPOSE 8001

# Run the monitoring service
CMD ["python", "monitor_mlflow.py"]
