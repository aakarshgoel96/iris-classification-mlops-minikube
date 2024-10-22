import time
import mlflow
import requests
from prometheus_client import start_http_server, Gauge
from datetime import datetime
from mlflow.tracking import MlflowClient

# Prometheus metrics
accuracy_gauge = Gauge('mlflow_model_accuracy', 'Accuracy of the latest MLflow model')
last_model_time_gauge = Gauge('mlflow_model_last_deploy_time', 'Timestamp of the last model deployment in seconds since epoch')

# Start Prometheus HTTP server to expose metrics
start_http_server(8001)

# Initialize MLflow client
mlflow_client = MlflowClient()

def get_latest_model_metrics():
    try:
        # Query for the latest registered model
        model_name = "BestIrisModel"
        latest_versions = mlflow_client.get_latest_versions(model_name, stages=["Production"])
        if latest_versions:
            latest_model = latest_versions[0]
            run_id = latest_model.run_id

            # Get the run's accuracy
            run = mlflow_client.get_run(run_id)
            accuracy = run.data.metrics.get("accuracy", 0)

            # Update accuracy gauge
            accuracy_gauge.set(accuracy)

            # Get the model creation timestamp
            last_update_time = latest_model.creation_timestamp / 1000  # Convert to seconds
            last_model_time_gauge.set(last_update_time)

            print(f"Latest model: {latest_model.name}, Accuracy: {accuracy}, Last updated: {datetime.utcfromtimestamp(last_update_time)}")

        else:
            print("No models found in production stage")
    except Exception as e:
        print(f"Error fetching model metrics: {e}")

def monitor_mlflow_models():
    while True:
        # Fetch the latest model metrics every 5 minutes
        get_latest_model_metrics()
        time.sleep(300)

if __name__ == "__main__":
    print("Starting MLflow model monitoring service...")
    monitor_mlflow_models()
