"""MLflow utilities"""

import mlflow


def setup_mlflow(tracking_uri: str = None, experiment_name: str = None):
    """Setup MLflow tracking"""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📊 MLflow tracking URI: {tracking_uri}")

    if experiment_name:
        mlflow.set_experiment(experiment_name)
        print(f"🧪 MLflow experiment: {experiment_name}")
