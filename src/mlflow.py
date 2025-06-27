import mlflow
from src.constants import TRACKING_URI, EXPERIMENT_NAME

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = mlflow.tracking.MlflowClient()


def get_or_create_run(run_name: str):
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )

    for r in runs:
        return r

    return client.create_run(
        experiment_id=client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id,
        tags={"mlflow.runName": run_name},
    )


def get_run_by_name(run_name: str):
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )

    if not runs:
        raise ValueError(f"No run found with name: {run_name}")

    assert len(runs) == 1, f"Multiple runs found with name: {run_name}"

    return runs[0]


def get_runs_by_names(run_names: list[str]):
    return [get_run_by_name(run_name) for run_name in run_names]
