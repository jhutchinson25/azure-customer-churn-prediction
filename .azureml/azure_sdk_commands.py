from config import SUBSCRIPTION, RESOURCE_GROUP, WS_NAME
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities import AmlCompute, Data, Model, ManagedOnlineEndpoint
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import command, Input


# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)
print(ml_client)

# provision cluster
compute_name = 'cpu-cluster-1'

# Check if cluster exists
try:
    ml_client.compute.get(compute_name)
    print(f"Compute cluster '{compute_name}' already exists.")
except ResourceNotFoundError:
    print(f"Creating compute cluster '{compute_name}'...")
    compute = AmlCompute(
        name=compute_name,
        size="STANDARD_DS1_v2",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120
    )
    ml_client.begin_create_or_update(compute).result()
    print(f"Compute cluster '{compute_name}' created.")

# create data asset
dataset_name = "churn-data"
dataset_version = "1"

# Check if data asset exists
try:
    existing_data = ml_client.data.get(name=dataset_name, version=dataset_version)
    print(f"Dataset already exists: {existing_data.name}:{existing_data.version}")
except ResourceNotFoundError:
    print("Dataset not found. Creating and uploading...")

    my_data = Data(
        path="../data/customer_churn_data.csv",
        type=AssetTypes.URI_FILE,
        description="Customer churn dataset",
        name=dataset_name,
        version=dataset_version
    )

    ml_client.data.create_or_update(my_data)
    print(f"Dataset {dataset_name}:{dataset_version} registered.")

# configure job
job = command(
    code="../src",
    command="python train.py --data ${{inputs.input_data}} --registered_model_name churn-model",
    inputs={"input_data": Input(type="uri_file", path="churn-data:1")},
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="cpu-cluster-1",
    display_name="customer-churn",
    experiment_name="customer-churn-training"
)

returned_job = ml_client.create_or_update(job)
url = returned_job.studio_url
print("Monitor your job at", url)

ml_client.jobs.stream(returned_job.name)  # Live logs + wait until done

# register model
job_name = returned_job.name
run_model = Model(
    path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/",
    name="mlflow-customer-churn",
    description="Model created from run.",
    type=AssetTypes.MLFLOW_MODEL,
)
ml_client.models.create_or_update(run_model)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="customer-churn-endpoint",
    description="Online endpoint for predicting customer churn",
    auth_mode="key",
)

ml_client.begin_create_or_update(endpoint).result()
