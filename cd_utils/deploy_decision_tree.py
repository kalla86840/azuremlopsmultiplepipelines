from azureml.core import Workspace, Environment, Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

ws = Workspace.from_config(path="config/cars_config.json")
model = Model(ws, name="decision_tree_model")

env = Environment("inference-env")
env.python.conda_dependencies.add_pip_package("scikit-learn")
env.python.conda_dependencies.add_pip_package("joblib")

inference_config = InferenceConfig(entry_script="models/decision_tree/score.py", environment=env)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service_name = "decision-tree-svc"
service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)

service.wait_for_deployment(show_output=True)
print(f"Deployed to ACI: {service.scoring_uri}")