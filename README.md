# Customer Churn Prediction with Azure ML
Predicting customer churn using machine learning with an end-to-end pipeline for training, deployment, and inference with Azure Machine Learning and Streamlit
## Overview
This project tackles a common business problem: customer churn. A model is trained using a real-world dataset to identify customers that are likely to leave a subscription service. The model is built using sklearn's GradientBoostingClassifer and deployed to Azure for scalable inference.
## Stack
* Azure ML
* Streamlit
* pandas
* scikit-learn
* Python
## Quickstart
### 1. Setup Local Environment
* Ensure that Python is installed
* Clone project into the desired directory
```
cd path/to/your/folder
git clone https://github.com/jhutchinson25/azure-customer-churn-prediction.git
```
* Install dependencies.  In PowerShell run:
```
pip install -r requirements.txt
```
* Install the Azure CLI. In PowerShell run:
```
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi; Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'
```
### 2. Configure Azure Machine Learning Workspace
In Azure, create a subscription, resource group, and machine learning workspace. It is recommended that each machine learning workspace gets its own resource group because each workspace becomes preprovisioned with multiple resources. Note the Subscription ID, resource group, and workspace name and insert them into a new file called config.py. Ensure that the information from this file does not become public, e.g. by pushing it to a GitHub repository.
```
SUBSCRIPTION = 'your-subscription-id'
RESOURCE_GROUP = 'your-resource-group'
WS_NAME = 'your-workspace-name'
```
### 3. Train and Deploy Model
* Authenticate your local device with Azure
```
az login
```
* Run azure_sdk_commands.py.  This will provision compute, create a dataset, run the training job, and deploy the model to a managed online endpoint. If a dataset or compute of the same name already exist, the existing resources are used. 
```
python ./.azureml/azure_sdk_commands.py
```

### 4. Launch Streamlit App
* Launch the Streamlit App, which intakes customer data and predicts churn
```
streamlit run dashboard.py
```
## Model
The project uses scikit-learn's gradient boosting classifier trained on features such as Partner, Dependents, tenure, PhoneService, and MultipleLines. The model acheives an F1 score of 0.54 out of sample and 0.83 on the in sample, indicating that it is overfit. In the future this could be remedied through a more careful selection of model, hyperparameter tuning with cross validation, or through using Azure AutoML.  
## Future Improvements
* Hyperparameter tuning/AutoML
* Explore model interpretability
* Create infrastructure for monitoring model and predictions
## Author
Built with💡by John Hutchinson
