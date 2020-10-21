# From notebooks to operational ML pipelines

This repository provides the code base for the [HSG CAS Big data & AI for managers](https://cas-bdai.iwi.unisg.ch/) session on how to operationalize machine learning models (October 2020).

##Pre-requisites

In order to be able to run all the code provided, the following pre-requisites must be met:

* An environment to run Jupyter notebooks in, e.g. [Visual Studio Code ](https://code.visualstudio.com/download) must be available. Also see ["working with Jupyter Notebooks in VS Code"](https://code.visualstudio.com/docs/python/jupyter-support).
* Azure ML Studio workspace must be available. Follow [these instructions](https://github.com/marcscho/amllab/blob/master/0_Intro_Azure_ML.md) to create it. In case you need a trial Azure subscription first, start [here](https://azure.microsoft.com/en-us/free/search/?&ef_id=EAIaIQobChMIyun91ZnF7AIVmLPtCh3duAc9EAAYASAAEgKWNPD_BwE:G:s&OCID=AID2100049_SEM_EAIaIQobChMIyun91ZnF7AIVmLPtCh3duAc9EAAYASAAEgKWNPD_BwE:G:s).

* Once you Azure ML workspace is provisioned, download the workspace config.json file as shown below. Make sure to put the file in the same folder that contains the notebooks.
![shown here:](https://docs.microsoft.com/en-us/azure/machine-learning/media/how-to-configure-environment/configure.png)
[Here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#local) you can find more information on setting up a local development environment to work with Azure ML.
* To run ML pipelines remotely, you will also need to have a service principal account as shown in the "Service Principal Authentication" section in [this notebook](https://github.com/Azure/MachineLearningNotebooks/blob/1f05157d24c8bd9866121b588e75dc95764ae898/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb). The service principal is used in remote runs for authentication and authorization.
* To interact with Azure ML from your local python environment, you must first install the <code>azureml-sdk</code> for python. It can be installed like any other library or package via pip. Siply type <code>!pip install azureml-sdk</code> in one cell of your Jupyter notebook and execute the cell. 
More information on the installation can be found [here](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py).

## Contents

To access the contents of this repo, simply download it from the GitHub website manually or execute <code>git clone https://github.com/marcscho/notebook-to-ml-pipeline</code> from the command line in a folder of your choosing.


| Notebook        | Content   |
| ------------- |-------------|
| [0_register_dataset.ipynb](./0_register_dataset.ipynb)      | Registers the csv file as a dataset in Azure ML |
| [0_register_secret.ipynb](./0_register_secret.ipynb)      | Registers the secret for the service principal in the Azure Keyvault to ensure it is not stored in clear text in a script. This secret is then retrieved by the pipeline runs and used to authenticate as well as to retrieve assets such as models and datasets from the Azure ML workspace. |
| [1_first_model.ipynb](./1_first_model.ipynb) | Trains a first basic model locally using python's sklearn library.      |
| [2_experiment_tracking.ipynb](./2_experiment_tracking.ipynb) | Re-runs the training process of the previous notebook, this time with tracking model metrics inside Azure ML experiments for traceability purposes. To view them, open [Azure ML Studio](http://ml.azure.com) and click "Experiments" in the navigation bar on the left.   |
| [3_predictions.ipynb](./3_predictions.ipynb) | Sets up a ML pipeline for batch scoring. Loads the previously registered dataset and registered model and generates a CSV file with the models predictions which is then uploaded to the Azure ML storage.      |
| [4_deploy_realtime.ipynb](./4_deploy_realtime.ipynb) | Publishes the previously trained model as a REST endpoint in a Docker container instance. This enables getting real-time predictions from the model.      |
| [5_test_ml_endpoint.http](test_ml_endpoint.http) | Can be used to send new data via HTTP request to the model endpoint deployed in the previous notebook. Requires the "HTTP client" extension to be installed in VS Code.

## Clean up

To ensure no unwanted costs are incured, ensure that the endpoint deployed with notebook #4 is deleted. To do so, in [Azure ML Studio](http://ml.azure.com), click "Endpoints" in the navigation bar on the left. Here you will find the endpoint named <code>german-credit-hsg</code>. Highlight it and finally click the "Delete" button in the menu bar just above it.

When no compute resources are running, the Azure ML Studio workspace does not incur any costs.

## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.