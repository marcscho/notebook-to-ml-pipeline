
### batch_score.py
### --------------------

from azureml.core import Workspace, Model, Dataset, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication #
import joblib
import pandas as pd
from datetime import datetime

subscription_id = '8c386bb9-fbb5-45dd-a9cd-2ca847235881'
resource_group = 'rg-aml-ws-ga'
workspace_name = 'aml-ws-ga'

####
 
#svc_pr_password = run.get_secret('mlhero-aml-ws-ga')
svc_pr_password = 'GdCl.3VDLIc.O_0DQxMBxKRaw9k8D~hMps'

svc_pr = ServicePrincipalAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47", service_principal_id="c8ef5ed8-7a7d-4831-984e-571b9c646603", service_principal_password=svc_pr_password)

###

ws = Workspace(subscription_id, resource_group, workspace_name, auth = svc_pr)

### Load data for scoring
df = Dataset.get_by_name(ws, 'german_credit_hsg').to_pandas_dataframe()
df.drop('Sno', axis = 1, inplace = True)
new_data = df[9:16]

### Load model for scoring
model = Model(workspace = ws, name='german-credit-hsg')
model.download()
loaded_model = joblib.load('model.pkl')

### Score new data
results = loaded_model.predict(new_data)
new_data['prediction'] = results

### write output csv
now = datetime.now()
now = now.strftime("%Y_%m_%d__%H_%M_%S")
filename = now + '.csv'
new_data.to_csv(filename)

### upload csv to datastore
ds = Datastore.get_default(ws)
ds.upload_files([filename], target_path = './predictions')
