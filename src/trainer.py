import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import cleanr
import logisticRegressor
import HierarchicalClustering
import RandomForest


print("Preparing dataset ...")
df = cleanr.cleanDataset('../datasets/Fraud.csv')

# relevant columns
features = df[['isFlaggedFraud', 'step','oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount', 'type']]

# one-hot encode the 'type' column
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_columns = onehot_encoder.fit_transform(features[['type']])
features_encoded = pd.concat([features.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
features_encoded.drop(['type'], axis=1, inplace=True)

# column to predict the value of
criteria = df['isFraud']
s
model_type: str = input("Choose your model from the following: \nLogistic Regression (lr)\nHierarchical Clustering (hc) \nRandomForest (rf)")
match(model_type):
    case 'lr':
        logisticRegressor.logisticRegressor(features_encoded, criteria)
    case 'hc':
        HierarchicalClustering.hierarchicalClusteringModeler(features_encoded, criteria)
    case 'rf':
        RandomForest.randomForestClassificator(features_encoded, criteria)