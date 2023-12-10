import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import argparse
import src.cleanr as cleanr
import src.logisticRegressor as logisticRegressor
import src.HierarchicalClustering as HierarchicalClustering
import src.RandomForest as RandomForest
import src.linearRegression as linearRegression
import src.ridgeRegression as ridgeRegression
import src.lassoRegression as lassoRegression
import src.decisionTree as decisionTree
import src.GradientBoosting as GradientBoosting
import src.xgBoost as xgBoost
import src.lightgbmregressor as lightgbmregressor
import src.gaussianMixture as gaussianMixture

# List (tuple) of algorithms available
models = [
    ('Logistic Regression', logisticRegressor.logisticRegressor),
    ('Linear Regression', linearRegression.linearRegressor),
    ('Ridge Regression', ridgeRegression.ridgeRegressor),
    ('Lasso Regression', lassoRegression.lassoRegressor),
    ('Decision Tree', decisionTree.decisionTreeClassifier),
    ('Random Forest', RandomForest.randomForestClassificator),
    ('Gradient Boosting', GradientBoosting.GradientBoostingRegress),
    ('XGBoost', xgBoost.xgbooster),
    ('LightGBM Regressor', lightgbmregressor.lightbgmregressor),
    ('Hierarchical Clustering', HierarchicalClustering.hierarchicalClusteringModeler),
    ('Gaussian Mixture', gaussianMixture.gaussianMixture)
]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='the dataset to use')
args = parser.parse_args()

dataset = args.dataset.replace('..', '').replace('/', '').replace('\\', '')
if not args.dataset:
    print("python3 <location>/trainer.py --dataset your_dataset")
    print("make sure to include your dataset in 'datasets/' folder'")
    exit(1)


# Preprocess the dataset
df = cleanr.cleanDataset(args.dataset)

# Relevant columns
features = df[['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount', 'type']]

# One-hot encode the 'type' column
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_columns = onehot_encoder.fit_transform(features[['type']])
features_encoded = pd.concat([features.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
features_encoded.drop(['type'], axis=1, inplace=True)

# Column to predict the value of
criteria = df['isFraud']


# define the correct log file based on the dataset
if (args.dataset == 'Fraud.csv'):
    logfile = open("originalDatasetLogs.txt", "a")
elif (args.dataset == 'small_Frauds.csv'):
    logfile = open("smallDatasetLogs.txt", "a")
else:
    logfile = open("nullDatasetLogs.txt", 'a')

# Write a delimitation in logfile so it's easier to read
logfile.write(f'{"=" * 40}\n')

# For each model in the tuple, write the model name in logfile, start 
for model_name, model_instance in models:
    print(f'\n{model_name}:\n')
    logfile.write(f'\n\n{model_name}:\n')
    start_time = time.time()
    model_instance(features_encoded, criteria, logfile)
    