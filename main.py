import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
import src.kMean as kMean
import src.gaussianMixture as gaussianMixture

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='the dataset to use')
parser.add_argument('-m', '--mode', help='the algorithm to use from the following: \nLogistic Regression (loReg)\nLinear Regression (liReg)\nRidge Regression (riReg)\nLasso Regression (laReg)\nDecision Tree (dt)\nRandomForest (rf)\nGradient Boosting (gbr)\nXGBoost (xgb)\nLightGBM Regressor (lgbr)\n K Mean (kmean)\nHierarchical Clustering (hc)\nGaussian Mixture Model (gmm)\n')
args = parser.parse_args()

dataset = args.dataset.replace('..', '').replace('/', '').replace('\\', '')
if not args.dataset:
    print("python3 <location>/trainer.py --dataset your_dataset")
    print("make sure to include your dataset in 'datasets/' folder'")
    exit(1)

print("Preparing dataset ...")
df = cleanr.cleanDataset(f'{args.dataset}')

# relevant columns
features = df[['isFlaggedFraud', 'step','oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount', 'type']]

# one-hot encode the 'type' column
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_columns = onehot_encoder.fit_transform(features[['type']])
features_encoded = pd.concat([features.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
features_encoded.drop(['type'], axis=1, inplace=True)

# column to predict the value of
criteria = df['isFraud']
model_type: str = args.mode
starting_float = time.time()
match(model_type):
    case 'loReg':
        print("starting in Logistic regression")
        logisticRegressor.logisticRegressor(features_encoded, criteria)
    case 'liReg':
        print("starting in Linear regression mode")
        linearRegression.linearRegressor(features_encoded, criteria)
    case 'riReg':
        print("starting in Ridge regression mode")
        ridgeRegression.ridgeRegressor(features_encoded, criteria)
    case 'laReg': 
        print("starting in Lasso regression mode")
        lassoRegression.lassoRegressor(features_encoded, criteria)
    case 'dt':
        print("starting in Decision Tree mode")
        decisionTree.decisionTreeClassifier(features_encoded, criteria)
    case 'rf':
        print("starting in Random Forest Classifier mode")
        RandomForest.randomForestClassificator(features_encoded, criteria)
    case 'gbr':
        print("starting in Gradient Boosting Regression mode")
        GradientBoosting.GradientBoostingRegress(features_encoded, criteria)
    case 'xgb':
        print("starting in xgbooster mode")
        xgBoost.xgbooster(features_encoded, criteria)
    case 'lgbr':
        print("starting in Light GMB Regression mode")
        lightgbmregressor.lightbgmregressor(features_encoded, criteria)
    case 'kmean':
        print("starting in K Mean Cluster mode")
        kMean.kmeancluster(features_encoded, criteria)
    case 'hc':
        print("starting in Hierarchical Cluster mode")
        HierarchicalClustering.hierarchicalClusteringModeler(features_encoded, criteria)
    case 'gmm':
        print("starting in Gaussian Mixture mode")
        gaussianMixture.gaussianMixture(features_encoded, criteria)

print(f"total time: {time.time() - starting_float}")