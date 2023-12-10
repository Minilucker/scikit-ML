import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
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

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='the dataset to use')
parser.add_argument('-m', '--mode', help='the algorithm to use from the following: \nLogistic Regression (loReg)\nLinear Regression (liReg)\nRidge Regression (riReg)\nLasso Regression (laReg)\nDecision Tree (dt)\nRandomForest (rf)\nGradient Boosting (gbr)\nXGBoost (xgb)\nLightGBM Regressor (lgbr)\n K Mean (kmean)\nHierarchical Clustering (hc)\nGaussian Mixture Model (gmm)\n')
args = parser.parse_args()

dataset = args.dataset.replace('..', '').replace('/', '').replace('\\', '')
if not args.dataset:
    print("python3 <location>/trainer.py --dataset your_dataset")
    print("make sure to include your dataset in 'datasets/' folder'")
    exit(1)

#print("Preparing dataset ...")
df = cleanr.cleanDataset(f'{args.dataset}')

# relevant columns
features = df[['step','oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount', 'type']]

# one-hot encode the 'type' column
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_columns = onehot_encoder.fit_transform(features[['type']])
features_encoded = pd.concat([features.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
features_encoded.drop(['type'], axis=1, inplace=True)

# column to predict the value of
criteria = df['isFraud']
model_type: str = args.mode
starting_float = time.time()

if (args.dataset == 'Fraud.csv'):
    logfile = open("originalDatasetLogs.txt", "a")
elif (args.dataset == 'small_Frauds.csv'):
    logfile = open("smallDatasetLogs.txt", "a")
else:
    logfile = open("nullDatasetLogs.txt", 'a')

match(model_type):
    case 'loReg':
        logfile.write("starting in Logistic regression\n")
        print("starting in Logistic regression")
        logisticRegressor.logisticRegressor(features_encoded, criteria, logfile)
    case 'liReg':
        logfile.write("starting in Linear regression mode\n")
        print("starting in Linear regression mode")
        linearRegression.linearRegressor(features_encoded, criteria, logfile)
    case 'riReg':
        logfile.write("starting in Ridge regression mode\n")        
        print("starting in Ridge regression mode")
        ridgeRegression.ridgeRegressor(features_encoded, criteria, logfile)
    case 'laReg': 
        logfile.write("starting in Lasso regression mode\n")
        print("starting in Lasso regression mode")
        lassoRegression.lassoRegressor(features_encoded, criteria, logfile)
    case 'dt':
        logfile.write("starting in Decision Tree mode\n")
        print("starting in Decision Tree mode")
        decisionTree.decisionTreeClassifier(features_encoded, criteria, logfile)
    case 'rf':
        logfile.write("starting in Random Forest Classifier mode\n")
        print("starting in Random Forest Classifier mode")
        RandomForest.randomForestClassificator(features_encoded, criteria, logfile)
    case 'gbr':
        logfile.write("starting in Gradient Boosting Regression mode\n")
        print("starting in Gradient Boosting Regression mode")
        GradientBoosting.GradientBoostingRegress(features_encoded, criteria, logfile)
    case 'xgb':
        logfile.write("starting in xgbooster mode\n")
        print("starting in xgbooster mode")
        xgBoost.xgbooster(features_encoded, criteria, logfile)
    case 'lgbr':
        logfile.write("starting in Light GMB Regression mode\n")
        print("starting in Light GMB Regression mode")
        lightgbmregressor.lightbgmregressor(features_encoded, criteria, logfile)
    case 'hc':
        logfile.write("starting in Hierarchical Cluster mode\n")        
        print("starting in Hierarchical Cluster mode")
        HierarchicalClustering.hierarchicalClusteringModeler(features_encoded, criteria, logfile)
    case 'gmm':
        logfile.write("starting in Gaussian Mixture mode\n")
        print("starting in Gaussian Mixture mode")
        gaussianMixture.gaussianMixture(features_encoded, criteria, logfile)

print(f"total time: {time.time() - starting_float}")