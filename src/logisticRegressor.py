import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

## clean dataset of useless data
#df = cleanr.cleanDataset('Fraud.csv')
#
## relevant columns
#features = df[['isFlaggedFraud', 'step','oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount', 'type']]
#
## one-hot encode the 'type' column
#onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
#encoded_columns = onehot_encoder.fit_transform(features[['type']])
#features_encoded = pd.concat([features.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
#features_encoded.drop(['type'], axis=1, inplace=True)
#
## column to predict the value of
#criteria = df['isFraud']

def logisticRegressor(relevant_columns, target, logfile):

    print('Splitting ...', end="")
    X_train, X_test, y_train, y_test = train_test_split(relevant_columns, target, test_size=0.2, random_state=42)
    print("Done")

    # use the LogisticRegression to properly separate the fraud data from the normal data
    print("Creating model ...", end="")
    # setting max_iteration number to reach convergence(= the most optimum result with the least failure,
    # whereas with 100 max_iter we can only reach what's called local optima a.k.a the most optimised model for the 100 given iterations)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    print("Done\nBeginning training...", end="")

    # train the model with the given dataset
    model.fit(X_train, y_train)
    print("Done")

    # test the model with a sample of the dataset it was built with
    y_pred = model.predict(X_test)

    # accuracy, the percentage of success in the prediction (0 is bad, whereas 1 is 100% accuracy, 1 should not be possible and might be an error)
    acc = accuracy_score(y_test, y_pred=y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # confusion, return an array containing 2 arrays, each containing respectively: 
    # - true positive (clean data predicted to be clean data) and false positive (fraud predicted to be clean data),
    # - true negative (fraud data predicted to be fraud data) and false negative (clean data predicted to be fraud data)

    print(f'Accuracy: {acc}\n')
    print(f"Confusion: \n{confusion}\n")
    print(f'f1 score: {f1_score(y_test, y_pred)}\n')

    logfile.writelines([f'Accuracy: {acc}\n', f"Confusion: \n{confusion}\n", f'f1 score: {f1_score(y_test, y_pred)}\n'])