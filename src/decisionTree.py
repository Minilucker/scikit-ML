import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def decisionTreeClassifier(dataframe: pd.DataFrame, criteria: pd.Series):

    print('Splitting ...', end="")
    X_train, X_test, y_train, y_test = train_test_split(dataframe, criteria, test_size=0.2, random_state=42)
    print("Done")

    # use the LogisticRegression to properly separate the fraud data from the normal data
    print("Creating model ...", end="")
    model = DecisionTreeClassifier(random_state=42)
    print("Done\nBeginning training...", end="")

    # train the model with the given dataset
    model.fit(X_train, y_train)
    print("Done")

    # test the model with a sample of the dataset it was built with
    y_pred = model.predict(X_test)

    # accuracy, the percentage of success in the prediction (0 is bad, whereas 1 is 100% accuracy, 1 should not be possible and might be an error)
    acc = accuracy_score(y_test, y_pred=y_pred)
    conf = confusion_matrix(y_test, y_pred)

    # confusion, return an array containing 2 arrays, each containing respectively: 
    # - true positive (clean data predicted to be clean data) and false positive (fraud predicted to be clean data),
    # - true negative (fraud data predicted to be fraud data) and false negative (clean data predicted to be fraud data)
    print(f'Accuracy: {acc}')
    print(f'confusion: \n{conf}')