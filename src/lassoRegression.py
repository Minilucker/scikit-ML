import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, confusion_matrix

def lassoRegressor(dataframe: pd.DataFrame, criteria: pd.Series):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataframe, criteria, test_size=0.2, random_state=42)

    # Initialize and fit the Lasso regression model
    model = Lasso(alpha=1.0)  # You can adjust the alpha parameter for regularization strength
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Convert predicted values to binary (fraud or not fraud)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_binary)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)

