import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def linearRegressor(dataframe, target):
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target, test_size=0.2, random_state=42)

    # Initialize and fit the linear regression model
    model = LinearRegression()
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
