import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import datasets
import cleanr

# Create a sample dataset
df = cleanr.cleanDataset('Fraud.csv')
mpl.rcParams['agg.path.chunksize'] = 10000

d = df.drop('isFraud', axis=1)
y = df['isFraud']


# Identify categorical columns and numerical columns
categorical_cols = d.select_dtypes(include=['string']).columns
numerical_cols = d.select_dtypes(include=['number']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(d, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

plt.scatter(X_test['amount'], y_test, color='black')
plt.plot(X_test['amount'], y_pred, color='blue', linewidth=1)
plt.xlabel('d')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()
