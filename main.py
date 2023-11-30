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
col = [['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df['isFraud']

# Create a column transformer for preprocessing


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(col, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
