import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cleanr

# Create a sample dataset
df = cleanr.cleanDataset('Fraud.csv')


d = df.drop('isFraud', axis=1)
features = df[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount']]
criteria = df['isFraud']

# Split the data into training and testing sets
print('Splitting ...', end="")
X_train, X_test, y_train, y_test = train_test_split(features, criteria, test_size=0.2, random_state=42)
print("Done")
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Create a linear regression model
print("Creating model ...", end="")
model = RandomForestClassifier(n_estimators=1000)
print("Done")


# Train the model on the training set
print('training ...', end="")
model.fit(X=X_train,y=y_train)
print("Done")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred=y_pred)
confusion = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)

print(f'classification report: {class_rep}')
print(f'Confusion: {confusion:.2f}')
print(f'Accuracy: {acc:.2f}')
