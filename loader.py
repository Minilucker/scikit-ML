import pandas as pd
import joblib  # If you're using scikit-learn version 0.23 or later, use 'from sklearn import joblib'
from sklearn.preprocessing import OneHotEncoder

# Load the saved model
model = joblib.load('fraudDetector.joblib')

# Load the dataset to test the model
test_data = pd.read_csv('cleaned.csv')  # Replace 'test_dataset.csv' with the actual file name

# Assuming the 'type' column needs to be encoded
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_columns = onehot_encoder.fit_transform(test_data[['type']])
test_data_encoded = pd.concat([test_data.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
test_data_encoded.drop(['type'], axis=1, inplace=True)

# Make predictions
predictions = model.predict(test_data_encoded)

# Display the predictions
print("Predictions:")
print(predictions)
