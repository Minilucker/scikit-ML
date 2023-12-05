import pandas as pd
import joblib  
from sklearn.preprocessing import OneHotEncoder

# load model   
model = joblib.load('fraudDetector.joblib')

# load dataset
test_data = pd.read_csv('cleaned.csv')  # Replace 'test_dataset.csv' with the actual file name

# encoding type column with OneHotEncoding (columns type: PAYMENT/TRANSFER/DEBIT/CASH_OUT/CASH_IN become type.DEBIT: 0/1, type.PAYMENT: 0/1, type.TRANSFER: 0/1, type.CASH_OUT: 0/1, TYPE.CASH_IN: 0/1)
# this encoding is only possible because this column has few variable values, wouldn't be possible with nameDest or nameOrig (which aren't correlated in the dataset anyway so kinda irrelevant)
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_columns = onehot_encoder.fit_transform(test_data[['type']])
test_data_encoded = pd.concat([test_data.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
test_data_encoded.drop(['type'], axis=1, inplace=True)

# Make predictions
predictions = model.predict(test_data_encoded)

# Display the predictions
print("Predictions:")
print(predictions)
