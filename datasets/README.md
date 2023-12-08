# Original dataset
Fraud.csv: https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data

# with null values


in python
```
import pandas as pd

dataframe = pd.read_csv('datasets/Fraud.csv')
for column in dataframe.columns:
    dataframe.loc[dataframe.sample(frac=0.1).index, column] = np.nan
dataframe.to_csv('datasets/Fraud_nulls.csv')

```

# shortened

in python
```
import pandas as pd

dataframe = pd.read_csv('datasets/Fraud.csv')
dataframe.sample(frac=0.1).to_csv('datasets/small_Frauds.csv')
```
