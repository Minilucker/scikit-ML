import pandas as pd


def cleanDataset(dataset) :
    df = pd.read_csv(dataset)
    df = df.drop(columns=['step', 'isFlaggedFraud'], axis=1)

    #clear les opérations de cashout incohérente : 
    #nonCoherentCashout = (df['type'] == 'CASH-OUT') & ()
    impossibleTransfer = (df['type'] == 'TRANSFER') & ((df['amount'] > df['oldbalanceOrg']) | df['amount'] + df['oldbalanceDest'] != df['newbalanceDest'])
    nonCoherentCashIn = (df['type'] == 'CASH_IN') & (df['newbalanceOrig'] != df['oldbalanceOrg'] + df['amount'])
    nonCoherentPayment = (df['type'] == 'PAYMENT') & (df['amount'] > df['oldbalanceOrg'])

    #filtereddf = df.drop(df[impossibleTransfer].index)
    #filtereddf.drop(df[nonCoherentCashout].index, inplace=True)
    filtereddf = df.drop(df[nonCoherentCashIn].index)
    filtereddf.drop(df[nonCoherentPayment].index, inplace=True)
    
    print("\nLignes après la filtration :")

    print(filtereddf[(filtereddf['isFraud'] == 1)].sum())
    print(df[(df['isFraud'] == 1)].sum())



cleanDataset('Fraud.csv')