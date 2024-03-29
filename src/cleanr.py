import pandas as pd
from pandas import DataFrame, Series
import numpy as np


#inject null values in a datasets
def injectNullValues(dataframe: DataFrame):
    for column in dataframe.columns:
        dataframe.loc[dataframe.sample(frac=0.1).index, column] = np.nan
    dataframe.to_csv('datasets/Fraud_nulls.csv')

def isBoolColumn(column: Series):
    column_values = column
    column_minValue = 0
    column_maxValue = 0
    for value in column_values:
        if value > column_maxValue: column_maxValue = value
        if value < column_minValue: column_minValue = value    
    # je considère qu'une colonne contenant des vrais float n'aura jamais ses valeure maximales et minimale respectivement à 1 ou 0 et 0
    if column_minValue == 0 and (column_maxValue == 1 or column_maxValue == 0):
        return True
    else:
        return False


def isNullCellInColumn(column: Series):
    nullcounter = column.isnull().sum()
    if nullcounter > 0:
        return True
    else: 
        return False
    

# fonction pour auto imput des valeurs lorsqu'une cellule est null, 
# ne prend pas en compte les relation entre chaque colonne d'une même ligne
def autoImputNullValuesBasedOnType(dataframe: DataFrame):
    for column_name in dataframe.columns:
            if not isNullCellInColumn(dataframe[column_name]):
                continue
            #print("----------------------------------------------------------------------------------------------------")
            #print(f"Null found in column with name: {column_name}")
            # on ne check que si c'est de type float64, car si il y a un null la colonne devient automatiquement de type float64 a moins d'être un dtype object
            # à améliorer selon les autres types possibles
            if dataframe[column_name].dtype != 'float64':
                #print(f"Column {column_name} is of type String")
                available_values = dataframe[column_name].dropna().unique()
                #print(f"Imputing random value from these: {available_values} ...", end=" ")
                dataframe[column_name].fillna(np.random.choice(available_values), inplace=True)
                #print("Done")
                #print("----------------------------------------------------------------------------------------------------")
            elif dataframe[column_name].dtype == 'float64' and checkIfTrulyFloatValue(dataframe[column_name]):
                #print(f"Column {column_name} is of type Float")
                #print(f"Imputing mean value from column {column_name}")
                dataframe[column_name].fillna(dataframe[column_name].mean(), inplace=True)
                #print("----------------------------------------------------------------------------------------------------")
            else: 
                #print(f"Column {column_name} is neither String nor Float, finding out...", end=" ")
                if isBoolColumn(dataframe[column_name]):
                    #print(f"Column {column_name} is a boolean column")
                    dataframe[column_name].fillna(value=0, inplace=True)
                    #print("----------------------------------------------------------------------------------------------------")
                # cas où c'est un int64
                else: 
                    #print(f"Column {column_name} is an int column")
                    dataframe[column_name] = dataframe[column_name].interpolate(method='linear').astype(int)
                    #print(f"added value based on neighbour values")
                    #print("----------------------------------------------------------------------------------------------------")
            

def checkIfTrulyFloatValue(series):
    digits_after_decimal = []
    for float_value in series:
        if str(float_value) == 'nan': 
            continue
        digits_after_decimal.append(int((float_value % 1) * 10**len(str(float_value).split(".")[1])))
    if sum(digits_after_decimal) != 0:
        return True
    else:
        return False


def clearWrongValues(dataframe: DataFrame):
   
    #retirer les opération de transfer incohérentes, à savoir : transférer plus d'argent que disponible sur le compte
    #définition de la condition du prochain filtre pour qu'elle soit définie selon le dataset actuel (avec les filtres précédents)
    impossibleTransfer = (dataframe['type'] == 'TRANSFER') & ((dataframe['amount'] > dataframe['oldbalanceOrg']))
    
    dataframe.drop(dataframe[impossibleTransfer].index, inplace=True)

    #retirer les opérations de cashout incohérentes, à savoir : retirer plus d'argent que disponible 
    #définition de la condition du prochain filtre pour qu'elle soit définie selon le dataset actuel (avec les filtres précédents)
    nonCoherentCashout = (dataframe['type'] == 'CASH-OUT') & ((dataframe['amount'] > dataframe['oldbalanceOrg']))

    dataframe.drop(dataframe[nonCoherentCashout].index, inplace=True)

    #retirer les opération de paiements incohérentes, à savoir : payer plus d'argent que disponible sur le compte d'origine
    #définition de la condition du prochain filtre pour qu'elle soit définie selon le dataset actuel (avec les filtres précédents)
    nonCoherentPayment = (dataframe['type'] == 'PAYMENT') & (dataframe['amount'] > dataframe['oldbalanceOrg'])

    dataframe.drop(dataframe[nonCoherentPayment].index, inplace=True)

def cleanDataset(dataset: str) :
    df = pd.read_csv(f"datasets/{dataset}")

    autoImputNullValuesBasedOnType(df)
    clearWrongValues(df)

    df.drop('isFraud', axis=1).to_csv(f"testing_datasets/testing_{dataset}")

    return df

#pour générer un dataset avec des valeurs null
#injectNullValues(pd.read_csv('datasets/Fraud.csv'))