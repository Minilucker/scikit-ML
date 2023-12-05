
import pandas as pd
from pandas import DataFrame
import numpy as np

# fonction pour auto imput des valeurs lorsqu'une cellule est null, 
# ne prend pas en compte les relation entre chaque colonne d'une même ligne
def autoImputNullValuesBasedOnType(dataframe: DataFrame):
    for column_name in dataframe.columns:
            if dataframe[column_name].dtype != 'int64' and dataframe[column_name].dtype != 'float64':
                available_values = dataframe[column_name].dropna().unique()
                dataframe[column_name].fillna(np.random.choice(available_values), inplace=True)
            elif dataframe[column_name].dtype == 'float64' and checkIfTrulyFloatValue(dataframe[column_name]):
                dataframe[column_name].fillna(dataframe[column_name].mean(), inplace=True)
            else: 

                column_values = dataframe[column_name]
                column_minValue = 0
                column_maxValue = 0
                for value in column_values:
                    if value > column_maxValue: column_maxValue = value
                    if value < column_minValue: column_minValue = value
                # je considère qu'une colonne contenant des vrais float n'aura jamais ses valeure maximales et minimale respectivement à 1 ou 0 et 0
                if column_minValue == 0 and (column_maxValue == 1 or column_maxValue == 0):
                    print("column is boolean")
                    dataframe[column_name].fillna(value=0, inplace=True)
                else: 
                    dataframe[column_name] = dataframe[column_name].interpolate(method='linear').astype(int)
            

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


def cleanDataset(dataset) :
    df = pd.read_csv(dataset)

    df.drop(['nameDest', 'nameOrig'], axis=1)

    
    
    #drop chaque partie du dataframe selon les conditions

    #créé un nouveau dataframe (inplace = False --> ne modifie pas directement la variable df mais retourne le résultat de l'opération)
    filtereddf = df 
    nullcounter = 0
    for column_name in filtereddf.columns:
        nullcounter += filtereddf[column_name].isnull().sum()

    if nullcounter > 0:
        autoImputNullValuesBasedOnType(filtereddf)
        


    print(filtereddf)

    #retirer les opération de transfer incohérentes, à savoir : transférer plus d'argent que disponible sur le compte / le compte de destination ne reçoit pas exactement la somme envoyée
    #définition de la condition du prochain filtre pour qu'elle soit définie selon le dataset actuel (avec les filtres précédents)
    #impossibleTransfer = (filtereddf['type'] == 'TRANSFER') & ((filtereddf['amount'] > filtereddf['oldbalanceOrg']) | (filtereddf['amount'] + filtereddf['oldbalanceDest'] != filtereddf['newbalanceDest']))
    #
    ###modifie directement le dataset ou le drop est effectué, car on a déjà créé le nouveau dataframe
    #print('Dropping useless Transfer ...', end="")
    #filtereddf.drop(df[impossibleTransfer].index, inplace=True)
    #print('Done')
    #
    ##retirer les opérations de cashout incohérentes, à savoir : retirer plus d'argent que disponible / avoir plus d'argent sur le compte après le cashout qu'avant  
    ##définition de la condition du prochain filtre pour qu'elle soit définie selon le dataset actuel (avec les filtres précédents)
    #nonCoherentCashout = (filtereddf['type'] == 'CASH-OUT') & ((filtereddf['amount'] > filtereddf['oldbalanceOrg']) |( filtereddf['newbalanceOrig'] > filtereddf['oldbalanceOrg']))
    #
    #print('Dropping useless cashout ...', end="")
    #filtereddf.drop(filtereddf[nonCoherentCashout].index, inplace=True)
    #print('Done')
#
    ##retirer les opérations de cashin incohérentes, à savoir : recevoir plus d'argent que la quantité ajoutée
    ##définition de la condition du prochain filtre pour qu'elle soit définie selon le dataset actuel (avec les filtres précédents)
    #nonCoherentCashIn = (filtereddf['type'] == 'CASH_IN') & (filtereddf['newbalanceOrig'] != filtereddf['oldbalanceOrg'] + filtereddf['amount']) 
    #
    #print('Dropping useless cashin ...', end="")
    #filtereddf.drop(filtereddf[nonCoherentCashIn].index, inplace=True)
    #print('Done')
#
    ##retirer les opération de paiements incohérentes, à savoir : payer plus d'argent que disponible sur le compte d'origine
    ##définition de la condition du prochain filtre pour qu'elle soit définie selon le dataset actuel (avec les filtres précédents)
    #nonCoherentPayment = (filtereddf['type'] == 'PAYMENT') & (filtereddf['amount'] > filtereddf['oldbalanceOrg'])
    #
    #print('Dropping useless Payment ...', end="")
    #filtereddf.drop(filtereddf[nonCoherentPayment].index, inplace=True)
    #print('Done')
#
    #filtereddf.drop('isFraud', axis=1).to_csv('cleaned.csv')
#
    #print(f"total number of Rows: {filtereddf['step'].__len__()}")
    #print(f"total number of Frauds: {filtereddf[filtereddf['isFraud'] == 1]['isFraud'].sum()}")
    
    return filtereddf

cleanDataset('test.csv')