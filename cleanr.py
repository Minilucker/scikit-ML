import pandas as pd


def cleanDataset(dataset) :
    df = pd.read_csv(dataset)

    #retirer les opérations de cashout incohérentes, à savoir : retirer plus d'argent que disponible / avoir plus d'argent sur le compte après le cashout qu'avant  
    nonCoherentCashout = (df['type'] == 'CASH-OUT') & ((df['amount'] > df['oldbalanceOrg']) |( df['newbalanceOrig'] > df['oldbalanceOrg']))
    
    #retirer les opération de transfer incohérentes, à savoir : transférer plus d'argent que disponible sur le compte / le compte de destination ne reçoit pas exactement la somme envoyée
    impossibleTransfer = (df['type'] == 'TRANSFER') & ((df['amount'] > df['oldbalanceOrg']) | (df['amount'] + df['oldbalanceDest'] != df['newbalanceDest']))
    
    #retirer les opérations de cashin incohérentes, à savoir : recevoir plus d'argent que la quantité ajoutée
    nonCoherentCashIn = (df['type'] == 'CASH_IN') & (df['newbalanceOrig'] != df['oldbalanceOrg'] + df['amount'])

    #retirer les opération de paiements incohérentes, à savoir : payer plus d'argent que disponible sur le compte d'origine
    nonCoherentPayment = (df['type'] == 'PAYMENT') & (df['amount'] > df['oldbalanceOrg'])
    
    #drop chaque partie du dataframe selon les conditions

    #créé un nouveau dataframe (inplace = False --> ne modifie pas directement la variable df mais retourne le résultat de l'opération)
    #print('Dropping useless Transfer ...', end="")
    #filtereddf = df.drop(df[impossibleTransfer].index)
    #print('Done')
    ##modifie directement le dataset ou le drop est effectué, car on a déjà créé le nouveau dataframe
    #print('Dropping useless cashout ...', end="")
    #filtereddf.drop(df[nonCoherentCashout].index, inplace=True)
    #print('Done')
    #print('Dropping useless cashin ...', end="")
    #filtereddf.drop(df[nonCoherentCashIn].index, inplace=True)
    #print('Done')
    #print('Dropping useless Payment ...', end="")
    #filtereddf.drop(df[nonCoherentPayment].index, inplace=True)
    #print('Done')

    return df