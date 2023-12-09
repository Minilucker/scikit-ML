# Compte rendu Analyse de Fraude financière

## Introduction:

Dans le cadre du projet d’analyse de fraude financière, j’ai pu expérimenter de différentes façons avec des algorithmes de machine learning, qu’ils soient supervisés ou non supervisés, l’objectif de ce projet était d’apprendre à manier les différents algorithmes via scikit-learn et python afin de pouvoir les utiliser dans un contexte qui leur est propice (exemple: analyse de logs, analyse de trafic réseau) si le besoin s’en fait sentir.

Plusieurs sujets vont être abordés dans ce compte rendu comme le nettoyage des données, la réduction de la dimensionnalité, ou encore quel algorithme est à privilégier dans le cadre de la détection de Fraude financière.

## Présentation du Dataset:


Fraudulent Transactions Data | Kaggle

Avant de faire comprendre un dataset à un modèle de machine learning, il est selon moi nécessaire de comprendre comment le dataset est organisé et comment le comprendre.

Ce dataset suit la structure suivante:
step (unité temporelle, step 1 = 1ere heure d’enregistrement, step 2 = 2e heure etc.)
type (enumération suivante: 	CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
amount (quantité d’argent de la transaction)
nameOrig (nom de l’origine de l’argent, un id sous la forme ‘CXXXXXXXX’)
nameDest (nom de la destination de l’argent, un id sous la forme ‘CXXXXXXXXXX’ ou ‘MXXXXXXXXXX’ pour les marchands dans le cas de PAYMENT (le type))
oldbalanceOrig (balance initiale du propriétaire de l’argent)
newbalanceOrig (nouvelle balance du propriétaire de l’argent)
oldbalanceDest (balance initiale du destinataire)
newbalanceDest (nouvelle balance du destinataire)
isFraud (flag à 0 ou 1 indiquant si la transaction est une fraude (1) ou non (0))
isFlaggedFraud (flag à 0 ou 1 indiquant que la transaction dépasse 200 000, n’est pas vraiment corrélé au cas de fraude, seulement 16 transactions ont ce flag à 1 et isFraud à 1 simultanément)

De plus, on nous donne l’information suivante dans la description du dataset concernant le flag isFraud : “This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.”




## Nettoyage

A partir de cette indication, on sait déjà que les fraudes seront effectuées en 2 temps, d’abord avec une opération TRANSFER, puis avec un opération CASH-OUT, cela va nous permettre d’analyser plus en détail potentiellement le nom de destinataire et les nom d’origine de transaction pour voir si ces derniers sont corrélés entre un TRANSFER et un CASH-OUT (si le destinataire d’un TRANSFER se retrouve a CASH-OUT le même montant par exemple, mais …. non, étrangement ce n’est pas corrélé, ce qui est probablement une conséquence des datasets synthétiques, toutes les données ne sont pas forcément parfaitement logique.

En sachant cela on peut déjà procéder à un premier nettoyage, on peut drop les colonnes “nameOrig” et “nameDest” puisqu’elle n’ont aucun rapport avec les transactions et sont généré purement aléatoirement, autrement cela constituerait du bruit inutile pour le modèle et au final pourrait contribuer à du potentiel overfitting.

Après cela, on va définir des fonctions pour nettoyer plus précisément le dataset, par exemple, en explorant un peu plus profondément les données contenu dans ce dataset, j’ai très vite remarqué des transactions qui n’avaient absolument aucun sens, par exemple:
1,TRANSFER,77957.68,C207471778,0.0,0.0,C1761291320,94900.0,22233.65,0,0

ici la transaction consiste à transférer 77957.0 depuis un compte vide, ce qui ne fait évidemment aucun sens, on peut donc définir des règles pour supprimer les transactions dont amount est supérieur à oldbalanceOrig, dans le même style, certaine transactions transfèrent un montant précis vers un compte sans que ce dernier ne voit sa nouvelle balance augmenté (uniquement pour les TRANSFER ici, les CASH-OUT n’update pas le compte de destination, pareil pour les PAYMENT ou les CASH-IN ou encore DEBIT), idéalement il faudrait supprimer ce genre de transaction mais les fraudes se passant par paires de transaction, on se retrouverait avec des CASH-OUT non corrélé avec un TRANSFER, on va donc partir du principe que le fait que le TRANSFER n’update pas le compte de destination n’est pas une erreur à prendre en compte.

## Tests des différents datasets:

Pour juger l’efficacité d’un dataset, plusieurs metrics peuvent être utilisées, on à par exemple la précision, qui va jauger à quel point le model à prédit correctement la catégorie de la fraude, cette précision sera le plus souvent élevée puisque la plupart du temps, une transaction clean sera jugée par le model comme une transaction clean car le dataset est fortement déséquilibré, l’accuracy n’est donc pas ce qui va être priorisée comme métrique, une autre métrique plus intéressante est le recall qui va jauger le nombre de fraude trouvées par rapport aux nombre de non fraude identifié comme des fraudes, les métriques les plus intéressantes dans le contexte de la fraude financière vont être la matrice de confusion et le f1 score (qui est un taux calculé à partir du recall et de la précision, plus le f1 est élevé, plus on peut comprendre que le recall et la précision sont élevé derrière, et donc que le modèle est fiable ou non). parmi tous les algorithmes que j’ai pu tester, 

