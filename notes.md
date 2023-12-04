# Avant nettoyage
total : 6.362.620 rows 

total : 8.213 fraud rows

systematic fraud pattern : Transfer money to an account, then cash it out in another operation

## rapport de construction du modèle

L'accuracy dans le rapport est arrondie, le résultat est affiché à 1 mais correspond en réalité au résultat obtenu plus haut
```
Accuracy: 0.999635370334862
Confusion: [[635409     36]
 [   196    621]]
classification report:               precision    recall  f1-score   support

           0       1.00      1.00      1.00    635445
           1       0.95      0.76      0.84       817

    accuracy                           1.00    636262
   macro avg       0.97      0.88      0.92    636262
weighted avg       1.00      1.00      1.00    636262
```
# Après nettoyage


## rapport de construction du modèle

L'accuracy dans le rapport est arrondie, le résultat est affiché à 1 mais correspond en réalité au résultat obtenu plus haut
```
Accuracy: 0.9995203051597176
Confusion: [[433122     21][187    279]]
classification report:               precision    recall  f1-score   support

           0       1.00      1.00      1.00    433143
           1       0.93      0.60      0.73       466

    accuracy                           1.00    433609
   macro avg       0.96      0.80      0.86    433609
weighted avg       1.00      1.00      1.00    433609
```