# Dataset 1

https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data
systematic fraud pattern : Transfer money to an account, then cash it out in another operation

## Avant nettoyage
total : 6.362.620 rows 

total : 8.213 fraud rows

## Après nettoyage

total : 4.336.089 rows

total frauds : 4116 rows

## Supervised


### Linear Regression

Rapide à entrainer (- 1 min)

#### Results

```
Accuracy: 0.9990959597240833
Confusion Matrix:
[[866335      0]
 [   784     99]]

```
### Logistic Regression

Rapide à entrainer (- 1 min)

#### Results

```
Accuracy: 0.9994914773447968
Confusion: [[866227    108]
            [   333    550]]
```

### Ridge Regression

Rapide à entrainer (-1 min)

#### Results
```
Accuracy: 0.9990533070363066
Confusion Matrix:
[[899123      2]
 [   850      0]]
```


### Lasso Regression

Rapide à entrainer (-1 min)

#### Results
```
Accuracy: 0.9990533070363066
Confusion Matrix:
[[899123      2]
 [   850      0]]
```
### Decision Tree

Rapide à entrainer (- 3 min)

#### Results 
```
Accuracy: 0.999303313980944
Confusion Matrix:
[[898853    272]
 [   355    495]]
```

### Random Forest Classification

Prend très longtemps à entrainer sur ce dataset (+ de 3 heures)

#### Results

```
Accuracy: 0.9995203051597176
            #true negative    #false positive   #true positive    #false negative
Confusion: [[433122           21]               [187              279]]

```
### Gradient Boosting Regression

#### Results

```
Accuracy: 0.9994188312511963
Confusion: [[866324     11]
            [   493    390]]
```
### XGBoost

#### Results
```
Accuracy: 0.9995537454250257
Confusion: [[866311     24]
 [   363    520]]
```
### LightGMB Regressor

#### Results 
```
Accuracy: 0.9995698890013814
Confusion: [[866318     17]
            [   356    527]]
```
## Unsupervised

### Hierarchical Clustering 

Non fiable car il n'est pas possible de tester avec suffisamment de données sans avoir une énorme quantité de RAM, avec 16 Go de RAM on peut aller jusqu'a 1% du dataset de 4M de ligne, ce qui donne une accuracy qui varie de 0.1 a 0.9, selon les données de test.

#### Results

```
Accuracy: 0.9362389023405973
Confusion: [[8120  547]
            [   6    0]]
```

### Gaussian Mixture

#### Results
```
Accuracy: 0.9989818015769968
confusion: [[866335      0]
            [   883      0]]
```

### KNeighbourClassifier

#### Results
```
Accuracy: 0.9993346540316276
```