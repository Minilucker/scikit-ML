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
            #true negative    #false positive   #true positive    #false negative
Confusion: [[866227           108]              [333              550]]

```

### Ridge Regression

Rapide à entrainer (-1 min)

#### Results

Accuracy: 0.9990533070363066
Confusion Matrix:
[[899123      2]
 [   850      0]]



### Lasso Regression

Rapide à entrainer (-1 min)

#### Results

Accuracy: 0.9990533070363066
Confusion Matrix:
[[899123      2]
 [   850      0]]

### Decision Tree

Rapide à entrainer (- 3 min)

#### Results 

Accuracy: 0.999303313980944
Confusion Matrix:
[[898853    272]
 [   355    495]]


### Random Forest Classification

Prend très longtemps à entrainer sur ce dataset (+ de 3 heures)

#### Results

```
Accuracy: 0.9995203051597176
            #true negative    #false positive   #true positive    #false negative
Confusion: [[433122           21]               [187              279]]

```

## Unsupervised

### Hierarchical Clustering 

Bien moins précis que la régression et le Tree-Based, ne peux pas utiliser un dataset trop large (celui utilisé ici est égale a 5% du dataset de 4M de lignes) car le modèle utilise des opérations trop complexes et demanderais trop de mémoire pour un dataset plus grand (68 TiB pour 4M de lignes à process), beaucoup de faux positifs ( ce qui reste quand même mieux que beaucoup de faux négatifs)

#### Results

```
Accuracy: 0.7461254612546125
            #true negative    #false positive   #true positive    #false negative
Confusion: [[16170           5484]             [20               6]]

```