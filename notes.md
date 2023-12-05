systematic fraud pattern : Transfer money to an account, then cash it out in another operation

# Avant nettoyage
total : 6.362.620 rows 

total : 8.213 fraud rows

# Après nettoyage

total : 4.336.089 rows

total frauds : 4116 rows

# Supervised

## Random Forest Classification

Prend très longtemps à entrainer sur ce dataset (+ de 3 heures)

### rapport de construction du modèle

L'accuracy dans le rapport est arrondie, le résultat est affiché à 1 mais correspond en réalité au résultat obtenu plus haut
```
Accuracy: 0.9995203051597176
            #true negative    #false positive   #true positive    #false negative
Confusion: [[433122           21]               [187              279]]

```

## Logistic Regression

Rapide à entrainer (- 1 min)

### Results

```
Accuracy: 0.9995335658201184
            #true negative    #false positive   #true positive    #false negative
Confusion: [[866227           108]              [333              550]]

```
# Unsupervised

## Hierarchical Clustering 

Bien moins précis que la régression et le Tree-Based, ne peux pas utiliser un dataset trop large (celui utilisé ici est égale a 5% du dataset de 4M de lignes) car le modèle utilise des opérations trop complexes et demanderais trop de mémoire pour un dataset plus grand (68 TiB pour 4M de lignes à process), beaucoup de faux positifs ( ce qui reste quand même mieux que beaucoup de faux négatifs)

### Results

```
Accuracy: 0.7461254612546125
            #true negative    #false positive   #true positive    #false negative
Confusion: [[16170           5484]             [20               6]]

```