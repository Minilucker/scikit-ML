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

Très Rapide à entrainer (2.28515887260437 s)

#### Results

```
Accuracy: 0.9990959597240833
Confusion Matrix:
[[866335      0]
 [   784     99]]

```
### Logistic Regression

Rapide à entrainer (27.996023178100586 s)


#### Results

```
Accuracy: 0.9994914773447968
Confusion: [[866227    108]
            [   333    550]]
```

### Ridge Regression
alpha=1
Rapide à entrainer (1.7530155181884766 s)

#### Results
```
Accuracy: 0.9990959597240833
Confusion Matrix:
[[866335      0]
 [   784     99]]
```


### Lasso Regression

Rapide à entrainer (total time: 515.2598950862885 s)

#### Results
```
Accuracy: 0.9990752036973403
Confusion:
[[866335      0]
 [   802     81]]
```
### Decision Tree


Rapide à entrainer (68.54882574081421 s)

#### Results 
```
Accuracy: 0.9995779607895593
confusion: 
[[866188    147]
 [   219    664]]
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

total time: 6.61203670501709 s

#### Results 
```
Accuracy: 0.9995698890013814
Confusion: 
[[866318     17]
 [   356    527]]

```
## Unsupervised

### Hierarchical Clustering 

total time: total time: 88.80291199684143

Non fiable car il n'est pas possible de tester avec suffisamment de données sans avoir une énorme quantité de RAM, avec 16 Go de RAM on peut aller jusqu'a 1% du dataset de 4M de ligne, ce qui donne une accuracy qui varie de 0.1 a 0.9, selon les données de test.

#### Results

```
Accuracy: 0.9524962527383835
Confusion: [[8261  405]
            [   7    0]]
```

### Gaussian Mixture

total time: 6.4790003299713135

#### Results
```
Accuracy: 0.9989818015769968
confusion:
        [[866335      0]
        [   883      0]]
```
______________________________________________________________________________________________________________________________

# Dataset 2 (Fraud_nulls.csv avec des valeurs à null)

https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data
systematic fraud pattern : Transfer money to an account, then cash it out in another operation

with Null value injected (10% of all value per column is null)

## Avant nettoyage
total : 6.362.620 rows 

total : 8.213 fraud rows

## Après nettoyage

total : 4.499.874 rows

total frauds : 4058 rows

## Supervised


### Linear Regression

total time: 2.96699595451355 s

#### Results

```
Accuracy: 0.9990521958943304
Confusion: 
[[899122      3]
 [   850      0]]

```
### Logistic Regression

total time: 19.529011964797974 s


#### Results

```
Accuracy: 0.9990555293202589
Confusion:
[[899125      0]
 [   850      0]]
```

### Ridge Regression
alpha=1
2.3770225048065186 s

#### Results
```
Accuracy: 0.9990959597240833
Confusion Matrix:
[[866335      0]
 [   784     99]]
```


### Lasso Regression
alpha=1 
total time: 23.51989769935608

#### Results
```
Accuracy: 0.9990521958943304
Confusion: 
[[899122      3]
 [   850      0]]
```
### Decision Tree


total time: 76.21825218200684 s

#### Results 
```
Accuracy: 0.9993055362648963
confusion: 
[[898856    269]
 [   356    494]]
```

### Random Forest Classification

total time: 25m24s

#### Results

```
Accuracy: 0.9994888746909636
Confusion: 
[[899084     41]
 [   419    431]]

```
### Gradient Boosting Regression

total time: 23 min 34 s
#### Results

```
Accuracy: 0.9992122003388983
Confusion: 
[[899110     15]
 [   694    156]]
```
### XGBoost

total time: 12.67868447303772

#### Results
```
Accuracy: 0.9994399844440124
Confusion: 
[[899079     46]
 [   458    392]]
```
### LightGMB Regressor

total time: 5.226939916610718

#### Results 
```
Accuracy: 0.9994466512958693
Confusion: 
[[899074     51]
 [   447    403]]

```
## Unsupervised

### Hierarchical Clustering 

total time: 58.74644351005554

Non fiable car il n'est pas possible de tester avec suffisamment de données sans avoir une énorme quantité de RAM, avec 16 Go de RAM on peut aller jusqu'a 1% du dataset de 4M de ligne, ce qui donne une accuracy qui varie de 0.01 a 0.9, selon les données de test.

#### Results

```
Accuracy: 0.06911111111111111
Confusion:
[[ 616 8377]
 [   1    6]]
```

### Gaussian Mixture

total time: 5.718069553375244

#### Results
```
Accuracy: 0.9990555293202589
confusion: 
[[899125      0]
 [   850      0]]
```
______________________________________________________________________________________________________________________________

# Dataset 3 (small_Frauds.csv 10% du dataset original)

https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data
systematic fraud pattern : Transfer money to an account, then cash it out in another operation

10% of total length (Fraud.csv)

## Avant nettoyage
total : 636.262 rows 

total : 844 fraud rows

## Après nettoyage

total : 433.985 rows

total : 428 fraud rows

## Supervised


### Linear Regression

total time: 0.1939992904663086 s

#### Results

```
Accuracy: 0.9989976612094889
Confusion:
[[86699     0]
 [   87    11]]

```
### Logistic Regression

total time: 3.40999436378479 s


#### Results

```
Accuracy: 0.9997234927474452
Confusion:
[[86693     6]
 [   18    80]]
```

### Ridge Regression
alpha=1
total time: 0.13901257514953613

#### Results
```
Accuracy: 0.9989976612094889
Confusion:
[[86699     0]
 [   87    11]]
```


### Lasso Regression
alpha=1 
total time: 17.11948323249817

#### Results
```
Accuracy: 0.9989861400739657
Confusion:
[[86699     0]
 [   88    10]]
```
### Decision Tree

total time: 3.7720017433166504 s

#### Results 
```
Accuracy: 0.9992050416489049
confusion:
[[86670    29]
 [   40    58]]
```

### Random Forest Classification

total time: 69.72503089904785 s

#### Results

```
Accuracy: 0.9994469854948904
Confusion: 
[[86698     1]
 [   47    51]]

```
### Gradient Boosting Regression

total time: 147.02206444740295 s
#### Results

```
Accuracy: 0.9991819993778587
Confusion: 
[[86676    23]
 [   48    50]]
```
### XGBoost

total time: 1.0880804061889648 s

#### Results
```
Accuracy: 0.9993663375462286
Confusion: 
[[86693     6]
 [   49    49]]
```
### LightGMB Regressor

total time: 0.6339993476867676 s

#### Results 
```
Accuracy: 0.9994239432238441
Confusion: 
[[86695     4]
 [   46    52]]

```
## Unsupervised

### Hierarchical Clustering 

total time: 0.35699987411499023 s

Non fiable car il n'est pas possible de tester avec suffisamment de données sans avoir une énorme quantité de RAM, avec 16 Go de RAM on peut aller jusqu'a 1% du dataset de 4M de ligne, ce qui donne une accuracy qui varie de 0.01 a 0.9, selon les données de test.

#### Results

```
Accuracy: 0.9009216589861752
Confusion:
[[782  85]
 [  1   0]]
```

### Gaussian Mixture

total time: 0.6300051212310791

#### Results
```
Accuracy: 0.9988709287187345
confusion:
[[86699     0]
 [   98     0]]
```