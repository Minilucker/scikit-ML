from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import time

def hierarchicalClusteringModeler(dataframe: DataFrame, target: Series, logfile):
 
    timer_start = time.time()
    print('Splitting ...', end="")
    dataframe = dataframe.sample(frac=0.01)
    target = target.sample(frac=0.01)
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target, test_size=0.2, random_state=42)
    print("Done")

    # use the AgglomerativeClustering to properly separate the fraud data from the normal data
    print("Creating model ...", end="")
    model = AgglomerativeClustering()
    print("Done\nBeginning training...", end="")

    # train the model with the given dataset
    model.fit(X_train, y_train)
    print("Done")

    # test the model with a sample of the dataset it was built with
    y_pred = model.fit_predict(X_test)
    
    # accuracy, the percentage of success in the prediction (0 is bad, whereas 1 is 100% accuracy, 1 should not be possible and might be an error)
    acc = accuracy_score(y_test, y_pred)
    
    # confusion, return an array containing 2 arrays, each containing respectively: 
    # - true positive (clean data predicted to be clean data) and false positive (fraud predicted to be clean data),
    # - true negative (fraud data predicted to be fraud data) and false negative (clean data predicted to be fraud data)
    confusion = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'Accuracy: {acc}')
    print(f"Confusion: \n{confusion}")
    print(f'f1 score: {f1}')
    print(f'total time: {time.time() - timer_start}')
    logfile.writelines([f'Accuracy: {acc}\n', f"Confusion: \n{confusion}\n", f'f1 score: {f1}\n', f'total time: {time.time() - timer_start}'])
