import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score
import cleanr

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
#data = cleanr.cleanDataset('Fraud.csv')
#
## Assuming your dataset has a 'fraud' column indicating whether a transaction is fraudulent or not
## If not, you might need labeled data to evaluate the clustering performance
## Fractionate the dataset
#labels = data['isFraud'].sample(frac=fraction)
#
## Drop non-numeric columns if any (assuming transactions are represented by numerical features)
#data = data.drop(['nameDest', 'nameOrig'], axis=1)
#
#
#onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
#encoded_columns = onehot_encoder.fit_transform(data[['type']])
#data_encoded = pd.concat([data.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
#data_encoded.drop(['type'], axis=1, inplace=True)

def hierarchicalClusteringModeler(relevant_columns: DataFrame, target: Series):
    fraction = 0.005
    relevant_columns = relevant_columns.sample(frac=fraction)
    target = target.sample(frac=fraction)
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(relevant_columns)

    # Perform hierarchical clustering
    linkage_matrix = linkage(data_scaled, method='ward')


    # Determine the number of clusters based on the dendrogram
    # You can choose a threshold based on the dendrogram to cut the tree and get clusters
    # Alternatively, you can use AgglomerativeClustering with a specific number of clusters

    cluster_model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    clusters = cluster_model.fit_predict(data_scaled)


    accuracy_avg = silhouette_score(data_scaled, clusters)
    confusion = confusion_matrix(data_scaled, clusters)

    print(f'silhouette Score: {accuracy_avg}')
    print(f'Confusion: {confusion}')

        # Plot dendrogram
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()
