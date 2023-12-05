import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import cleanr

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = cleanr.cleanDataset('Fraud.csv')

# Assuming your dataset has a 'fraud' column indicating whether a transaction is fraudulent or not
# If not, you might need labeled data to evaluate the clustering performance
# Fractionate the dataset
fraction = 0.005
labels = data['isFraud'].sample(frac=fraction)

# Drop non-numeric columns if any (assuming transactions are represented by numerical features)
data = data.drop(['nameDest', 'nameOrig'], axis=1)
data = data.sample(frac=fraction)

onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_columns = onehot_encoder.fit_transform(data[['type']])
data_encoded = pd.concat([data.reset_index(drop=True), pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)
data_encoded.drop(['type'], axis=1, inplace=True)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Perform hierarchical clustering
linkage_matrix = linkage(data_scaled, method='ward')

# Plot dendrogram
#dendrogram(linkage_matrix)
#plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('Samples')
#plt.ylabel('Distance')
#plt.show()

# Determine the number of clusters based on the dendrogram
# You can choose a threshold based on the dendrogram to cut the tree and get clusters
# Alternatively, you can use AgglomerativeClustering with a specific number of clusters

# Example using AgglomerativeClustering with a specified number of clusters
# Choose an appropriate number based on your data
cluster_model = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
clusters = cluster_model.fit_predict(data_scaled)

# Evaluate the performance (use silhouette score for unsupervised clustering)
silhouette_avg = silhouette_score(data_scaled, clusters)

print(f'Silhouette Score: {silhouette_avg}')
