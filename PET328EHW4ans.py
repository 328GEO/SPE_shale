import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

file="/Users/farizhajiyev/Downloads/PET328EHW4SPE_shale.xlsx"
data = pd.read_excel(file)
df=data.drop(['Lease', 'Formation','State'], axis=1)


columns_for_clustering = [' Initial Pressure Estimate (psi) ', ' Reservoir Temperature (deg F) ',
                          ' Net Pay (ft) ', 'Porosity', ' Water Saturation ', ' Oil Saturation ',
                          ' Gas Saturation ', ' Gas Specific Gravity ', 'CO2', 'N2',
                          ' TVD (ft) ', ' Spacing ', '# Stages', '# Clusters ',
                          '# Clusters per Stage', '# of Total Proppant (MM Lbs)',
                          ' Lateral Length (ft) ', ' Top Perf (ft) ', ' Bottom Perf (ft) ',
                          ' Sandface Temp (deg F) ', ' Static Wellhead Temp (deg F) ',
                          'Cumulative Gas Produced after 1 year, MCF']


#2.a)
scaler = StandardScaler()
X = scaler.fit_transform(data[columns_for_clustering])
k_values = range(1, 11)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_values, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


#2.b)
X = data[columns_for_clustering]
linked = linkage(X, method='single', metric='euclidean')

plt.figure(figsize=(9, 5))
dendrogram(linked, orientation='top', labels=data['Well Number'].values, distance_sort='descending',
           show_leaf_counts=True)
plt.xlabel('Well Number')
plt.ylabel('Distance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#2.c)
k_values = [5, 10]

for k in k_values:
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(df)
    distances, _ = knn.kneighbors()
    avg_distances = np.mean(distances, axis=1)
    threshold = np.mean(avg_distances) + 2 * np.std(avg_distances)
    anomalies = np.where(avg_distances > threshold)[0]
    print(f"Anomalies with {k} neighbors: {anomalies}")


lof = LocalOutlierFactor(n_neighbors=4, contamination=0.1)  
anomaly_scores = lof.fit_predict(data[columns_for_clustering])
data['Anomaly Score'] = anomaly_scores
anomalies = data[data['Anomaly Score'] == -1]
print("Detected anomalies with LOF:")
print(anomalies)


isolation_forest_1 = IsolationForest(n_estimators=1, contamination='auto', random_state=42)
isolation_forest_1.fit(data[columns_for_clustering])
anomalies_1 = isolation_forest_1.predict(data[columns_for_clustering])
isolation_forest_100 = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
isolation_forest_100.fit(data[columns_for_clustering])
anomalies_100 = isolation_forest_100.predict(data[columns_for_clustering])
data['Anomaly IF 1 Tree'] = anomalies_1
data['Anomaly IF 100 Trees'] = anomalies_100
print("Detected anomalies with 1 tree:")
print(data[data['Anomaly IF 1 Tree'] == -1])
print("Detected anomalies with 100 trees:")
print(data[data['Anomaly IF 100 Trees'] == -1])



