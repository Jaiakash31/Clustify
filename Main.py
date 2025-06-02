import pandas as pd
import numpy as nm
import matplotlib.pyplot as pyt
import seaborn as sea
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import silhouette_score


fl = pd.read_csv('CTrainSet.csv')
x = fl[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scalar = ss()
x_scale = scalar.fit_transform(x)

inr= []
for k in range(1,11):
    km = KMeans(n_clusters=5, random_state=42)
    km.fit(x_scale)
    inr.append(km.inertia_)

pyt.figure(figsize=(8, 5))
pyt.plot(range(1, 11), inr, marker='o' )
pyt.xlabel('Number of Clusters')
pyt.ylabel('Inertia')
pyt.show()

km = KMeans(n_clusters= 5, random_state= 42)
km_labels = km.fit_predict(x_scale)
fl['KMeans Cluster'] = km_labels

sil = silhouette_score(x_scale,km_labels)
print(f"\nSilhoutte Score: {sil:.2f}")

pyt.figure(figsize=(8, 6))
sea.scatterplot(data= fl, x= 'Annual Income (k$)', y= 'Spending Score (1-100)', hue ='KMeans Cluster', palette='Set1', s=100)
pyt.title('KMeans Clusters')
pyt.show()

pyt.figure(figsize=(10, 7))
den = sch.dendrogram(sch.linkage(x_scale, method= 'ward'))
pyt.title('Dendrogram')
pyt.xlabel('Customers')
pyt.ylabel('Euclidean Distance')
pyt.show()

hc = AgglomerativeClustering(n_clusters= 5, metric='euclidean', linkage='ward')
hc_labels = hc.fit_predict(x_scale)
fl['HC Cluster'] = hc_labels

pyt.figure(figsize=(8, 6))
sea.scatterplot(data =fl, x= 'Annual Income (k$)', y= 'Spending Score (1-100)', hue ='HC Cluster', palette='Set1', s=100)
pyt.title('Hierarchical Customer Segmentation')
pyt.show()

n = input("Name the file for the Extended Segments:")
fl.to_csv(n+'.csv', index= False)
print(f"\nExtended Segment Data are saved to '{n}'")