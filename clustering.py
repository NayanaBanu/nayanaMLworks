import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
dataset = pd.read_csv(r'WaterQualityData.csv')
dataset.dropna(inplace=True)
df=pd.DataFrame(dataset)
print(df.to_string())
x = dataset.iloc[:, [3, 4]].values
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))
mtp.title("Dendrogrma Plot")
mtp.ylabel("Euclidean Distances")
mtp.xlabel("Site_Id")
mtp.show()
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred= hc.fit_predict(x)
print(y_pred)




