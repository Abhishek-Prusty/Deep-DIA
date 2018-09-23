import numpy as np 
from sklearn.cluster import KMeans,FeatureAgglomeration
import pickle
import matplotlib.pyplot as plt 
from sklearn.decomposition import IncrementalPCA
#from scipy.cluster.vq import kmeans,vq,whiten
from sklearn import metrics

np.random.seed(42)
with open('extracted_features.pickle','rb') as f:
	data=pickle.load(f)

with open('extracted_features2.pickle','rb') as f:
	data2=pickle.load(f)

labels1=np.array([0 for i in range(70)])
labels2=np.array([1 for i in range(60)])

labels=np.concatenate((labels1,labels2),axis=0)
#print(labels)

print(data.shape)
print(data2.shape)
x=np.concatenate((data,data2))
print(x.shape)
kmeans = KMeans(n_clusters=2)
clx=kmeans.fit_predict(x)
print(clx)
print(metrics.accuracy_score(labels,clx))