import numpy as np 
from sklearn.cluster import KMeans,FeatureAgglomeration
import pickle
import matplotlib.pyplot as plt 
from sklearn.decomposition import IncrementalPCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import cross_validation

np.set_printoptions(threshold=np.nan)

with open('features.pickle','rb') as f:
	data=pickle.load(f)

with open('labels.pickle','rb') as f:
	labels=pickle.load(f)

labels=np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20,stratify=labels)  

scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test) 
  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 