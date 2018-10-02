import numpy as np 
from sklearn.cluster import KMeans,FeatureAgglomeration
import pickle
import matplotlib.pyplot as plt 
from sklearn.decomposition import IncrementalPCA
from sklearn import metrics
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from itertools import cycle
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.nan)

with open('features_balanced_varsize.pickle','rb') as f:
	data=pickle.load(f)

with open('labels1.pickle','rb') as f:
	labels=pickle.load(f)



labels=np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20,random_state=42)  

scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

k=5
classifier = KNeighborsClassifier(n_neighbors=k,metric='minkowski',algorithm='auto')  
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test) 

plt.figure(figsize=(20, 15))
plt.imshow(np.array(confusion_matrix(y_test, y_pred)),cmap='gray')
plt.savefig('confusion_balanced_varsize_k'+str(k)+'.jpg')
plt.show()  
print(classification_report(y_test, y_pred)) 