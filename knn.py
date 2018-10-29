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
from sklearn.metrics import precision_recall_fscore_support as score

np.set_printoptions(threshold=np.nan)

with open('features.pickle','rb') as f:
	data=pickle.load(f)

with open('bal_labels.pickle','rb') as f:
	labels=pickle.load(f)


#data=data.reshape(data.shape[0],-1)

labels=np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20,random_state=42)  

scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

res=[]
ks=[]
for k in range(3,14,2):
	classifier = KNeighborsClassifier(n_neighbors=k,metric='euclidean',algorithm='auto')  
	classifier.fit(X_train, y_train) 
	y_pred = classifier.predict(X_test) 

	#plt.figure(figsize=(20, 15))
	#cm=np.array(confusion_matrix(y_test, y_pred))
	#plt.imshow(cm,interpolation='none',cmap='plasma')
	#plt.colorbar()

	#plt.savefig('confusion_augmented2_k'+str(k)+'.jpg') 
	#plt.show() 
	#print(classification_report(y_test, y_pred))
	#dy=classification_report(y_test, y_pred , output_dict=True)
	#print(dy)
	print(k)
	precision,recall,fscore,support=score(y_test,y_pred,average='macro')
	res.append(fscore)
	ks.append(k)
	print(precision,recall,fscore,support)
#print(np.array(confusion_matrix(y_test, y_pred)).tolist()) 

plt.plot(ks,res)
plt.xlabel('k')
plt.ylabel('f-score')
plt.savefig('knn_plot.png')
plt.show()