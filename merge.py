import numpy as np 
import pickle
from collections import Counter


with open('bal_data.pickle','rb') as f:
	bal_data=pickle.load(f)

with open('bal_labels.pickle','rb') as f:
    bal_labels=pickle.load(f)

with open('khmer_data.pickle','rb') as f:
	khmer_data=pickle.load(f)

with open('khmer_labels.pickle','rb') as f:
    khmer_labels=pickle.load(f)

with open('sunda_data.pickle','rb') as f:
	sunda_data=pickle.load(f)

with open('sunda_labels.pickle','rb') as f:
    sunda_labels=pickle.load(f)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

final_labels=bal_labels+khmer_labels+sunda_labels
final_data=bal_data+khmer_data+sunda_data
bal_data=np.array(bal_data)
bal_labels=np.array(bal_labels)
khmer_data=np.array(khmer_data)
khmer_labels=np.array(khmer_labels)
sunda_data=np.array(sunda_data)
sunda_labels=np.array(sunda_labels)
final_data=np.array(final_data)
final_labels=np.array(final_labels)
# print(bal_data.shape)
# print(bal_labels.shape)
# print(khmer_data.shape)
# print(khmer_labels.shape)
# print(sunda_data.shape)
# print(sunda_labels.shape)
print(final_data.shape)

le.fit(final_labels)
print(len(le.classes_))
labels=le.transform(final_labels)
count_dict=dict(Counter(labels))
print(count_dict)
print(final_data.shape)

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

from numpy import unique
from numpy import random 

import numpy as np
def balanced_sample_maker(X, y, sample_size, random_seed=None):
    """ return a balanced data set by sampling all classes with sample_size 
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    return (X[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx)

final_data,labels,aaa=balanced_sample_maker(final_data,labels,2000)
# for lab in range(len(labels)):
# 	if(count_dict[labels[lab]]<2000):
# 		for i in range(2000-count_dict[labels[lab]]):
# 			np.insert()

count_dict=dict(Counter(labels))
print(count_dict)
print(final_data.shape)

with open('final_data.pickle','wb') as f:
	pickle.dump(final_data,f) 

with open('final_labels.pickle','wb') as f:
	pickle.dump(labels,f)


