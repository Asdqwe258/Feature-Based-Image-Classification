import cv2
import numpy as np
from PIL.Image import open
import os
import PIL
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt

#find SIFT features
sift = cv2.SIFT_create()
count = 0
traingroup = []
imagehash = {}
for filename in os.listdir('Project2_data/TrainingDataset'):
    count += 1
    path = (os.path.join('Project2_data/TrainingDataset', filename))
    I = cv2.imread(path, 0)
    [f, d] = sift.detectAndCompute(I, None)
    imagehash[path] = (f,d)
    try:
        allf = np.append(allf,f,axis= 0)
        alld = np.append(alld,d,axis= 0)
    except:
        allf = f
        alld = d
    #stores class information
    if filename.startswith("024_"):
        traingroup.append(24)
    elif filename.startswith("051_"):
        traingroup.append(51)
    elif filename.startswith("251_"):
        traingroup.append(251)

print('processed ' + str(count) + ' images')
#doing kmeans with all of the descriptors
kmeans = KMeans(n_clusters=100, max_iter=100, n_init=1).fit(alld)
print('did kmeans')
#Making histograms
def histogram(path):
    f, d = imagehash[path]
    x = np.zeros(100)
    #making the list that will be used as weights to normalize the counts
    norm = np.zeros(100)
    norm.fill(1/len(f))
    for k in d:
        #thank you error message for telling to reshape(1,-1)
        label = kmeans.predict(k.reshape(1,-1))
        x[label] += 1

    #normalize
    x = x/len(d)
    #plt.title(path)
    #plt.hist(x,bins=100,weights=norm)
    #plt.show()
    return x
alltrain = []
for filename in os.listdir('Project2_data/TrainingDataset'):
    path = (os.path.join('Project2_data/TrainingDataset', filename))
    h = histogram(path)
    alltrain.append(h)
print(np.shape(alltrain))
    #print(alltrain)
print('made the histograms')


print('moving on to test data')
count = 0
testgroup = []
for filename in os.listdir('Project2_data/TestingDataset'):
    count += 1
    path = (os.path.join('Project2_data/TestingDataset', filename))
    I = cv2.imread(path, 0)
    [f, d] = sift.detectAndCompute(I, None)
    imagehash[path] = (f, d)
    #saving the features and descriptors of each group in its own array
    #because the document recommended it
    try:
        testf = np.append(testf,f,axis= 0)
        testd = np.append(testd,d,axis= 0)
    except:
        testf = f
        testd = d
    if filename.startswith("024_"):
        testgroup.append(24)
    elif filename.startswith("051_"):
        testgroup.append(51)
    elif filename.startswith("251_"):
        testgroup.append(251)
print('processed ' + str(count) + ' images')
alltest = []
for filename in os.listdir('Project2_data/TestingDataset'):
    path = (os.path.join('Project2_data/TestingDataset', filename))
    h = (histogram(path))
    alltest.append(h)
print('made the histograms')


#Finding nearest neighbors
print('finished processing test data')
print('finding k nearest neighbor')
neigh = KNeighborsClassifier(n_neighbors=1).fit(np.array(alltrain), np.array(traingroup))
count = 0
confusion = np.zeros((3,3))
correct = 0
for x in alltest:
    predict = neigh.predict(x.reshape(1, -1))
    actual = testgroup[count]
    if predict == actual:
        correct += 1
    if actual == 24:
        actual = 0
    elif actual == 51:
        actual = 1
    elif actual == 251:
        actual = 2
    if predict == 24:
        predict = 0
    elif predict == 51:
        predict = 1
    elif predict == 251:
        predict = 2
    confusion[actual][predict] += 1
    count += 1
for x in range(len(confusion)):
    confusion[x] = confusion[x]/np.sum(confusion[x])
print(correct/count)
print(confusion)
print('-----------')


#Now for linear SVM
linear = make_pipeline(StandardScaler(),LinearSVC()).fit(np.array(alltrain), np.array(traingroup))
count = 0
confusion = np.zeros((3,3))
correct = 0
for x in alltest:
    predict = linear.predict(x.reshape(1, -1))
    actual = testgroup[count]
    if predict == actual:
        correct += 1
    if actual == 24:
        actual = 0
    elif actual == 51:
        actual = 1
    elif actual == 251:
        actual = 2
    if predict == 24:
        predict = 0
    elif predict == 51:
        predict = 1
    elif predict == 251:
        predict = 2
    confusion[actual][predict] += 1
    count += 1
for x in range(len(confusion)):
    confusion[x] = confusion[x]/np.sum(confusion[x])
print(correct/count)
print(confusion)
print('-----------')
#Kernel SVM
kern = make_pipeline(StandardScaler(), SVC()).fit(np.array(alltrain), np.array(traingroup))
count = 0
confusion = np.zeros((3,3))
correct = 0
for x in alltest:
    predict = kern.predict(x.reshape(1, -1))
    actual = testgroup[count]
    if predict == actual:
        correct += 1
    if actual == 24:
        actual = 0
    elif actual == 51:
        actual = 1
    elif actual == 251:
        actual = 2
    if predict == 24:
        predict = 0
    elif predict == 51:
        predict = 1
    elif predict == 251:
        predict = 2
    confusion[actual][predict] += 1
    count += 1
for x in range(len(confusion)):
    confusion[x] = confusion[x]/np.sum(confusion[x])
print(correct/count)
print(confusion)