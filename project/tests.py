#!/usr/bin/env python3

import time
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from functions import *

# Data is in the shape (# of samples, # of dimensions) or (n x d)

""" data = np.loadtxt("/home/volkan/Dropbox/5644/projectcode/data/beantraindata.csv",delimiter=",")
data = np.transpose(data)
labels = np.loadtxt("/home/volkan/Dropbox/5644/projectcode/data/beantrainlabels.csv",delimiter=",")
test = np.loadtxt("/home/volkan/Dropbox/5644/projectcode/data/beantestdata.csv",delimiter=",")
test = np.transpose(test)
target = np.loadtxt("/home/volkan/Dropbox/5644/projectcode/data/beantestlabels.csv",delimiter=",") """

def baseline_experimenter(labelset, X_train, X_test, y_train, y_test):
    tic = time.time()
    clfs, hps = cvtrainOvR(labelset, X_train, y_train)
    toc = time.time()
    txt = 'ovr with CV trained in ' + repr(round(toc - tic,2)) + 's'

    tic = time.time()
    pred = predictOvR(clfs, X_test)
    toc = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(toc - tic), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'
    print(txt)

    tic = time.time()
    clfs = trainOvR(labelset, X_train, y_train, hps)
    toc = time.time()
    txt = 'ovr without CV trained in ' + repr(round(1000*(toc - tic), 2)) + 'ms'

    tic = time.time()
    pred = predictOvR(clfs, X_test)
    toc = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(toc - tic), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'
    print(txt)

    tic = time.time()
    clfs, hps = cvtrainOvO(labelset, X_train, y_train)
    toc = time.time()
    txt = 'ovo with CV trained in ' + repr(round(toc - tic,2)) + 's'

    tic = time.time()
    pred = predictOvO(clfs, X_test, labelset)
    toc = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(toc - tic), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'
    print(txt)

    tic = time.time()
    clfs = trainOvO(labelset, X_train, y_train, hps)
    toc = time.time()
    txt = 'ovo without CV trained in ' + repr(round(1000*(toc - tic), 2)) + 'ms'

    tic = time.time()
    pred = predictOvO(clfs, X_test, labelset)
    toc = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(toc - tic), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'
    print(txt)

def btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, half_method):
    start = time.time()
    svm_root, hp = cvtrainBTSVM(labelset, X_train, y_train, half_method)
    stop = time.time()
    txt = half_method + ' with CV trained in ' + repr(round(stop - start,2)) + 's'

    start = time.time()
    pred = predictBTSVM(svm_root, X_test)
    stop = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(stop - start), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'
    print(txt)

    start = time.time()
    svm_root = trainBTSVM(labelset, X_train, y_train, hp, half_method)
    stop = time.time()
    txt = half_method + ' without CV trained in ' + repr(round(1000*(stop - start),2)) + 'ms'

    start = time.time()
    pred = predictBTSVM(svm_root, X_test)
    stop = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(stop - start), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'

    print(txt)

def experimenter(labelset, data, target, dataset_name, train_size = 0.33):
    print('-- ' + dataset_name + ' dataset --')
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size = train_size, random_state = 0, stratify = target)

    #hp = baseline(X_train, X_test, y_train, y_test, method = 'ovr')
    #baseline(X_train, X_test, y_train, y_test, method = 'ovr', cv = False, hp = hp)
    #hp = baseline(X_train, X_test, y_train, y_test, method = 'ovo')
    #baseline(X_train, X_test, y_train, y_test, method = 'ovo', cv = False, hp = hp)

    baseline_experimenter(labelset, X_train, X_test, y_train, y_test)

    btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, "kmeans")
    #btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, "distance")
    #btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, "mahalanobis")
    #btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, "lda")

    #start = time.time()
    #bhc_root = buildBHC(labelset, X_train, y_train)
    #stop = time.time()
    #txt = 'BHC built in ' + repr(round(stop - start,2)) + 's'

    #start = time.time()
    #pred = predictBHC(bhc_root, X_test)
    #stop = time.time()
    #txt = txt + ' and predicted in ' + repr(round(1000*(stop - start))) + 'ms'
    #txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'

def ext_baseline_experimenter(labelset, X_train, X_test, y_train, y_test, hp):
    hps = [hp] * len(labelset)
    tic = time.time()
    clfs = trainOvR(labelset, X_train, y_train, hps)
    toc = time.time()
    txt = 'ovr without CV trained in ' + repr(round(1000*(toc - tic), 2)) + 'ms'

    tic = time.time()
    pred = predictOvR(clfs, X_test)
    toc = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(toc - tic), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'
    print(txt)

    hps = [hp] * int(len(labelset) * (len(labelset)-1) / 2)
    tic = time.time()
    clfs = trainOvO(labelset, X_train, y_train, hps)
    toc = time.time()
    txt = 'ovo without CV trained in ' + repr(round(1000*(toc - tic), 2)) + 'ms'

    tic = time.time()
    pred = predictOvO(clfs, X_test, labelset)
    toc = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(toc - tic), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'
    print(txt)

def ext_btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, hp, half_method):
    start = time.time()
    svm_root = trainBTSVM(labelset, X_train, y_train, hp, half_method)
    stop = time.time()
    txt = half_method + ' without CV trained in ' + repr(round(1000*(stop - start),2)) + 'ms'

    start = time.time()
    pred = predictBTSVM(svm_root, X_test)
    stop = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(stop - start), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'

    print(txt)

def ext_experimenter(labelset, X_train, X_test, y_train, y_test, hp, dataset_name):
    print('-- ' + dataset_name + ' dataset --')

    #baseline(X_train, X_test, y_train, y_test, method = 'ovr', cv = False, hp = hp)
    #baseline(X_train, X_test, y_train, y_test, method = 'ovo', cv = False, hp = hp)

    #ext_baseline_experimenter(labelset, X_train, X_test, y_train, y_test, hp)

    ext_btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, hp, "kmeans")
    #ext_btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, hp, "distance")
    #ext_btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, hp, "mahalanobis")
    #ext_btsvm_experimenter(labelset, X_train, X_test, y_train, y_test, hp, "lda")

    #start = time.time()
    #bhc_root = buildBHC(labelset, X_train, y_train)
    #stop = time.time()
    #txt = 'BHC built in ' + repr(round(stop - start,2)) + 's'

    #start = time.time()
    #pred = predictBHC(bhc_root, X_test)
    #stop = time.time()
    #txt = txt + ' and predicted in ' + repr(round(1000*(stop - start))) + 'ms'
    #txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'

    #print(txt)
    
iris = load_iris()
experimenter(np.arange(0,len(iris.target_names),1), iris.data, iris.target, "Iris")

#digits = load_digits()
#experimenter(np.arange(0,len(digits.target_names),1), digits.data, digits.target, "Digits (small)")

digit_train = np.loadtxt("optdigits.tra", delimiter = ",")
digit_test = np.loadtxt("optdigits.tes", delimiter = ",")
ext_experimenter(np.arange(0,10,1), digit_train[:,0:-1], digit_test[:,0:-1], digit_train[:,-1], digit_test[:,-1], {'C': 32, 'gamma': 0.0078125}, "Digits (against BTS)")

data = np.loadtxt("letter-recognition.data", delimiter = ",",usecols = range(1,17))
target = np.loadtxt("letter-recognition.data", delimiter = ",", usecols = 0, converters = {0: lambda c: ord(c)-65})

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size = 16000, random_state = 0, stratify = target)

ext_experimenter(np.arange(0,26,1), X_train, X_test, y_train, y_test, {'C': 100, 'gamma': 0.447}, "Letter")