#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed, parallel_backend
from itertools import combinations, product
from collections import deque
from scipy.stats import multivariate_normal
import math
import random
import time

class btsvm_node:
    def __init__(self, clf, lhalf, rhalf):
        self.clf = clf
        self.left = None
        self.right = None
        self.lhalf = lhalf
        self.rhalf = rhalf
    def insert(self, clf, lhalf, rhalf, dir):
        if dir == 1:
            self.left = btsvm_node(clf, lhalf, rhalf)
        elif dir == 2:
            self.right = btsvm_node(clf, lhalf, rhalf)

class bhc_node:
    def __init__(self, w, means, covs, priors, lhalf, rhalf):
        self.w = w
        self.means = means
        self.covs = covs
        self.priors = priors
        self.left = None
        self.right = None
        self.lhalf = lhalf
        self.rhalf = rhalf

    def insert(self, w, means, covs, priors, lhalf, rhalf, dir):
        if dir == 1:
            self.left = bhc_node(w, means, covs, priors, lhalf, rhalf)
        if dir == 2:
            self.right = bhc_node(w, means, covs, priors, lhalf, rhalf)

    def predict(self, data):
        likelihood_alpha = lambda x: multivariate_normal.pdf(np.matmul(self.w.T, x), mean = np.matmul(self.w.T, self.means[0]), cov = np.matmul(np.matmul(self.w.T, self.covs[0]), self.w))
        likelihood_beta = lambda x: multivariate_normal.pdf(np.matmul(self.w.T, x), mean = np.matmul(self.w.T, self.means[1]), cov = np.matmul(np.matmul(self.w.T, self.covs[1]), self.w))
        return np.array([ 2 - (likelihood_alpha(data[n,:])*self.priors[0] >= likelihood_beta(data[n,:])*self.priors[1]) for n in range(data.shape[0]) ])
        

def kmeansHalves(labelset, data, labels):
    P = list( combinations( labelset, int( len(labelset)/2 ) ) )
    L = [ np.empty( (len(labels),), dtype=int) for i in range(len(P)) ]

    with parallel_backend('threading', n_jobs=10):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

    for i, p in enumerate(P):
        idx = np.in1d(labels,p)
        L[i][idx] = 1
        L[i][~idx] = 2
    
    scores = [ sum(l == kmeans.labels_) for l in L ]
    i = np.argmin(scores)
    lhalf = np.asarray(P[i])
    rhalf = np.setdiff1d(labelset,lhalf)

    return L[i], lhalf, rhalf

def randomHalves(labelset, labels):
    L = np.empty( (len(labels),), dtype = int )
    h = random.sample(list(labelset), int( len(labelset)/2 ))
    idx = np.in1d(labels, h)
    L[idx] = 1
    L[~idx] = 2
    lhalf = np.asarray(h)
    rhalf = np.setdiff1d(labelset, lhalf)

    return L, lhalf, rhalf

def distanceHalves(labelset, data, labels):
    means = [ np.mean(data[labels == label,:], axis = 0) for label in labelset ]
    dists = np.empty( (len(labelset),len(labelset)) )

    for i in range(len(means)):
        for j in range(len(means)):
            dists[i,j] = np.linalg.norm(means[i] - means[j])
    
    idx = np.unravel_index(np.argmax(dists), dists.shape)

    lhalf = np.array( [ labelset[idx[0]] ] )
    rhalf = np.array( [ labelset[idx[1]] ] )
    lmean = means[idx[0]]
    rmean = means[idx[1]]

    lset = labelset[labelset != lhalf]
    lset = lset[lset != rhalf]
    for label in lset:
        idx = np.where(labelset == label)
        idx = idx[0][0]
        if np.linalg.norm(means[idx] - lmean) <= np.linalg.norm(means[idx] - rmean):
            lhalf = np.append(lhalf, label)
        else:
            rhalf = np.append(rhalf, label)
    
    L = np.empty( (len(labels),), dtype = int )
    idx = np.in1d(labels, lhalf)
    L[idx] = 1
    L[~idx] = 2
    return L, lhalf, rhalf

def mahalanobisHalves(labelset, data, labels):
    means = [ np.mean(data[labels == label,:], axis = 0) for label in labelset ]
    covs = [ np.cov(data[labels == label,:], rowvar=False) for label in labelset ]
    dists = np.empty( (len(labelset),len(labelset)) )

    regularize = lambda A: A + ( 0.01 * max(np.linalg.eig(A)[0]) ) * np.identity(A.shape[0])
    dist = lambda i,j: math.sqrt( np.matmul( np.matmul( np.reshape( means[j] - means[i], (means[i].shape[0],1) ).T,  np.linalg.inv( regularize(covs[i]) ) ), np.reshape( means[j] - means[i], (means[i].shape[0],1) )) )

    for i in range(len(means)):
        for j in range(len(means)):
            dists[i,j] = dist(i,j)
    
    idx = np.unravel_index(np.argmax(dists), dists.shape)

    lhalf = np.array( [ labelset[idx[0]] ] )
    rhalf = np.array( [ labelset[idx[1]] ] )

    lset = labelset[labelset != lhalf]
    lset = lset[lset != rhalf]
    for label in lset:
        idy = np.where(labelset == label)
        idy = idy[0][0]
        if dist(idx[0],idy) <= dist(idx[1],idy):
            lhalf = np.append(lhalf, label)
        else:
            rhalf = np.append(rhalf, label)
    
    L = np.empty( (len(labels),), dtype = int )
    idx = np.in1d(labels, lhalf)
    L[idx] = 1
    L[~idx] = 2
    return L, lhalf, rhalf

## BHC algorithm from Kumar et al "Hierarchical Fusion of Multiple Classifiers for Hyperspectral Data Analysis"
def ldaHalves(labelset, data, labels):
    ## Sample measures
    priors = np.array([ sum(labels == label) for label in labelset ]) / len(labels) # Class priors derived from sample counts
    sample_means = [ np.mean(data[labels == label,:], axis = 0) for label in labelset ]
    sample_covs = [ np.cov(data[labels == label,:], rowvar=False) for label in labelset ]

    ## Bookkeeping
    w = 0
    mean_alpha = 0
    mean_beta = 0
    cov_alpha = 0
    cov_beta = 0
    regularize = lambda A: A + ( 0.01 * max(np.linalg.eig(A)[0]) ) * np.identity(A.shape[0])

    ## Initialization (Step 1)
    posteriors_alpha = np.append([1], 0.5*np.ones(len(labelset)-1)) # Meta-class 1 (alpha) posteriors
    posteriors_beta = 1 - posteriors_alpha # Meta-class 1 (alpha) posteriors
    temp = 1 # T (Temperature)
    cooler = 0.8 # theta_T (cooling factor)
    Tau = 0 # Fisher discriminant
    count = 0 # 
    threshold = 0.05 # theta_H (entropy threshold)
    entropy = 1 # H (entropy)

    ## Outer loop (established in Step 8)
    while entropy > threshold:
        ## Inner loop (established in Step 5)
        while True:
            ## Meta-class mean and covariance calculations (Step 2)
            # Alpha
            conditionals = [ (priors[l]*posteriors_alpha[l]/np.dot(posteriors_alpha,priors)) for l in range(len(labelset)) ]
            mean_alpha = sum([ conditionals[l]*sample_means[l] for l in range(len(labelset)) ])
            temp_xcovs = [ np.reshape(sample_means[l]-mean_alpha, (len(mean_alpha),1)) for l in range(len(labelset)) ]
            cov_alpha = sum([ conditionals[l]*(sample_covs[l] +  np.matmul(temp_xcovs[l], temp_xcovs[l].T)) for l in range(len(labelset)) ])

            # Beta
            conditionals = [ (priors[l]*posteriors_beta[l]/np.dot(posteriors_beta,priors)) for l in range(len(labelset)) ]
            mean_beta = sum([ conditionals[l]*sample_means[l] for l in range(len(labelset)) ])
            temp_xcovs = [ np.reshape(sample_means[l]-mean_beta, (len(mean_beta),1)) for l in range(len(labelset)) ]
            cov_beta = sum([ conditionals[l]*(sample_covs[l] +  np.matmul(temp_xcovs[l], temp_xcovs[l].T)) for l in range(len(labelset)) ])

            ## LDA (Fisher ? ) (Step 3)
            W = np.dot(posteriors_alpha,priors) * cov_alpha + np.dot(posteriors_alpha,priors) * cov_beta
            #W = W + 0.1 * np.identity(W.shape[0]) # TODO: Play around with this
            W = regularize(W)
            B = np.matmul( np.reshape( (mean_alpha - mean_beta), (len(mean_alpha), 1) ), np.reshape( (mean_alpha - mean_beta), (1, len(mean_alpha)) ) )
            w = np.matmul( np.linalg.inv(W), np.reshape( (mean_alpha - mean_beta), (len(mean_alpha), 1) ) ) # make sure means have same dimension, it is a bug otherwise
            prev_Tau = Tau
            Tau = np.matmul(np.matmul(w.T, B), w) / np.matmul(np.matmul(w.T, W), w)

            ## Calculate likelihoods (Step 4)
            #likelihood = lambda mean, cov, X, label: sum([ multivariate_normal.pdf(np.matmul(w.T, X[n,:]), mean = np.matmul(w.T, mean), cov = np.matmul(np.matmul(w.T,cov),w)) for n in range(X.shape[0]) ]) / sum(labels == label)
            # Alpha
            #likelihoods_alpha = Parallel(n_jobs = 10) ( delayed(likelihood)(mean_alpha, cov_alpha, data[labels == label, :], label) for label in labelset)
            likelihoods_alpha = np.zeros(len(labelset))
            for (l,label) in enumerate(labelset):
                X = data[labels == label,:]
                likelihoods_alpha[l] = sum([ multivariate_normal.pdf(np.matmul(w.T, X[n,:]), mean = np.matmul(w.T, mean_alpha), cov = np.matmul(np.matmul(w.T,cov_alpha),w)) for n in range(X.shape[0]) ]) / sum(labels == label)                
            
            # Beta
            #likelihoods_beta = Parallel(n_jobs = 10) ( delayed(likelihood)(mean_beta, cov_beta, data[labels == label, :], label) for label in labelset)
            likelihoods_beta = np.zeros(len(labelset))
            for (l,label) in enumerate(labelset):
                X = data[labels == label,:]
                likelihoods_beta[l] = sum([ multivariate_normal.pdf(np.matmul(w.T, X[n,:]), mean = np.matmul(w.T, mean_beta), cov = np.matmul(np.matmul(w.T,cov_beta),w)) for n in range(X.shape[0]) ]) / sum(labels == label)

            ## Update meta-class posteriors (Step 5)
            posteriors_alpha = np.asarray([ np.exp(likelihoods_alpha[l]/temp) / ( np.exp(likelihoods_alpha[l]/temp) + np.exp(likelihoods_beta[l]/temp) ) for l in range(len(labelset)) ])
            posteriors_beta = 1 - posteriors_alpha

            if Tau < 1.05 * prev_Tau:
                break

        ## Entropy calculation (Step 7)
        entropy = (-1/len(labelset)) * sum( [ posteriors_alpha[l]*np.log2(posteriors_alpha[l]) + posteriors_beta[l]*np.log2(posteriors_beta[l]) for l in range(len(labelset))] )

        count = count + 1
        temp = temp*cooler
    
    L = np.empty( (len(labels),), dtype = int )
    lhalf = labelset[posteriors_alpha >= 0.5]
    rhalf = labelset[posteriors_alpha < 0.5]
    prior_alpha = sum(priors[posteriors_alpha >= 0.5])
    idx = np.in1d(labels, lhalf)
    L[idx] = 1
    L[~idx] = 2

    return L, lhalf, rhalf, w, (mean_alpha, mean_beta), (cov_alpha, cov_beta), (prior_alpha, 1 - prior_alpha)  

def getHalves(labelset, data, labels, half_method = "kmeans"):
    if half_method == "kmeans":    
        return kmeansHalves(labelset, data, labels) # Get best halves for root SVM node
    elif half_method == "random":
        return randomHalves(labelset, labels)
    elif half_method == "distance":
        return distanceHalves(labelset, data, labels)
    elif half_method == "mahalanobis":
        return mahalanobisHalves(labelset, data, labels)
    elif half_method == "lda":
        return ldaHalves(labelset, data, labels)[0:3]

def cvtrainBTSVM(labelset, data, labels, half_method = "kmeans"):
    parameters = {'gamma':2 ** np.linspace(-15, 3, num=10), 'C':2 ** np.linspace(-5, 15, num=10)}    

    L, lhalf, rhalf = getHalves(labelset, data, labels, half_method)

    with parallel_backend('threading', n_jobs=10):
        clf = make_pipeline(StandardScaler(), GridSearchCV(svm.SVC(), parameters, cv=10)) # Hyperparameter tuning with 10-fold CV
        clf.fit(data, L) # Train root node SVM model
    hp = clf[1].best_params_ # Save best parameters (dict)

    root = btsvm_node(clf, lhalf, rhalf) # Create root node
    current = root
    node_stack = deque() # Create empty stack

    while True:
        if current is not None:
            node_stack.append(current)
            if len(current.lhalf) > 1:
                lidx = np.in1d(labels, current.lhalf)
                L, lhalf, rhalf = getHalves(current.lhalf, data[lidx,:], labels[lidx], half_method)
                clf = make_pipeline(StandardScaler(), svm.SVC(**hp))
                clf.fit(data[lidx,:], L)
                current.insert(clf, lhalf, rhalf, 1)
            if len(current.rhalf) > 1:
                ridx = np.in1d(labels, current.rhalf)
                L, lhalf, rhalf = getHalves(current.rhalf, data[ridx,:], labels[ridx], half_method)
                clf = make_pipeline(StandardScaler(), svm.SVC(**hp))
                clf.fit(data[ridx,:], L)
                current.insert(clf, lhalf, rhalf, 2)

            current = current.left

        elif len(node_stack):
            current = node_stack.pop()
            current = current.right
        
        else:
            break

    return root, hp

def trainBTSVM(labelset, data, labels, hp, half_method = "kmeans"):    
    L, lhalf, rhalf = getHalves(labelset, data, labels, half_method)

    clf = make_pipeline(StandardScaler(), svm.SVC(**hp))
    clf.fit(data, L) # Train root node SVM model

    root = btsvm_node(clf, lhalf, rhalf) # Create root node
    current = root
    node_stack = deque() # Create empty stack

    while True:
        if current is not None:
            node_stack.append(current)

            if len(current.lhalf) > 1:
                lidx = np.in1d(labels, current.lhalf)
                L, lhalf, rhalf = getHalves(current.lhalf, data[lidx,:], labels[lidx], half_method)
                clf = make_pipeline(StandardScaler(), svm.SVC(**hp))
                clf.fit(data[lidx,:], L)
                current.insert(clf, lhalf, rhalf, 1)
            if len(current.rhalf) > 1:
                ridx = np.in1d(labels, current.rhalf)
                L, lhalf, rhalf = getHalves(current.rhalf, data[ridx,:], labels[ridx], half_method)
                clf = make_pipeline(StandardScaler(), svm.SVC(**hp))
                clf.fit(data[ridx,:], L)
                current.insert(clf, lhalf, rhalf, 2)

            current = current.left

        elif len(node_stack):
            current = node_stack.pop()
            current = current.right
        
        else:
            break

    return root

def predictBTSVM(svm_root, data):
    pred = np.empty( (data.shape[0],), dtype=int )

    y = svm_root.clf.predict(data)

    if len(svm_root.lhalf) > 1:
        pred[y == 1] = predictBTSVM(svm_root.left, data[y == 1, :])
    else:
        pred[y == 1] = svm_root.lhalf[0]

    if len(svm_root.rhalf) > 1:
        pred[y == 2] = predictBTSVM(svm_root.right, data[y == 2, :])
    else:
        pred[y == 2] = svm_root.rhalf[0]

    return pred

def buildBHC(labelset, data, labels):
    L, lhalf, rhalf, w, means, covs, priors  = ldaHalves(labelset, data, labels) # We don't need to use L here, but we keep it because it's convenient for using ldaHalves in BT-SVM

    root = bhc_node(w, means, covs, priors, lhalf, rhalf)
    current = root
    node_stack = deque() # Create empty stack

    while True:
        if current is not None:
            node_stack.append(current)

            if len(current.lhalf) > 1:
                lidx = np.in1d(labels, current.lhalf)
                L, lhalf, rhalf, w, means, covs, priors  = ldaHalves(current.lhalf, data[lidx,:], labels[lidx])
                current.insert(w, means, covs, priors, lhalf, rhalf, 1)
            if len(current.rhalf) > 1:
                ridx = np.in1d(labels, current.rhalf)
                L, lhalf, rhalf, w, means, covs, priors  = ldaHalves(current.rhalf, data[ridx,:], labels[ridx])
                current.insert(w, means, covs, priors, lhalf, rhalf, 2)

            current = current.left

        elif len(node_stack):
            current = node_stack.pop()
            current = current.right
        
        else:
            break

    return root

def predictBHC(bhc_root, data):
    pred = np.empty( (data.shape[0],), dtype=int )

    y = bhc_root.predict(data)

    if len(bhc_root.lhalf) > 1:
        pred[y == 1] = predictBHC(bhc_root.left, data[y == 1, :])
    else:
        pred[y == 1] = bhc_root.lhalf[0]

    if len(bhc_root.rhalf) > 1:
        pred[y == 2] = predictBHC(bhc_root.right, data[y == 2, :])
    else:
        pred[y == 2] = bhc_root.rhalf[0]

    return pred     

def cvtrainOvR(labelset, data, labels):
    parameters = {'gamma':2 ** np.linspace(-15, 3, num=10), 'C':2 ** np.linspace(-5, 15, num=10)}
    L = np.empty( (len(labels),), dtype=int )
    clfs = list()
    hps = list()

    for i, label in enumerate(labelset):
        idx = np.in1d(labels,label)
        L[idx] = 1
        L[~idx] = 2
        with parallel_backend('threading', n_jobs=10):
            clf = make_pipeline(StandardScaler(), GridSearchCV(svm.SVC(probability = True, random_state = 0), parameters, cv=10))
            clfs.append(clf.fit(data, L))
            hps.append(clf[1].best_params_)
            
    return clfs, hps

def trainOvR(labelset, data, labels, hps):
    L = np.empty( (len(labels),), dtype=int )
    clfs = list()

    for i, label in enumerate(labelset):
        idx = np.in1d(labels,label)
        L[idx] = 1
        L[~idx] = 2
        clf = make_pipeline(StandardScaler(), svm.SVC(probability = True, random_state = 0, **hps[i]))
        clfs.append(clf.fit(data, L))

    return clfs

def predictOvR(clfs, data):
    probs = np.concatenate( [ np.reshape(clf.predict_proba(data)[:,0], (data.shape[0],1)) for clf in clfs], axis =1 )    
    return np.argmax(probs, axis = 1)

def cvtrainOvO(labelset, data, labels):
    parameters = {'gamma':2 ** np.linspace(-15, 3, num=10), 'C':2 ** np.linspace(-5, 15, num=10)}
    P = list( combinations( labelset, 2 ) )
    L = np.empty( (len(labels),), dtype=int )
    clfs = list()
    hps = list()

    for i, p in enumerate(P):
        idx = np.in1d(labels,p[0])
        idy = np.in1d(labels,p[1])
        L[idx] = 1
        L[idy] = 2
        with parallel_backend('threading', n_jobs=10):
            clf = make_pipeline(StandardScaler(), GridSearchCV(svm.SVC(probability = True, random_state = 0), parameters, cv=10))
            clfs.append( clf.fit( np.concatenate((data[idx,:],data[idy,:]),axis = 0), np.concatenate((L[idx],L[idy])) ) )
            hps.append(clf[1].best_params_)
            
    return clfs, hps

def trainOvO(labelset, data, labels, hps):
    P = list( combinations( labelset, 2 ) )
    L = np.empty( (len(labels),), dtype=int )
    clfs = list()

    for i, p in enumerate(P):
        idx = np.in1d(labels,p[0])
        idy = np.in1d(labels,p[1])
        L[idx] = 1
        L[idy] = 2
        clf = make_pipeline(StandardScaler(), svm.SVC(probability = True, random_state = 0, **hps[i]))
        clfs.append( clf.fit( np.concatenate((data[idx,:],data[idy,:]),axis = 0), np.concatenate((L[idx],L[idy])) ) )

    return clfs

def predictOvO(clfs, data, labelset):
    probs = [ clf.predict_proba(data) for clf in clfs ]
    P = list( combinations( labelset, 2) )
    sum_probs = np.zeros( (data.shape[0],len(labelset)) )

    for i, p in enumerate(P):
        sum_probs[:,[p[0],p[1]]] += probs[i]
    
    return np.argmax(sum_probs, axis = 1)

def baseline(X_train, X_test, y_train, y_test, method = 'ovr', cv = True, hp = {'C': 1, 'gamma': 0.1}):
    parameters = {'gamma':2 ** np.linspace(-15, 3, num=10), 'C':2 ** np.linspace(-5, 15, num=10)}
    tic = time.time()

    if cv:
        clf = make_pipeline(StandardScaler(), GridSearchCV(svm.SVC(decision_function_shape = method), parameters, cv=10)) # Hyperparameter tuning with 10-fold CV
    else:
        clf = make_pipeline(StandardScaler(), svm.SVC(**hp))
    
    with parallel_backend('threading', n_jobs=10):
        clf.fit(X_train, y_train)
    
    toc = time.time()

    txt = method
    if cv:
        txt = txt + ' with CV trained in ' + repr(round(toc - tic,2)) + 's'
    else:
        txt = txt + ' without CV trained in ' + repr(round(1000*(toc - tic),2)) + 'ms'
    
    txt = txt 

    tic = time.time()
    pred = clf.predict(X_test)
    toc = time.time()
    txt = txt + ' and predicted in ' + repr(round(1000*(toc - tic), 2)) + 'ms'
    txt = txt + ' with ' + repr(round(100*sum(pred == y_test) / len(y_test), 2)) + '% accuracy'

    print(txt)

    if cv:
        return clf[1].best_params_