"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-
#Aka test 
import numpy as np
import matplotlib.pyplot as plt


from data import make_dataset
from plot import plot_boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def knn(n_neighbors : int , data ) :
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    X  = data[0]
    y  = data[1]
    model.fit(X,y)
    return model 




if __name__ == "__main__":
    accuracy = np.empty(5,dtype=float) 
    n_neighbors = 500
    size = 3000
    for i in range(5) :    
        data  = make_dataset(size)
        rand_indices = np.random.choice(size, size=1000, replace=False)
        X = data[0][rand_indices]
        y = data[1][rand_indices]
        LS = [X,y]
        
        model  = knn(n_neighbors , LS)
        
        remaining_indices = np.setdiff1d(np.arange(size), rand_indices)
        X = data[0][remaining_indices]
        y = data[1][remaining_indices]
        accuracy[i]= model.score(X,y)
    
    TS = [X,y]
    fname = str(n_neighbors) +"_neighbors"
    title = 'accuracy = '+str(round(sum(accuracy)/5 , 2)) + "  " + "std = " +str(round(max(accuracy)- min(accuracy), 2)) 
    plot_boundary(fname, model, X, y, mesh_step_size=0.1, title= title)
