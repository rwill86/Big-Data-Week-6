#!/usr/bin/python

# Try to open imports
try:
    import sys
    import random
    import math
    import os
    import time
    import numpy as np
	import pandas as pd
    from matplotlib import pyplot as plt
	from sklearn.decompostion import PCA as sklearnPCA
    from scipy.spatial import distance
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
	from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis

# Error when importing
except ImportError:
    print('### ', ImportError, ' ###')
    # Exit program
    exit()
	
# DTW
def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix
	
# Read input
def read():
    X = np.random.random((100,10))
    y = np.random.randint(0,2, (100))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	parameters = {'knn':[2, 4, 8]}
    clf = GridSearchCV(KNeighborsClassifier(metric=DTW), parameters, cv=3, verbose=1)
    clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
	dtw(X, y)
	distance = dtw.distance(X, y)
	labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',4:'SITTING', 5:'STANDING', 6:'LAYING'}
    confusion_matrix = []

# Main
def main():
    # Read Input
    read()
    # Close Program
    exit()


# init
if __name__ == '__main__':
    # Begin
    main()


