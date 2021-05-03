# CuttingPlane-Algorithm
Algorithm to run CuttingPlane algorithm in generalized inverses

To run CuttingPlane use (list_g,time_FV) = CuttingPlane(A,r)

Input: a rank-r matrix A as numpy array and the rank r

Output:

- list_g: a list with information about each iteration of the algorithm, as described below:
- 
    list_g[x][0] = Total number of P2 constraints applied
    
    list_g[x][1] = 1-norm of H
    
    list_g[x][2] = rank of H (number of singular values greater than 10**(-5))
    
    list_g[x][3] = Number of elements of H with absolute values greater than 10**(-5)
    
    list_g[x][4] = Number of elements of R with absolute values greater than 10**(-5) (R = HAA+ - H)
    
    list_g[x][5] = Percentage of elements of H that satisfy P2.
    
    list_g[x][6] = Frobenius-norm of R
    
    list_g[x][7] = Elapsed time to optimize the problem in that specific iteration
    
- time_FV: total elapsed time to run the algorithm


Example:

    import scipy.io
    
    import matplotlib.pyplot as plt
    
    import time
    
    import math
    
    from numpy import linalg as LA
    
    import numpy as np
    
    import gurobipy as gp
    
    from gurobipy import GRB
    
    from sklearn.utils.extmath import randomized_svd
    
    M = scipy.io.loadmat('example.mat')
    
    A= M['A']
    
    r = 10
    
    (list_g,time_FV) = CuttingPlane(A,r) 
