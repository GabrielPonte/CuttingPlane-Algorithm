# -*- coding: utf-8 -*-
"""
To run CuttingPlane use (list_g,time_FV) = CuttingPlane(A,r)

Input: a rank-r matrix A as numpy array and the rank r

Output:
- list_g: a list with information about each iteration of the algorithm, as described below:
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
    
"""

import scipy.io
import matplotlib.pyplot as plt
import time
import math
from numpy import linalg as LA
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.utils.extmath import randomized_svd

def getInfCP(H,R,m,n,num_const_add,elapsed_time,list_g,list_sing):
    """
    This functions returns general informations about H, and its singular values 
    to the Cutting Planes Method (CP)
    Informations about list_g:
    list_g[x][0] = Total number of P2 constraints applied
    list_g[x][1] = 1-norm of H
    list_g[x][2] = rank of H. Number of singular values greater than 10**(-5)
    list_g[x][3] = Number of elements of abs(H) that are greater than 10**(-5)
    list_g[x][4] = Number of elements of abs(R) that are greater than 10**(-5) (R = HAA+ - H)
    list_g[x][5] = Percentage of elements of H that are satisfying P2.
    list_g[x][6] = Frobenius-norm of R
    list_g[x][7] = Elapsed time to optimize the model   
    Informations about list_sing:
    list_sing[x][:] = Singular values of H that are greater than 10**(-5)
    If rank(H) = min(m,n) -> Will show rank(H)     singular values
    If rank(H) < min(m,n) -> Will show rank(H) + 1 singular values
    """
    list_sing_p = []
    rank_H = 0
    norma_1 = sum(sum(abs(H)))
    U, S, Vh = LA.svd(H)
    for value_sing in S:
        if abs(value_sing) > 10**(-5): 
            rank_H += 1
    if min(m,n) == rank_H:
        value_r = rank_H
    else:
        value_r = rank_H + 1
    for sing_i in range(value_r):
        list_sing_p.append(S[sing_i])
    list_sing.append(list_sing_p)
    num_nonzero = 0
    norm_fro = 0
    for j in range(n):
        for i in range(m):
            norm_fro += R[j,i]**2
            if abs(H[j,i]) > 10**(-5):
                num_nonzero += 1
    norm_fro = math.sqrt(norm_fro)
    cont_k = 0
    for i in range(n):
        for j in range(m):
            if abs(R[i,j]) > 10**(-6):
                cont_k +=1
    per_SatisfiedP2 = 1-(cont_k/(m*n))
    list_g.append([num_const_add,norma_1,rank_H,num_nonzero,cont_k,per_SatisfiedP2,norm_fro,elapsed_time])
    return (list_g,list_sing)

def generateH(m,n,Var,model):
    """
    This function generates the H matriz. 
    It only can be applied when it is received a Optimal Solution
    """
    H = np.zeros((n,m))
    for j in range(n):
        for i in range(m): 
            index = j*m  + i
            H[j,i] = Var[index].X
    return H

def generateR(m,n,P2_s,Var,model):
    """
    This function generates R = HAA+ - H by the P2 sparse function
    R is not a constraint matrix, but a matrix if its residues
    This matrix R is a 2d matrix in a dictionary form
    R is a n x m sparse matrix
    """
    R = {}    
    for i in range(n):
        for j in range(m):
            col_k = i*m + j
            row_k = i*m  #indice of H in Var
            R[(i,j)] = 0
            for k in range(m):
                R[(i,j)] += P2_s[row_k ,col_k] * Var[row_k].x
                row_k += 1
    return R

def sparseInitialBlock(A,m,n,r,U,S,V,flag_ineq):
    """
    This function satisfies the left hand side of: H + Z >= 0, Z - H >= 0, AHA = A, (AH)'- AH = 0"
    It creates a sparse matrix in a dictionary form
    M_sparse[0]:  H + Z (from  H + Z >= 0)
    M_sparse[1]: -H + Z (from -H + Z >= 0)
    M_sparse[2]: AHA    (from AHA == A)
    M_sparse[3]: (AH)'-AH (from (AH)' - AH == 0)
    """
    """ H + Z >= 0 """
    one_NL_sparse = {}
    for i in range(n):
        for j in range(m):
            ij = i*m+j
            # H(i,j) coefficient
            one_NL_sparse[(ij,ij)] = 1
            # Z(i,j) coefficient
            one_NL_sparse[(ij,n*m + ij)] = 1
    """ Z - H >= 0 """
    one_NU_sparse = {}
    for i in range(n):
        for j in range(m):
            ij = i*m+j
            # Z(i,j) coefficient
            one_NU_sparse[(ij,n*m + ij)] = 1
            # H(i,j) coefficient
            one_NU_sparse[(ij,ij)] = -1
    """ P1 Property """
    P1_sparse = {}
    # Setting P1_sparse       
    for i in range(r):
        for j in range(r):
            ij = i*r + j
            for l in range(n):
                for k in range(m): 
                    lk = l*m + k
                    coeff_hlk = (S[i]**2) * V[l,i] * U[k,j]  
                    if (ij,lk) in P1_sparse:
                        P1_sparse[(ij,lk)] = P1_sparse[ij,lk] + coeff_hlk       
                    else:
                        P1_sparse[(ij,lk)] = coeff_hlk
    """ P3 Property """
    P3_sparse = {}
    count = -1 # Row index
    for i in range(m):
        for j in range(m):
            if i!=j and j>i:
                count += 1
                for k in range(n):  
                    #Variable h_{kj}: AH
                    kj = k*m + j
                    if (count,kj) in P3_sparse:
                        P3_sparse[(count,kj)] = P3_sparse[count,kj] - A[i,k]
                    else:
                        P3_sparse[(count,kj)] = -A[i,k]
                    #Variable h_{ki}: (AH)'
                    ki = k*m + i                    
                    if (count,ki) in P3_sparse:                        
                        P3_sparse[(count,ki)] = P3_sparse[count,ki] + A[j,k]                        
                    else:                        
                        P3_sparse[(count,ki)] = A[j,k]                
                if flag_ineq != 0:                
                    count+=1                            
                    for k in range(n):                        
                        #Variable h_{kj}: AH                        
                        kj = k*m + j                        
                        if (count,kj) in P3_sparse:                            
                            P3_sparse[(count,kj)] = P3_sparse[count,kj] + A[i,k]                        
                        else:                            
                            P3_sparse[(count,kj)] = A[i,k]                        
                        #Variable h_{ki}: (AH)'                        
                        ki = k*m + i                      
                        if (count,ki) in P3_sparse:                            
                            P3_sparse[(count,ki)] = P3_sparse[count,ki] - A[j,k]                            
                        else:                            
                            P3_sparse[(count,ki)] = - A[j,k]                    
    M_sparse = [one_NL_sparse,one_NU_sparse,P1_sparse,P3_sparse]
    return M_sparse


def sparseP2(A,m,n,r,U,S,V):
    
    """
    This function satisfies the left hand side of: HAA+ - H == 0"
    It creates a sparse P2 matrix in a dictionary form
    P2_sparse: HAA+ - H (from HAA+ - H == 0)    
    A  = U S  V'
    A* = V S* U'    
    AA* = U(m,r) @ U'(r,m)
    """    
    P2L_sparse = {}    
    Mat = {}    
    "Matrix Multiplication:  AA* = U(m,r) @ U'(r,m) "
    for i in range(m):
        for j in range(m):            
            Mat[(i,j)] = 0            
            for k in range(r):                
                Mat[(i,j)] = Mat[i,j] + U[i,k] * U[j,k]   
    count = -1    
    for i in range(n):
        for j in range(m):            
            ij = i*m + j            
            count += 1            
            P2L_sparse[(count,ij)] = -1           
            for k in range(m):                
                # Variable h_{ik} : HMat                
                ik = i*m + k                
                if (count,ik) in P2L_sparse:
                    P2L_sparse[(count,ik)] = P2L_sparse[count,ik] + Mat[k,j]                    
                else:
                    P2L_sparse[(count,ik)] = Mat[k,j]
    return P2L_sparse

def rhsInitialBlock(A,m,n,r,S,flag_ineq):
    """
    This function satisfies the right hand side of: H + Z >= 0, Z - H >= 0, AHA = A, (AH)'- AH = 0"
    It creates a rhs in a list form for the sparse Initial Block
    z_rhs[0 : nm-1]:                             0 (from  H + Z >= 0)
    z_rhs[nm : 2nm - 1]:                         0 (from -H + Z >= 0)
    z_rhs[2nm: 2nm+ r**2 - 1]:                   A (from AHA == A)
    z_rhs[2nm+ r**2: 2nm+ r**2+ (m*m-m)/2 - 1]:  0 (from (AH)' - AH == 0)
    """
    z_rhs = []    
    """ 1-norm objective reformulation: H+Z >=0 and Z - H >= 0 """    
    for i in range(2*n*m):        
        z_rhs.append(0)    
    """ P1: AHA = A, rhs = A[i,j] for i in range(m), for j in range(n) """    
    # SVD Formulation r*r entries    
    for i in range(r):
        for j in range(r):            
            if i == j:
                z_rhs.append(S[i])                
            else:
                z_rhs.append(0)
    """ P3: (AH) - (AH)' == 0, rhs = 0 """    
    if flag_ineq == 0:    
        for i in range(int((m**2 - m)/2)):        
            z_rhs.append(0)   
    else:
        for i in range(int((m**2 - m)/2)):            
            z_rhs.append(-10**-6)            
        for i in range(int((m**2 - m)/2)):            
            z_rhs.append(10**-6)   
    return z_rhs

def rhsAddP2(m,n,z_rhs):    
    """
    This function satisfies the right hand side of: HAA+ - H == 0"
    It adds a rhs in a list form for the sparse P2
    z_rhs[x : x + nm-1]: 0 (from  HAA+ - H == 0)
    """
    """ P2L: HAA+ - H == 0, rhs = 0"""    
    for i in range(n*m):        
        z_rhs.append(0)        
    return z_rhs 

def senseInitialBlock(A,m,n,r,flag_ineq):    
    """
    This function satisfies the sense form of: H + Z >= 0, Z - H >= 0, AHA = A, (AH)'- AH = 0"
    It creates a sense form in a list form for the sparse Initial Block    
    sense_form[0 : nm-1]:                              '>=' (from  H + Z >= 0)
    sense_form[nm : 2nm - 1]:                          '>=' (from -H + Z >= 0)
    sense_form[2nm : 2nm+ r**2 - 1]:                   '==' (from AHA == A)
    sense_form[2nm+ r**2 : 2nm+ r**2+ (m*m-m)/2 - 1]:  '==' (from (AH)' - AH == 0)
    """    
    sense_form = []    
    """ 1-norm constraint formulation: -Z <= H <= Z """    
    for k in range(n*m):        
        # H + Z >= 0        
        sense_form.append(">=")        
        # Z - H >= 0        
        sense_form.append(">=")   
    """ P1: AHA - A == 0 """    
    nr_P1svd = r*r    
    for k in range(nr_P1svd):        
        sense_form.append("==")       
    """ P3: AH - (AH)' == 0 """    
    nr_P3 = int((m**2 - m)/2)   
    if flag_ineq == 0:        
        for k in range(nr_P3):            
             sense_form.append("==")
    else:    
        for k in range(nr_P3):            
            sense_form.append(">=")            
        for k in range(nr_P3):            
            sense_form.append("<=")       
    return sense_form

def senseAddP2(m,n,sense_form):
    """
    This function satisfies the sense form of: HAA+ - H == 0
    It adds a sense form in a list form for the sparse P2
    sense_form[x , x : nm-1]: '==' (from  HAA+ - H == 0)
    """
    nr_P2L = m*n    
    for k in range(nr_P2L):        
        sense_form.append("==")        
    return sense_form

def applyConstraints(num_prop,M_s,sense_form,z_rhs,Var,model):    
    """
    This function apply the constraints into the model for some Var.
    For do it, you have to say how many properties (num_prop) it's satisfied in M_s
    """
    old_row = 0
    constr = 0   
    count = 0    
    for k in range(num_prop):        
        for indice in M_s[k]:            
            row_Ms = indice[0]
            col_Ms = indice[1]            
            if row_Ms == old_row:                
                constr += M_s[k][row_Ms,col_Ms] * Var[col_Ms] # Multiplicacao de matrizes, linha por coluna       
            else:                
                model.addLConstr(constr,sense_form[count],z_rhs[count])
                old_row = row_Ms                
                count += 1                
                constr = M_s[k][row_Ms,col_Ms] * Var[col_Ms]              
    model.addLConstr(constr,sense_form[count],z_rhs[count])        
    return model

def applyConstraintsP2_CP(m,n,list_res,P2_s,Var,model):
    """
    This function apply some constraints into the model for some Var.
    It receives res: Indices to apply these constraints, it is in a list form: [row,col]    
    The idea of this function is to apply the P2 constraint in H[row,col]    
    To do it with the P2 Sparse, it is good to know some things:        
        H * P2_s = H * [AA+ - 1] = H * P2_s        
        You will choose a row of H and multiply by a col of P2_s        
        H is in a column form, so we have to use row_k = im, i.e, this is the first
        element of the row you want to use for the multiplication        
        The row of P2_s starts from the same way as the row of H        
        The col of P2_s is the same, that is col_k = im + j (because of the column form)        
    It will be a multiplication of m elements, because H is (n,m) and P2_s (m,m)
    Here we are using the Sense Form of P2 : '=='
    And we are using the RHS of P2: 0
    """
    for indice in list_res:        
        constr = 0        
        col_k = indice[0]*m + indice[1]
        row_k = indice[0]*m  #indice of H in Var        
        for i in range(m):            
            constr += P2_s[row_k,col_k] * Var[row_k]            
            row_k += 1
        model.addLConstr(constr,"==",0)        
    return model

class createGraphic():
    
    def __init__(self):
        self = plt.plot([],[])

    def update_line(self, new_dataX,new_dataY):
        self.set_xdata(np.append(hl.get_xdata(), new_dataX))
        self.set_ydata(np.append(hl.get_ydata(), new_dataY))
        plt.draw()

def CuttingPlane(A,r,factor = 0.01,flag_ineq = 1):    
    """
    This function applies P1 and P3 properties.
    After that, it iteratively satisfies P2, searching for the first k indices
    greater than 10^{-6} to apply these constraints.    
    Stopping Criterion:        
        - Frobenius norm
        - Total number of contraints added = mn    
    """
    x = [0]
    (m,n) = np.shape(A)    
    U, S, VT = randomized_svd(A, n_components=r,n_iter=20,random_state=None)
    V = VT.T    
    start_time = time.time()    
    model = gp.Model()    
    "First nm variables are from H, then the next nm variable are from Z"    
    Var_1 = model.addVars(2*n*m, lb= -GRB.INFINITY, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS,name = "Var")    
    " Z lower bound = 0 "    
    for i in range(n*m,2*n*m):
       Var_1[i].lb = 0
    "LHS Sparse Matrix, 4 properties"       
    M_s = sparseInitialBlock(A,m,n,r,U,S,V,flag_ineq)   
    "Sense form, 4 properties"    
    sense_form = senseInitialBlock(A,m,n,r,flag_ineq)
    "RHS List, 4 properties"    
    z_rhs = rhsInitialBlock(A,m,n,r,S,flag_ineq)   
    "Set objective function: min sum(sum(Z))"
    obj_1n = 0    
    for i in range(n*m,2*n*m):        
        obj_1n += Var_1[i]    
    model.setObjective(obj_1n, GRB.MINIMIZE)    
    "Apply the constraints in the model"    
    num_prop = 4 # There are five properties: H+Z,-H+Z,P1 and P3    
    applyConstraints(num_prop,M_s,sense_form,z_rhs,Var_1,model)    
    start_time2 = time.time()    
    "Optimize model"    
    model.optimize()    
    "Get elapsed time to run the model"    
    elapsed_time = time.time() - start_time2    
    "Generate H"    
    H = generateH(m,n,Var_1,model)    
    "Generate P2 sparse"    
    P2_s = sparseP2(A,m,n,r,U,S,V)    
    "Number of P2 restrictions applied"
    num_const_add = 0    
    "Generate Iterative Information List"    
    list_g = []     # All the general informations of the iterations
    list_sing = []  # The singular values of the iterations    
    "Discover the residue: R = HAA+ - H"    
    R = generateR(m,n,P2_s,Var_1,model)    
    "Update Informations"    
    getInfCP(H,R,m,n,num_const_add,elapsed_time,list_g,list_sing)    
    "Number of constraints to add in the iteration"    
    num_add_max = factor*m*n        
    "Start a flag to start while cicle"    
    flag = 1    
    "Start while cicle"    
    while flag > 0:        
        """
        Find the first k indices to add the restrictions
        If the element R[i,j] > 10^{-6}, add as a restriction
        If the number it already added 0.1*mn constraints. Break the for loop        
        """        
        res = {}        
        num_add = 0      
        for i in range(m):
            for j in range(n):                    
                if num_add < num_add_max:
                    if abs(R[j,i]) > 10**(-6):                
                        first_row = j
                        first_col = i
                        res[(first_row,first_col)] = [first_row,first_col]                        
                        num_add += 1
                else:
                    break                
            if num_add == num_add_max:                
                break
        "Convert dict type to list type"        
        list_res = list(res.values())
        "Apply the First Contraints into the model"        
        applyConstraintsP2_CP(m,n,list_res,P2_s,Var_1,model)        
        "Constraints added in the total number of constraints"        
        num_const_add += num_add        
        start_time2 = time.time()        
        "Optimize the model"        
        model.optimize()        
        "Get elapsed time to run the model"        
        elapsed_time = time.time() - start_time2        
        "Generate the new H"        
        H = generateH(m,n,Var_1,model)        
        "Discover the residue: R = HAA+ - H"        
        R = generateR(m,n,P2_s,Var_1,model)        
        "Update Informations"        
        getInfCP(H,R,m,n,num_const_add,elapsed_time,list_g,list_sing)        
        "Find percentage of P2 satisfied"        
        per_P2 = list_g[-1][5]          
        "Stopping Criterion"        
        if per_P2 == 1  or num_const_add == m*n:            
            flag = 0 
            x.append(x[-1]+1)
        else:            
            flag = 1  
            x.append(x[-1]+1)
    
    array_listG = np.array(list_g)
    
    plt.plot(x,array_listG[:,1])
    plt.xlabel('iteration')
    plt.ylabel('1-norm')
    plt.show()
    
    plt.plot(x,array_listG[:,2])
    plt.xlabel('iteration')
    plt.ylabel('rank')
    plt.show()
    
    plt.plot(x,array_listG[:,6])
    plt.xlabel('iteration')
    plt.ylabel('frobenius-norm')
    plt.show()
    
    plt.plot(x,array_listG[:,5])
    plt.xlabel('iteration')
    plt.ylabel('P2 satisfied')
    plt.show()
    
    time_FV = time.time() - start_time        
    return (list_g,time_FV)