import numpy as np
from cvxopt import matrix
from cvxopt import solvers

var_all_numpy= np.array([[4,2,4,6,8,6,12,10,8,6],[8,4,6,6,8,10,6,6,2,2]])
var_1_numpy = np.array([4,2,4,6,8,6,12,10,8,6])
var_2_numpy = np.array([8,4,6,6,8,10,6,6,2,2])
y_numpy = np.array([-1,-1,-1,-1,-1,-1,1,1,1,1])

H = matrix(np.array([[1,0,0],[0,1,0],[0,0,0]]),tc='d')
C = -matrix(np.array([1,1,1,1,1,1,1,1,1,1]),tc='d')
test_1 = np.multiply(var_1_numpy,y_numpy)
test_2 = np.multiply(var_2_numpy,y_numpy)
A = matrix(np.concatenate((test_1.reshape(-1,1),test_2.reshape(-1,1),y_numpy.reshape(-1,1)), axis = 1),tc='d')
zeros=matrix(np.zeros((3,1)),tc='d')

sol = solvers.qp(H,zeros,A,C)
print(sol['primal objective'])
print(sol['x'])






