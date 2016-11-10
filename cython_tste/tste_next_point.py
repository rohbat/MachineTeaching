from 
import cy_tste as cy_tste
import numpy as np


def probability(X, 
	K, 
	N, 
	no_dims, 
	alpha): 

	triplets_A = triplets[:,0]
	triplets_B = triplets[:,1]
	triplets_C = triplets[:,2]

	K = np.zeros(N, N)
	Q  = np.zeros(N, N)
	sum_X = np.zeros(N)

    for i in range([a,b,c]):
        sum_X[i] = 0
        for k in xrange(no_dims):
            # Squared norm
            sum_X[i] += X[i,k]*X[i,k]
   
    for i in range([a, b, c]): 
        for j in range(N):
            K[i,j] = sum_X[i] + sum_X[j]
            for k in range(no_dims):
                K[i,j] += -2 * X[i,k]*X[j,k]
            Q[i,j] = (1 + K[i,j] / alpha) ** -1
            K[i,j] = (1 + K[i,j] / alpha) ** ((alpha+1)/-2)

    return K


def best_point(X, 
	K, 
	N
	no_dims, 
	alpha, 
	triplets, 
	class): 

	K = probability(X, K, N, no_dims, alpha)

    # Compute probability (or log-prob) for each triplet
    for t in range(len(triplets)):
        P = K[triplets_A[t], triplets_B[t]] / (
            K[triplets_A[t],triplets_B[t]] +
            K[triplets_A[t],triplets_C[t]])
    return P


def test_point(a, b, c, alpha):
	for i in [a, b, c]: 


