from 
import cy_tste as cy_tste
import numpy as np



def probability(X, 
	N, 
	a, 
	b, 
	c, 
	no_dims, 
	alpha):

	K = np.zeros(N, N)
	sum_X = np.zeros(N)

    for i in [a,b,c]:
        sum_X[i] = 0
        for k in xrange(no_dims):
            sum_X[i] += X[i,k]*X[i,k]
   
    for i in [a, b, c]: 
        for j in range(N):
            K[i,j] = sum_X[i] + sum_X[j]
            for k in range(no_dims):
                K[i,j] += -2 * X[i,k]*X[j,k]
            K[i,j] = (1 + K[i,j] / alpha) ** ((alpha+1)/-2)

    return K


def prob_difference(X, 
	N, 
	no_dims, 
	alpha, 
	a, 
	b, 
	c, 
	correct, 
	classes = [], 
	no_classes = 3, 
	w_right=0.5, 
	w_wrong=0.5): 
	
	G = tste_grad(X, N, no_dims, (a, b, c), lamb, alpha)
	X = X - (float(eta) / no_classes * N) * G
	K = probability(X, N, a, b, c no_dims, alpha)

	correct_class = 0 # classes[b]
	not_in_class = []
	in_class = []
	for i in range(N): 
		if classes[i] != correct_class: 
			not_in_class.append(i)
		else: 
			in_class.append(i)

    # Compute probability (or log-prob) for each triplet
    diff1 = 0
    for i in [a, b, c]: 
    	for j in in_class: 
    		for k in not_in_class: 
		        P = K[i, j] / (K[i,j] + K[i,k])
		        diff1 += P - correct[i, j, k]
	
	
	G = tste_grad(X, N, no_dims, (a, c, b), lamb, alpha)
	X = X - (float(eta) / no_classes * N) * G
	K = probability(X, N, a, b, c no_dims, alpha)

	correct_class = 1 # classes[c]
	not_in_class = []
	in_class = []
	for i in range(N): 
		if classes[i] != correct_class: 
			not_in_class.append(i)
		else: 
			in_class.append(i)
	
	diff2 = 0
    for i in [a, c, b]: 
    	for j in in_class: 
    		for k in not_in_class: 
		        P = K[i, j] / (K[i,j] + K[i,k])

		        diff2 += P - correct[i, j, k]

    return w_right*diff1 + w_wrong*diff2



