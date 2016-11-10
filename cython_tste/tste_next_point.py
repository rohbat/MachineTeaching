import cy_tste as cy_tste
import numpy as np
import math

def tste_grad(X, N, no_dims, triplet, lamb, alpha, sum_X, K, Q, dC):
    """ Compute the cost function and gradient update of t-STE """
    triplet_A = triplet[0]
    triplet_B = triplet[1]
    triplet_C = triplet[2]
    P = 0
    C = 0
    A_to_B = 0
    A_to_C = 0
    # L2 Regularization cost
    C += lamb * np.sum(X**2)
    for i in xrange(N):
        sum_X[i] = 0
        for k in xrange(no_dims):
            # Squared norm
            sum_X[i] += X[i,k]*X[i,k]
    for i in triplet:
        for j in xrange(N):
            K[i,j] = sum_X[i] + sum_X[j]
            for k in xrange(no_dims):
                K[i,j] += -2 * X[i,k]*X[j,k]
            Q[i,j] = (1 + K[i,j] / alpha) ** -1
            K[i,j] = (1 + K[i,j] / alpha) ** ((alpha+1)/-2)
            # Now, K[i,j] = ((sqdist(i,j)/alpha + 1)) ** (-0.5*(alpha+1)),
            # which is exactly the numerator of p_{i,j} in the lower right of
            # t-STE paper page 3.
            # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab

    P = K[triplet_A, triplet_B] / (
        K[triplet_A,triplet_B] +
        K[triplet_A,triplet_C])
    # This is exactly p_{ijk}, which is the equation in the lower-right
    # of page 3 of the t-STE paper.
    C += -math.log(P)
    # This is exactly the cost.

    for i in xrange(no_dims):
        # For i = each dimension to use
        # Calculate the gradient of *this triplet* on its points.
        const = (alpha+1) / alpha
        A_to_B = ((1 - P) *
                  # K[triplets_A[t],triplets_B[t]] *
                  Q[triplet_A,triplet_B] *
                  (X[triplet_A, i] - X[triplet_B, i]))
        A_to_C = ((1 - P) *
                  # (K[triplets_A[t],triplets_C[t]]) *
                  Q[triplet_A,triplet_C] *
                  (X[triplet_A, i] - X[triplet_C, i]))

        dC[triplet_A, i]  = -const * (A_to_B - A_to_C)
        dC[triplet_B, i] = -const * (-A_to_B)
        dC[triplet_C, i] = -const * (A_to_C)

    for n in triplet:
        for i in xrange(no_dims):
            # The 2*lamb*npx is for regularization: derivative of L2 norm
            dC[n,i] = (dC[n,i]*-1) + 2*lamb*X[n,i]
    return C

def probability(X, 
    N, 
    a, 
    b, 
    c, 
    no_dims, 
    alpha,
    K):

    sum_X = np.zeros(N)

    for i in range(N):
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

diff1s = []
diff2s = []

def prob_difference(X, 
    N, 
    no_dims, 
    alpha,
    lamb, 
    triplet,
    classes = [], 
    no_classes = 3, 
    w_right=0.5, 
    w_wrong=0.5,
    eta = 2.0): 
    
    a, b, c = triplet
    sum_x = np.zeros(N)
    K = np.zeros((N, N))
    Q = np.zeros((N, N))
    G = np.zeros((N, no_dims))
    tste_grad(X, N, no_dims, (a, b, c), lamb, alpha, sum_x, K, Q, G)
    X1 = X - (float(eta) / no_classes * N) * G
    probability(X1, N, a, b, c, no_dims, alpha, K)

    correct_class = classes[a]
    not_in_class = []
    in_class = []
    for i in range(N): 
        if classes[i] != correct_class: 
            not_in_class.append(i)
        else: 
            in_class.append(i)

    # Compute probability (or log-prob) for each triplet
    diff1 = 0
    sm = 0
    for i in [a, b, c]: 
        for j in in_class: 
            for k in not_in_class: 
                P = K[i, j] / (K[i,j] + K[i,k])
                diff1 += abs(P - 1.0)
                sm += 1
    diff1s.append(diff1/sm)
    tste_grad(X, N, no_dims, (a, c, b), lamb, alpha, sum_x, K, Q, G)
    X2 = X - (float(eta) / no_classes * N) * G
    probability(X2, N, a, b, c, no_dims, alpha, K)

    correct_class = classes[c]
    not_in_class = []
    in_class = []
    for i in range(N): 
        if classes[i] != correct_class: 
            not_in_class.append(i)
        else: 
            in_class.append(i)
    #print diff1
    diff2 = 0
    sm = 0.0
    for i in [a, c, b]: 
        for j in in_class: 
            for k in not_in_class: 
                P = K[i, j] / (K[i,j] + K[i,k])
                sm += 1
                diff2 += P
    diff2s.append(diff2/sm)
    return X1


N = 10
no_dims = 10
X = np.random.rand(N, no_dims)
alpha = no_dims-1
classes = np.random.randint(2, size=N)
#print triplet
#print classes[triplet_a], classes[triplet_b], classes[triplet_c]
no_classes = 2
lamb = 0

for t in range(1000):
	for i in range(N):
		for j in range(N):
			for k in range(N):
				if classes[i] == classes[j] and classes[i] != classes[k]:
					triplet = (i, j , k)
					X = prob_difference(X, 
					    N, 
					    no_dims, 
					    alpha,
					    lamb,
					    triplet,
					    classes, 
					    no_classes, 
					    w_right=0.5, 
					    w_wrong=0.5)
	print t
	print "Diff1", diff1s[-1]
	print "Diff2", diff2s[-1]
