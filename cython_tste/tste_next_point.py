import numpy as np
from numpy.core.umath_tests import inner1d
import random

def tste_grad(X, N, no_dims, triplet, lamb, alpha, K, Q, dC):
    """ Compute the cost function and gradient update of t-STE """
    triplet_A = triplet[0]
    triplet_B = triplet[1]
    triplet_C = triplet[2]
    for i in [triplet_A]:
        for j in [triplet_B, triplet_C]:
            diff = X[i, :] - X[j, :]
            K[i, j] = inner1d(diff, diff)
            Q[i, j] = (1 + K[i, j] / alpha) ** -1
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)
            # Now, K[i,j] = ((sqdist(i,j)/alpha + 1)) ** (-0.5*(alpha+1)),
            # which is exactly the numerator of p_{i,j} in the lower right of
            # t-STE paper page 3.
            # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab

    P = K[triplet_A, triplet_B] / (
        K[triplet_A, triplet_B] + 
        K[triplet_A, triplet_C])
    # This is exactly p_{ijk}, which is the equation in the lower-right
    # of page 3 of the t-STE paper.

    const = (alpha + 1) / alpha
    A_to_B = ((1 - P) * 
              Q[triplet_A, triplet_B] * 
              (X[triplet_A, :] - X[triplet_B, :]))
    A_to_C = ((1 - P) * 
              Q[triplet_A, triplet_C] * 
              (X[triplet_A, :] - X[triplet_C, :]))

    dC[triplet_A, :] = const * (A_to_B - A_to_C)
    dC[triplet_B, :] = const * (-A_to_B)
    dC[triplet_C, :] = const * (A_to_C)

    if lamb > 0:
        for n in triplet:
            dC[n, :] = dC[n, :] + 2 * lamb * X[n, :]

def probability(X,
    N,
    triplet,
    no_dims,
    alpha,
    K):

    for i in triplet: 
        for j in range(N):
            diff = X[i, :] - X[j, :]
            K[i, j] = inner1d(diff, diff)
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)
            
            
def probability_rand(X, N, triplet, no_dims, alpha, K, rand_triplets):
    for i in triplet: 
        for j in rand_triplets[i][0]:
            diff = X[i, :] - X[j, :]
            K[i, j] = inner1d(diff, diff)
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)
        for j in rand_triplets[i][1]:
            diff = X[i, :] - X[j, :]
            K[i, j] = inner1d(diff, diff)
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)

diff1s = []
diff2s = []

def prob_difference(X,
    N,
    no_dims,
    alpha,
    lamb,
    triplet,
    classes,
    classes_dict,
    no_classes=3,
    w_right=0.5,
    w_wrong=0.5,
    eta=0.2,
    sample_class = 0.05): 

    a, b, c = triplet
    K = np.zeros((N, N))
    Q = np.zeros((N, N))
    G = np.zeros((N, no_dims))

    tste_grad(X, N, no_dims, (a, b, c), lamb, alpha, K, Q, G)
    X1 = X - (float(eta) / no_classes * N) * G
    
    rand_triplets = {i:(random.sample(classes_dict[classes[i]], int(sample_class*len(classes_dict[classes[i]]))), random.sample(classes_dict["not"+str(classes[i])], int(sample_class*len(classes_dict["not"+str(classes[i])])))) for i in triplet}
    probability_rand(X1, N, triplet, no_dims, alpha, K, rand_triplets)

    # Compute probability (or log-prob) for each triplet
    diff1 = 0.0
    sm = 0.0
    for i in triplet:
        for j in rand_triplets[i][0]: 
            for k in rand_triplets[i][1]: 
                P = K[i, j] / (K[i, j] + K[i, k])
                diff1 += 1.0-P
                sm += 1
    diff1 /= sm
    diff1s.append(diff1)

    tste_grad(X, N, no_dims, (a, c, b), lamb, alpha, K, Q, G)
    X2 = X - (float(eta) / no_classes * N) * G
    probability_rand(X2, N, triplet, no_dims, alpha, K, rand_triplets)

    diff2 = 0.0
    sm = 0.0
    for i in triplet:
        for j in rand_triplets[i][0]: 
            for k in rand_triplets[i][1]:
                P = K[i, j] / (K[i, j] + K[i, k])
                diff2 += 1.0-P
                sm += 1
    #print sm
    diff2 /= sm
    diff2s.append(diff2)
    return X1, (diff1+diff2)/2.0

def main():
    global diff1s, diff2s
    N = 750
    no_dims = 10
    no_classes = 3
    X = np.random.rand(N, no_dims)
    alpha = no_dims - 1
    classes = np.random.randint(no_classes, size=N)
    classes_dict = {}
    for i in range(no_classes):
        classes_dict[i] = []
    for i in range(len(classes)):
        classes_dict[classes[i]].append(i)
    
    for key in classes_dict.keys():
        classes_dict['not'+str(key)] = []
        for key1 in classes_dict.keys():
            if key != key1 and not 'not' in str(key1):
                classes_dict['not'+str(key)].extend(classes_dict[key1])
    # print triplet
    # print classes[triplet_a], classes[triplet_b], classes[triplet_c]
    sample_class = 0.02
    lamb = 0
    import time
    for t in range(1000):
        count = 0
        min_avg = 1000000
        best_triplet = None
        diff1s = []
        diff2s = []
        t1 = time.time()
        for i in random.sample(range(N), int(N*sample_class)):
            for j in random.sample(classes_dict[classes[i]], int(sample_class*len(classes_dict[classes[i]]))):
                for k in random.sample(classes_dict['not'+str(classes[i])], int(sample_class*len(classes_dict["not"+str(classes[i])]))):
                    triplet = (i, j , k)
                    X, avg = prob_difference(X,
    				    N,
    				    no_dims,
    				    alpha,
    				    lamb,
    				    triplet,
    				    classes,
    				    classes_dict,
    				    no_classes,
    				    w_right=0.5,
    				    w_wrong=0.5)
                    if avg < min_avg:
                        min_avg = avg
                        best_triplet = triplet
                    count += 1
        t2 = time.time()
        print "Time", t2 - t1
        print "Triplets Sampled", count
        print "Diff1", np.average(diff1s)
        print "Diff2", np.average(diff2s)
        print "Min Avg", min_avg, best_triplet, '\n'

if __name__ == '__main__':
    main()
