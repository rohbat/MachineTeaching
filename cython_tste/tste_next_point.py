import numpy as np
import random
import time

def tste_grad(X, N, no_dims, triplet, lamb, alpha, K, Q, dC):
    """ Compute the cost function and gradient update of t-STE """
    triplet_A = triplet[0]
    triplet_B = triplet[1]
    triplet_C = triplet[2]
    for i in [triplet_A]:
        for j in [triplet_B, triplet_C]:
            diff = X[i, :] - X[j, :]
            K[i, j] = np.dot(diff, diff)
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
    return P

def compute_kernel(X, N, no_dims, alpha, K):
    for i in range(N): 
        for j in range(i, N):
            diff = X[i, :] - X[j, :]
            K[i, j] = np.dot(diff, diff)
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)
            K[j, i] = K[i, j]

def compute_kernel_from_triplet(X, N, triplet, no_dims, alpha, K):
    for i in triplet: 
        for j in range(N):
            diff = X[i, :] - X[j, :]
            K[i, j] = np.dot(diff, diff)
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)
            
            
def compute_kernel_from_triplet_to_dst_triplets(X, X_triplet, N, triplet, no_dims, alpha, K, dst_triplets_dict):
    for i in triplet: 
        for j in dst_triplets_dict[i][0]:
            if j in triplet:
                diff = X_triplet[i, :] - X_triplet[j, :]
            else:
                diff = X_triplet[i, :] - X[j, :]
            K[i, j] = np.dot(diff, diff)
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)
        for j in dst_triplets_dict[i][1]:
            if j in triplet:
                diff = X_triplet[i, :] - X_triplet[j, :]
            else:
                diff = X_triplet[i, :] - X[j, :]
            K[i, j] = np.dot(diff, diff)
            K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)

def compute_kernel_at_pair(X, N, pair, no_dims, alpha):
    i, j = pair
    diff = X[i, :] - X[j, :]
    k_ij = np.dot(diff, diff)
    return (1 + k_ij / alpha) ** ((alpha + 1) / -2)

def compute_kernel_and_probability_at_triplet(X, N, triplet, no_dims, alpha, K):
    i, j, k = triplet
    diff = X[i, :] - X[j, :]
    K[i, j] = np.dot(diff, diff)
    K[i, j] = (1 + K[i, j] / alpha) ** ((alpha + 1) / -2)
    diff = X[i, :] - X[k, :]
    K[i, k] = np.dot(diff, diff)
    K[i, k] = (1 + K[i, k] / alpha) ** ((alpha + 1) / -2)
    return K[i, j] / (K[i, j] + K[i, k])
    

def random_triplet(train, classes, classes_dict):
    i = random.choice(train)
    j = random.choice(classes_dict[classes[i]])
    k = random.choice(classes_dict['not'+str(classes[i])])
    return (i, j, k)

def most_uncertain_triplet(train, X,N,no_dims,alpha,lamb,classes,classes_dict,no_classes=3,eta=0.2,sample_class = 0.135):
    K = np.zeros((N, N))
    #compute_kernel(X, N, no_dims, alpha, K)
    best_triplet = None
    best_diff = 10000000
    best_p = None
    count = 0
    for i in random.sample(train, int(len(train)*sample_class)):
        for j in random.sample(classes_dict[classes[i]], int(sample_class*len(classes_dict[classes[i]]))):
            k_ij = compute_kernel_at_pair(X, N, (i, j), no_dims, alpha)
            for k in random.sample(classes_dict['not'+str(classes[i])], int(sample_class*len(classes_dict["not"+str(classes[i])]))):
                k_ik = compute_kernel_at_pair(X, N, (i, k), no_dims, alpha)
                P = k_ij / (k_ij + k_ik)
                count += 1
                #P = compute_kernel_and_probability_at_triplet(X, N, (i, j, k), no_dims, alpha, K)
                if abs(P-0.5) < best_diff:
                    best_diff = abs(P-0.5)
                    best_triplet = (i, j, k)
                    best_p = P
    return best_triplet, best_p

def best_gradient_triplet(train, X,N,no_dims,alpha,lamb,classes,classes_dict,no_classes=3,eta=0.2, sample_class = 0.06):
    best_triplet = None
    max_val = 0
    best_p = None
    X1 = np.zeros((N, no_dims))
    K = np.zeros((N, N))
    Q = np.zeros((N, N))
    G = np.zeros((N, no_dims))
    for i in random.sample(train, int(len(train)*sample_class)):
        for j in random.sample(classes_dict[classes[i]], int(sample_class*len(classes_dict[classes[i]]))):
            for k in random.sample(classes_dict['not'+str(classes[i])], int(sample_class*len(classes_dict["not"+str(classes[i])]))):
                triplet = (i, j, k)
                p = tste_grad(X, N, no_dims, (i, j, k), lamb, alpha, K, Q, G)
                for a in triplet:
                    X1[a, :] = X[a, :] - (float(eta) / no_classes * N) * G[a, :]
                p1 = compute_kernel_and_probability_at_triplet(X1, N, triplet, no_dims, alpha, K)
                tste_grad(X, N, no_dims, (i, k, j), lamb, alpha, K, Q, G)
                for a in triplet:
                    X1[a, :] = X[a, :] - (float(eta) / no_classes * N) * G[a, :]
                p2 = compute_kernel_and_probability_at_triplet(X1, N, triplet, no_dims, alpha, K)
                val = p*p1+(1.0-p)*p2 - p
                if val > max_val:
                    max_val = val
                    best_triplet = (i, j, k)
                    best_p = p
    return best_triplet, best_p

def score_triplet_random_sample(X,N,no_dims,alpha,lamb,triplet,classes,classes_dict, K, Q, G, X_new, no_classes=3,eta=0.2,sample_class = 0.02): 

    a, b, c = triplet
    rand_triplets = {i:(random.sample(classes_dict[classes[i]], int(sample_class*len(classes_dict[classes[i]]))), random.sample(classes_dict["not"+str(classes[i])], int(sample_class*len(classes_dict["not"+str(classes[i])])))) for i in triplet}
    if not b in rand_triplets[a][0]:
        rand_triplets[a][0].append(b)
    if not c in rand_triplets[a][1]:
        rand_triplets[a][1].append(c) 
    compute_kernel_from_triplet_to_dst_triplets(X, X, N, triplet, no_dims, alpha, K, rand_triplets)
    prob0 = 0.0
    sm = 0.0
    for i in triplet:
        for j in rand_triplets[i][0]: 
            for k in rand_triplets[i][1]: 
                P = K[i, j] / (K[i, j] + K[i, k])
                prob0 += P
                sm += 1
    prob0 /= sm

    p = tste_grad(X, N, no_dims, (a, b, c), lamb, alpha, K, Q, G)
    for t in triplet:
        X_new[t, :] = X[t, :] - (float(eta) / no_classes * N) * G[t, :]
    compute_kernel_from_triplet_to_dst_triplets(X, X_new, N, triplet, no_dims, alpha, K, rand_triplets)

    # Compute probability (or log-prob) for each triplet
    prob1 = 0.0
    sm = 0.0
    for i in triplet:
        for j in rand_triplets[i][0]: 
            for k in rand_triplets[i][1]: 
                P = K[i, j] / (K[i, j] + K[i, k])
                prob1 += P
                sm += 1
    prob1 /= sm

    tste_grad(X, N, no_dims, (a, c, b), lamb, alpha, K, Q, G)
    for t in triplet:
        X_new[t, :] = X[t, :] - (float(eta) / no_classes * N) * G[t, :]
    compute_kernel_from_triplet_to_dst_triplets(X, X_new, N, triplet, no_dims, alpha, K, rand_triplets)

    prob2 = 0.0
    sm = 0.0
    for i in triplet:
        for j in rand_triplets[i][0]: 
            for k in rand_triplets[i][1]:
                P = K[i, j] / (K[i, j] + K[i, k])
                prob2 += P
                sm += 1
    #print sm
    prob2 /= sm
    return p*prob1+(1.0-p)*prob2-prob0, p

def best_gradient_triplet_rand_evaluation(train,X,N,no_dims,alpha,lamb,classes,classes_dict,no_classes=3,eta=0.2,sample_class = 0.025):
    best_triplet = None
    max_val = 0
    best_p = None
    K = np.zeros((N, N))
    Q = np.zeros((N, N))
    G = np.zeros((N, no_dims))
    X_new = np.zeros(X.shape)
    for i in random.sample(train, int(len(train)*sample_class)):
        for j in random.sample(classes_dict[classes[i]], int(sample_class*len(classes_dict[classes[i]]))):
            for k in random.sample(classes_dict['not'+str(classes[i])], int(sample_class*len(classes_dict["not"+str(classes[i])]))):
                triplet = (i, j , k)
                val, p = score_triplet_random_sample(X,N,no_dims,alpha,lamb,triplet,classes, classes_dict, K, Q, G, X_new, no_classes)
                if val > max_val:
                    max_val = val
                    best_triplet = triplet
                    best_p = p
    return best_triplet, best_p

def main():
    N = 750
    no_dims = 10
    no_classes = 3
    X = np.random.rand(N, no_dims)
    alpha = no_dims - 1.0
    train = list(range(N))
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

    lamb = 0
    for t in range(1000):
        print t
        t1 = time.time()
        best_triplet = random_triplet(train,classes,classes_dict)
        t2 = time.time()
        print best_triplet, t2-t1
        t1 = time.time()
        best_triplet, best_p = most_uncertain_triplet(train, X,N,no_dims,alpha,lamb,classes,classes_dict)
        t2 = time.time()
        print best_triplet, best_p, t2-t1
        t1 = time.time()
        best_triplet, best_p = best_gradient_triplet(train, X,N,no_dims,alpha,lamb,classes,classes_dict)
        t2 = time.time()
        print best_triplet, best_p, t2-t1
        t1 = time.time()
        best_triplet, best_p = best_gradient_triplet_rand_evaluation(train, X,N,no_dims,alpha,lamb,classes,classes_dict)
        t2 = time.time()
        print best_triplet, best_p, t2-t1

if __name__ == '__main__':
    main()
