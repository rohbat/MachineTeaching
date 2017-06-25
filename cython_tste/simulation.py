from __future__ import division, print_function
import numpy as np
import random 
import matplotlib.pyplot as plt
from tste_next_point import *

no_classes = 10
sample_class = 1
N = len(classes)
no_dims = 5
alpha = no_dims - 1.0
eta = 0.1
lamb = 0



# classes, classes_dict 


def teach_response(i, j, k, X): 
    i, j, k = triplet
    diff = X[i, :] - X[j, :]
    k_ij = np.dot(diff, diff)
    k_ij = (1 + k_ij / alpha) ** ((alpha + 1) / -2)
    
    diff = X[i, :] - X[k, :]
    k_ik = np.dot(diff, diff)
    k_ik = (1 + k_ik / alpha) ** ((alpha + 1) / -2)
    
    p =  k_ij / (k_ij + k_ik)
    r = np.random.uniform(0., 1.)
    if r <= p: 
        return [i, j, k]
    else:
        return [i, k, j]



def teach(n, rounds, triplet_dict, X_dict): 
    for i in range(n): 
        for key in range(4): 
            X = X_dict[key]
            K = np.zeros((N, N))
            Q = np.zeros((N, N))
            G = np.zeros((N, no_dims))
            
            if key == 0: 
                (main, comp1, comp2) = random_triplet(train, classes, classes_dict_bird)
            elif key == 1: 
                ((main, comp1, comp2), p) = most_uncertain_triplet(train,X,N,no_dims,alpha,lamb,classes,classes_dict_bird,eta,no_classes=no_classes,sample_class = sample_class)
            elif key == 2: 
                ((main, comp1, comp2), p) = best_gradient_triplet(train,X,N,no_dims,alpha,lamb,classes,classes_dict_bird,eta,no_classes=no_classes,sample_class = sample_class)
            elif key == 3: 
                ((main, comp1, comp2), p) = best_gradient_triplet_rand_evaluation(train,X,N,no_dims,alpha,lamb,classes,classes_dict_bird,eta,no_classes=no_classes,sample_class = sample_class)
            
            chosen = teach_response(main, comp1, comp2, X)
            tste_grad(X, N, no_dims, chosen, 0, no_dims-1.0, K, Q, G)
            X_dict[key] = X - 0.4 * G

            triplet_dict[key][(n-1)*rounds + i] = np.asarray([main, comp1, comp2])
    return triplet_dict, X_dict



def test(n, rounds, acc_dict, X_dict): 
    for i in range(n): 
        for key in range(4): 
            X = X_dict[key]
            (main, comp1, comp2) = random_triplet(train, classes, classes_dict_bird)
            chosen = teach_response(main, comp1, comp2, X)

            if classes[chosen[0]] == classes[chosen[1]]: 
                ans = True
            else: 
                ans = False

            acc_dict[key][n*rounds + i] = ans
    return acc_dict




def main(): 
    X = np.load("MachineTeaching/static/X_initial_bird.npy")
    triplet_dict = {}
    acc_dict = {}
    X_dict = {}
    n_teach = 1
    n_test = 1 
    rounds = 5

    for i in range(4): 
        triplet_dict[i] = np.zeros([n_teach * rounds, 3])
        acc_dict[i] = np.zeros([n_test * rounds])
        X_dict[i] = X

    for i in range(rounds): 
        triplet_dict, X_dict = teach(n_teach, i, triplet_dict, X_dict)
        acc_dict = test(n_test, i, acc_dict, X_dict)

    visualize(triplet_dict, acc_dict, X_dict, X)




if __name__ == '__main__':
    main()



