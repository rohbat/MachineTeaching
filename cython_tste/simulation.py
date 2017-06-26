from __future__ import division
import numpy as np
import random 
import os 
import glob
import matplotlib.pyplot as plt
from tste_next_point import *


img_path = '\\Users\\Audrey\\Documents\\birds\\images\\'
def get_classes(path): 
    class_names = glob.glob(path + '*')
    class_names.sort()

    class_name_dict_bird = {}
    for class_name in class_names:
        class_name_dict_bird[class_name] = glob.glob(class_name + "\\*")

    name_class = {}
    for k, v in class_name_dict_bird.items():
        for elm in v:
            name_class[elm] = k

    image_list = glob.glob(path + "*\\*")
    image_list.sort()
    N = len(image_list)

    classes = np.zeros(len(image_list), dtype=int)
    for i in range(len(image_list)):
        classes[i] = class_names.index(name_class[image_list[i]])


    class_names = [c.replace(path, "") for c in class_names]

    classes_dict_bird = {}
    for i in range(len(class_names)):
        classes_dict_bird[i] = []

    for i in range(N):
        classes_dict_bird[classes[i]].append(i)

    classes_dict_copy = classes_dict_bird.copy()

    for key in classes_dict_bird.keys():
        classes_dict_copy['not'+str(key)] = []
        for key1 in classes_dict_bird.keys():
            if key != key1 and not 'not' in str(key1):
                classes_dict_copy['not'+str(key)].extend(classes_dict_bird[key1])

    return classes, classes_dict_copy



    # return classes, classes_dict

classes, classes_dict = get_classes(img_path)

no_classes = 10
sample_class = .2
N = len(classes)
no_dims = 5
alpha = no_dims - 1.0
eta = 0.1
lamb = 0
train = range(N)
user = 'perfect'



# classes, classes_dict 


def teach_response_prob(i, j, k, X): 
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

def teach_response_perf(i, j, k, X): 
    if classes[i] == classes[j]: 
        return [i, j, k]
    elif classes[i] == classes[k]: 
        return [i, k, j]



def teach(n, rounds, triplet_dict, X_dict, classes, classes_dict_bird): 
    print('round: ', rounds)
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
            
            print('triplet: ', main, comp1, comp2, i, key)
            chosen = teach_response_perf(main, comp1, comp2, X)
            tste_grad(X, N, no_dims, chosen, 0, no_dims-1.0, K, Q, G)
            X_dict[key] = X - 0.4 * G

            triplet_dict[key][n*rounds + i] = np.asarray(chosen)
    return triplet_dict, X_dict



def test(n, rounds, acc_dict, X_dict, classes, classes_dict_bird): 
    for i in range(n): 
        for key in range(4): 
            X = X_dict[key]
            (main, comp1, comp2) = random_triplet(train, classes, classes_dict_bird)
            chosen = teach_response_perf(main, comp1, comp2, X)

            acc_dict[key][n*rounds + i] = np.asarray(chosen)

            # acc_dict[key][n*rounds + i] = ans
    return acc_dict







def main(): 
    
    # classes, classes_dict = get_classes(img_path)
    # print(classes_dict.keys())

    X_path = '\\Users\\Audrey\\Documents\\cs101a\\static\\X_initial_bird.npy'
    X = np.load(X_path)

    triplet_dict = {}
    test_dict = {}
    X_dict = {}
    n_teach = 5
    n_test = 100 
    rounds = 6

    for i in range(4): 
        triplet_dict[i] = np.zeros([n_teach * rounds, 3])
        test_dict[i] = np.zeros([n_test * rounds, 3])
        X_dict[i] = X

    for i in range(rounds): 
        triplet_dict, X_dict = teach(n_teach, i, triplet_dict, X_dict, classes, classes_dict)
        test_dict = test(n_test, i, test_dict, X_dict, classes, classes_dict)
    
        np.save('..\\sim\\triplet_dict_' + user, triplet_dict)
        np.save('..\\sim\\X_dict_' + user + '_' + str(i), X_dict)
        np.save('..\\sim\\test_dict_' + user, test_dict)






if __name__ == '__main__':
    main()



