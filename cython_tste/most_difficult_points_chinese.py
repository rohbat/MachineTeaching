import numpy as np
import random
import operator
import glob
import os

def compute_kernel_at_pair(X, N, pair, no_dims, alpha):
    i, j = pair
    diff = X[i, :] - X[j, :]
    k_ij = np.dot(diff, diff)
    return (1 + k_ij / alpha) ** ((alpha + 1) / -2)


def most_uncertain_triplets(train, X,N,no_dims,alpha,classes,classes_dict):

    triplet_probs = {}

    for i in train:
        print i
        for j in classes_dict[classes[i]]:
            if j == i:
                continue
            k_ij = compute_kernel_at_pair(X, N, (i, j), no_dims, alpha)
            for k in classes_dict['not'+str(classes[i])]:
                k_ik = compute_kernel_at_pair(X, N, (i, k), no_dims, alpha)
                P = k_ij / (k_ij + k_ik)
                triplet_probs[(i, j, k)] = abs(P - 0.5)

    sorted_triplets = sorted(triplet_probs.items(), key=operator.itemgetter(1))

    np.save('sorted_triplets.npy', sorted_triplets)
    hardest100 = sorted_triplets[:100]
    hard4 = random.sample(hardest100, 4)

    return hardest100, hard4

def main():
    X = np.load("MachineTeaching/static/X_initial_chinese.npy")
    no_dims = 5
    alpha = no_dims - 1.0

    # Set up classes and classes_dict_chinese
    path = os.getcwd()
    class_names = glob.glob(path + "/MachineTeaching/static/chinese/ims/*")
    class_names.sort()

    class_name_dict_chinese = {}
    for class_name in class_names:
        class_name_dict_chinese[class_name] = glob.glob(class_name + "/*")

    name_class = {}
    for k, v in class_name_dict_chinese.iteritems():
        for elm in v:
            name_class[elm] = k

    image_list = glob.glob(path + "/MachineTeaching/static/chinese/ims/*/*")
    image_list.sort()
    N = len(image_list)
    print N
    train = range(N)

    classes = np.zeros(N, dtype=int)
    for i in range(N):
        classes[i] = class_names.index(name_class[image_list[i]])


    class_names = [c.replace(path + "/MachineTeaching/static/chinese/ims/", "") for c in class_names]
    print 'class names: ', class_names

    classes_dict_chinese = {}
    for i in range(len(class_names)):
        classes_dict_chinese[i] = []

    for i in range(N):
        classes_dict_chinese[classes[i]].append(i)

    for key in classes_dict_chinese.keys():
        classes_dict_chinese['not'+str(key)] = []
        for key1 in classes_dict_chinese.keys():
            if key != key1 and not 'not' in str(key1):
                classes_dict_chinese['not'+str(key)].extend(classes_dict_chinese[key1])

    print classes_dict_chinese
    hardest100, hard4 = most_uncertain_triplets(train, X,N,no_dims,alpha,classes,classes_dict_chinese)
    print hardest100
    print hard4


if __name__ == '__main__':
    main()