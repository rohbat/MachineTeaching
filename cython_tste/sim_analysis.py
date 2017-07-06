from __future__ import division, print_function
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
import glob
from tste_next_point import *
from simulation import *

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


# def show_triplets(path): 
# 	triplet_dict = np.load(path).item()

# 	for key in range(4): 
# 		images = triplet_dict[key]
# 		for i in range(len(images)): 
# 			imgs = images[i]
# 			print(imgs)
# 			# for j in imgs: 
# 			# 	img = Image.open(img_path + imgs[i])
#    #      img.resize((200, 200), Image.ANTIALIAS) # resizes image in-place
#    #      plt.imshow(img)
#    #      plt.axis('off')
#    #  plt.savefig(path + '/MachineTeaching/imgs/' + str(user_selection_method_dict_bird[session['name']]) + '_' + \
#    #      str(user_nclicks_dict_bird[session['name']]))


n_teach = 5
n_test = 500 
rounds = 6
user = 'perf'
train = range(N)
labels = ['rand', 'uncertain', 'BG', 'BG Rand']

X = np.load('\\Users\\Audrey\\Documents\\cs101a\\static\\X_initial_bird.npy')
def test_0(n, X, classes, classes_dict_bird): 
	acc = 0
	for i in range(n): 
		(main, comp1, comp2) = random_triplet(train, classes, classes_dict_bird)
		chosen = teach_response_maj(main, comp1, comp2, X)

		if classes[chosen[0]] == classes[chosen[1]]: 
			acc += 1
	return acc / n
acc_0 = test_0(n_test, X, classes, classes_dict)
print(acc_0)

def test_acc(ans): 
	acc = np.zeros([rounds])

	for i in range(rounds): 
		for j in range(n_test):
			index = n_test * i + j
			triplet = ans[index].astype(int)
			# print(classes[triplet[0]], classes[triplet[1]], classes[triplet[2]])

			if classes[triplet[0]] == classes[triplet[1]]: 
				acc[i] += 1

	return acc 




def graph_testing(path): 
	test_dict = np.load(path + 'test_dict_' + user + '.npy').item()
	print(test_dict.keys())
	print(np.shape(test_dict[0]))

	test_dict2 = np.load(path + 'test_dict_' + user + '_2.npy').item()

	plt.figure()
	for i in range(4): 
		ans = np.mean(test_dict[i], axis=0)
		acc = test_acc(ans) / n_test
		print(acc)

		ans2 = np.mean(test_dict2[i], axis=0)
		acc2 = test_acc(ans2) / n_test
		
		plt.plot(range(rounds * 2 + 1), [acc_0] + list(acc) + list(acc2))
		plt.xlabel('Learning Iteration (% 5)')
		plt.ylabel('Accuracy')
		plt.title('Test Accuracy over 30 Teaching Examples, 30 Users')
	plt.legend(labels, loc = 'upper left')
	plt.savefig('./test_' + str(user) + '_2')


def main(): 
	path = '..\\sim\\'
	graph_testing(path)
	
	# visualize(triplet_dict, acc_dict, X_dict, X)



if __name__ == '__main__':
	main()