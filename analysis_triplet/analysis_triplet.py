from __future__ import print_function
from __future__ import division
import math
import random
import numpy as np
import matplotlib.pyplot as plt

train_total=15
test_total=20

difficult_triplets_distance = [(147, 142, 451), (30, 59, 619), (261, 288, 37), \
(452, 288, 536), (488, 701, 43), (539, 701, 376)]
difficult_triplets_acc = [(95, 216, 360), (95, 216, 607), (120, 81, 95), \
(120, 81, 607), (127, 69, 95), (127, 69, 360)]
difficult_triplets_tste = [(648, 617, 81), (612, 497, 392), (644, 666, 346), \
(701, 599, 7), (520, 664, 150), (115, 84, 708), (99, 53, 284), (561, 610, 289), \
(18, 159, 302), (26, 98, 641)] 

def load_files(n): 
	ans_dict = np.load('./analysis_triplet/ans_dict_chinese' + str(n) + '.npy')
	error_dict = np.load('./analysis_triplet/error_dict_chinese' + str(n) + '.npy')
	method_dict = np.load('./analysis_triplet/method_dict_chinese' + str(n) + '.npy')
	train_dict = np.load('./analysis_triplet/train_ans_dict_chinese' + str(n) + '.npy')
	question_dict = np.load('./analysis_triplet/test_images_dict_chinese' + str(n) + '.npy')

	return ans_dict.item(), error_dict.item(), method_dict.item(), train_dict.item(), question_dict.item()





def main(): 
	results_test_dict = {1:[], 
	2:[], 
	3:[], 
	4:[]}

	easy_test_dict = {1:[], 
	2:[], 
	3:[], 
	4:[]}

	diff_test_dict = {1:[], 
	2:[], 
	3:[], 
	4:[]}

	distance_dict = {1:[0, 0], 
	2:[0, 0], 
	3:[0, 0], 
	4:[0, 0]}

	acc_dict = {1:[0, 0], 
	2:[0, 0], 
	3:[0, 0], 
	4:[0, 0]}

	tste_dict = {1:[0, 0], 
	2:[0, 0], 
	3:[0, 0], 
	4:[0, 0]}

	results_train_dict = {1:[], 
	2:[], 
	3:[], 
	4:[]}

	learning_dict = {1:np.zeros([train_total]), 
	2:np.zeros([train_total]), 
	3:np.zeros([train_total]), 
	4:np.zeros([train_total])}

	num_total = {1:0, 
	2:0, 
	3:0, 
	4:0}

	for i in range(6): 
		ans_dict, error_dict, method_dict, train_dict, question_dict = load_files(i)
		for key in error_dict: 
			num = len(ans_dict[key])
			if num == test_total: 
				error = error_dict[key]
				method = method_dict[key]
				results_test_dict[method].append(error)

				diff_acc = ans_dict[key][-5:]
				diff_test_dict[method].append(sum(diff_acc))

				easy_acc = ans_dict[key][:-5]
				easy_test_dict[method].append(sum(easy_acc))

				for k in range(5): 
					if question_dict[key][15+k] in difficult_triplets_distance: 
						distance_dict[method][1] += 1
						if ans_dict[key][15+k] == True: 
							distance_dict[method][0] += 1
					
					elif question_dict[key][15+k] in difficult_triplets_acc: 
						acc_dict[method][1] += 1
						if ans_dict[key][15+k] == True: 
							acc_dict[method][0] += 1

					elif question_dict[key][15+k] in difficult_triplets_tste: 
						tste_dict[method][1] += 1
						if ans_dict[key][15+k] == True: 
							tste_dict[method][0] += 1


			num = len(train_dict[key])
			if num == train_total: 
				error = sum(train_dict[key]) / train_total
				method = method_dict[key]
				results_train_dict[method].append(error)
				learning_dict[method] += train_dict[key]
				num_total[method] += 1


	ave = []
	var = []
	nums = []
	for key in results_test_dict: 
		a = np.mean(results_test_dict[key])
		v = np.var(results_test_dict[key])
		n = len(results_test_dict[key])
		ave.append(a)
		var.append(v)
		nums.append(n)
	
	ave = np.ones(4) - ave
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']

	fig = plt.figure()
	plt.bar(range(4), ave, align='center', alpha=0.5, yerr=var)
	plt.xticks(range(4), objects)
	plt.title('Testing Accuracy for Different Teaching Methods')
	plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.savefig('triplet_test_acc_plot.png')


	ave = []
	var = []
	nums = []
	for key in diff_test_dict: 
		diff_test_dict[key] = np.asarray(diff_test_dict[key]) / 5

	for key in diff_test_dict: 
		a = np.mean(diff_test_dict[key])
		v = np.var(diff_test_dict[key])
		n = len(diff_test_dict[key])
		ave.append(a)
		var.append(v)
		nums.append(n)
	
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']

	fig = plt.figure()
	plt.bar(range(4), ave, align='center', alpha=0.5, yerr=var)
	plt.xticks(range(4), objects)
	plt.title('Testing Accuracy of Hard Questions for Different Teaching Methods')
	plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.savefig('triplet_diff_test_acc_plot.png')


	ave = []
	var = []
	nums = []
	for key in easy_test_dict: 
		easy_test_dict[key] = np.asarray(easy_test_dict[key]) / 15

	for key in easy_test_dict: 
		a = np.mean(easy_test_dict[key])
		v = np.var(easy_test_dict[key])
		n = len(easy_test_dict[key])
		ave.append(a)
		var.append(v)
		nums.append(n)
	
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']

	fig = plt.figure()
	plt.bar(range(4), ave, align='center', alpha=0.5, yerr=var)
	plt.xticks(range(4), objects)
	plt.title('Testing Accuracy of Normal Questions for Different Teaching Methods')
	plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.savefig('triplet_normal_test_acc_plot.png')


	ave = []
	var = []
	nums = []
	for key in results_train_dict: 
		a = np.mean(results_train_dict[key])
		v = np.var(results_train_dict[key])
		n = len(results_train_dict[key])
		ave.append(a)
		var.append(v)
		nums.append(n)
	
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']

	fig = plt.figure()
	plt.bar(range(4), ave, align='center', alpha=0.5, yerr=var)
	plt.xticks(range(4), objects)
	plt.title('Training Accuracy for Different Teaching Methods')
	plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.savefig('triplet_train_acc.png')

	for key in learning_dict: 
		for i in range(1, train_total): 
			learning_dict[key][i] += learning_dict[key][i-1]
		learning_dict[key] /= (train_total*num_total[key])
	print(learning_dict)
	fig = plt.figure()
	for key in learning_dict: 
		plt.plot(range(1, train_total+1), learning_dict[key])
	plt.title('Training Progress for Different Teaching Methods')
	plt.xlabel('Iteration')
	plt.ylabel('Accuracy')
	plt.legend(['random', 'most uncertain', 'best gradient', 'BG rand'], loc='lower right')
	plt.savefig('triplet_train_progress.png')
	plt.show()

	fig = plt.figure()
	plt.subplot(3, 1, 1)
	ave = []
	for key in distance_dict: 
		ave.append(distance_dict[key][0] / distance_dict[key][1])
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']
	plt.bar(range(4), ave, align='center', alpha=0.5)
	# plt.xticks(range(4), objects)
	plt.title('Testing Accuracy of Distance-Based Difficult Questions')
	plt.xticks([])
	# plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.ylim([0.0, 1.0])

	plt.subplot(3, 1, 2)
	ave = []
	for key in acc_dict: 
		ave.append(acc_dict[key][0] / acc_dict[key][1])
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']
	plt.bar(range(4), ave, align='center', alpha=0.5)
	# plt.xticks(range(4), objects)
	plt.title('Testing Accuracy of Kernel Accuracy-Based Difficult Questions')
	plt.xticks([])
	# plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.ylim([0.0, 1.0])

	plt.subplot(3, 1, 3)
	ave = []
	for key in tste_dict: 
		ave.append(tste_dict[key][0] / tste_dict[key][1])
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']
	plt.bar(range(4), ave, align='center', alpha=0.5)
	plt.xticks(range(4), objects)
	plt.title('Testing Accuracy of TSTE-Based Difficult Questions')
	plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.ylim([0.0, 1.0])
	
	plt.savefig('triplet_3_diff_acc_plot.png')

	print(distance_dict)
	print(acc_dict)
	print(tste_dict)





if __name__ == '__main__':
	main()	