from __future__ import print_function
from __future__ import division
import math
import random
import numpy as np
import matplotlib.pyplot as plt

MAX_TRAIN = 16
MAX_TEST = 20

def load_files(n): 
	ans_dict = np.load('../results/testfiles_seabed' + str(n) + '/ans_dict_seabed.npy')
	error_dict = np.load('../results/testfiles_seabed' + str(n) + '/error_dict_seabed.npy')
	method_dict = np.load('../results/testfiles_seabed' + str(n) + '/method_dict_seabed.npy')
	train_dict = np.load('../results/testfiles_seabed' + str(n) + '/train_ans_dict_seabed.npy')

	return ans_dict.item(), error_dict.item(), method_dict.item(), train_dict.item()

def combine_files(): 
	ans_dict_all = {}
	error_dict_all = {}
	method_dict_all = {}
	train_dict_all = {}

	# count1 = 0
	# count2 = 0
	# count3 = 0
	for i in range(6): 
		ans_dict, error_dict, method_dict, train_dict = load_files(i)
		ans_dict_all.update(ans_dict)
		error_dict_all.update(error_dict)
		method_dict_all.update(method_dict)
		train_dict_all.update(train_dict)
	# 	count1 += len(ans_dict.keys())
	# 	count2 += len(error_dict.keys())
	# 	count3 += len(method_dict.keys())
	# print(count1, count2, count3)
	# print(len(ans_dict_all.keys()), len(error_dict_all.keys()), len(method_dict_all.keys()))

	np.save('./ans_dict_all', ans_dict_all)
	np.save('./error_dict_all', error_dict_all)
	np.save('./method_dict_all', method_dict_all)
	np.save('./train_dict_all', train_dict_all)

# outdated, code from before i combined all of the answer/error/method dictionaries
def plot_overall_error(error_dict, method_dict, ans_dict): 
	results_dict = {1:[], 
	2:[], 
	3:[], 
	4:[]}


	for key in error_dict: 
		num = len(ans_dict[key])
		if num == MAX_TEST:  
			error = error_dict[key]
			method = method_dict[key]
			results_dict[method].append(error)

	ave = []
	var = []
	nums = []
	for key in results_dict: 
		a = np.mean(results_dict[key])
		v = np.var(results_dict[key])
		n = len(results_dict[key])
		ave.append(a)
		var.append(v)
		nums.append(n)
	
	print(ave)
	ave = np.ones(4) - ave
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']

	fig = plt.figure()
	plt.bar(range(4), ave, align='center', alpha=0.5, yerr=var)
	plt.xticks(range(4), objects)
	plt.title('Accuracy for Different Teaching Methods')
	plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.savefig('acc_seabed_plot.png')
	plt.show()

def plot_progress(ans_dict, error_dict, method_dict): 

	running_results_dict = {
	1:np.zeros(MAX_TRAIN), 
	2:np.zeros(MAX_TRAIN), 
	3:np.zeros(MAX_TRAIN), 
	4:np.zeros(MAX_TRAIN)}

	score_dict = {
	1:np.zeros(MAX_TRAIN), 
	2:np.zeros(MAX_TRAIN), 
	3:np.zeros(MAX_TRAIN), 
	4:np.zeros(MAX_TRAIN)}
	
	counts = [0, 0, 0, 0]

	for key in ans_dict: 
		answers = ans_dict[key]
		if len(answers) == MAX_TRAIN: # and error_dict[key] > 0.25: 
			counts[method_dict[key]-1] += 1
			# for i in range(1, len(answers)): 
			# 	answers[i] += answers[i-1]
			# # if method_dict[key] in [3, 4]: 
			# # 	print(answers)
			# # print(answers)
			# # print(error_dict[key])
			# running_results_dict[method_dict[key]] += answers

			for i in range(len(answers)): 
				if answers[i] == 1: 
					score_dict[method_dict[key]][i:] += 1

	# # average running results
	# for key in running_results_dict: 
	# 	running_results_dict[key] = running_results_dict[key] / float(16 * counts[key-1])
	# 	print('final accuracy: ', running_results_dict[key][-1])

	# fig = plt.figure()
	# plt.title('Progress Over Teaching Phase (No Removed Data)')
	# plt.xlabel('Teaching Iteration')
	# plt.ylabel('Accuracy')
	# for i in range(1, 5): 
	# 	plt.plot(range(0, 16), running_results_dict[i] / counts[i-1])
	# plt.legend(['random', 'most uncertain', 'best gradient', 'BG rand'], loc='lower right')
	# plt.savefig('progress_seabed.png')

	fig = plt.figure()
	plt.title('Progress Over Teaching Phase')
	plt.xlabel('Teaching Iteration')
	plt.ylabel('Total # Correct so far')
	print(counts)
	print(score_dict)
	for i in range(1, 5): 
		print(len(score_dict[i]))
		plt.plot(range(0, 16), score_dict[i]/ counts[i-1])
	plt.legend(['random', 'most uncertain', 'best gradient', 'BG rand'], loc='lower right')
	plt.savefig('learning_curve_seabed.png')
	plt.show()





def main(): 
	# combine_files()



	'''
	load dictionaries of user answers, overall error, & kernel update method
	'''
	ans_dict = np.load('./ans_dict_all.npy').item()
	error_dict = np.load('./error_dict_all.npy').item()
	method_dict = np.load('./method_dict_all.npy').item()
	train_dict = np.load('./train_dict_all.npy').item()

	# plot_overall_error(error_dict, method_dict, ans_dict)
	


	'''
	generate plot of progress over time
	'''
	plot_progress(train_dict, error_dict, method_dict)


	
	




if __name__ == '__main__':
	main()	