from __future__ import print_function
from __future__ import division
import math
import random
import numpy as np
import matplotlib.pyplot as plt



def load_files(n): 
	ans_dict = np.load('./analysis/ans_dict' + str(n) + '.npy')
	error_dict = np.load('./analysis/error_dict' + str(n) + '.npy')
	method_dict = np.load('./analysis/method_dict' + str(n) + '.npy')

	return ans_dict.item(), error_dict.item(), method_dict.item()


def main(): 
	results_dict = {1:[], 
	2:[], 
	3:[], 
	4:[]}

	for i in range(6): 
		ans_dict, error_dict, method_dict = load_files(i)
		for key in error_dict: 
			num = len(ans_dict[key])
			if num == 30: 
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
	
	ave = np.ones(4) - ave
	objects = ['random', 'most uncertain', 'best gradient', 'BG rand']

	fig = plt.figure()
	plt.bar(range(4), ave, align='center', alpha=0.5, yerr=var)
	plt.xticks(range(4), objects)
	plt.title('Accuracy for Different Teaching Methods')
	plt.xlabel('Method')
	plt.ylabel('Accuracy')
	plt.savefig('acc_chinese_plot.png')
	plt.show()




if __name__ == '__main__':
	main()	