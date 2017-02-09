import random
import numpy as np
from sklearn import decomposition
import cPickle as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os 


def get_img_class_mapping(): 
	path = '/Users/ottery/Documents/machine_teaching_data/chinese/ims/'
	image_list = glob.glob(path + '*/*')
	image_list.sort()

	img_list = [img.replace('/Users/ottery/Documents/machine_teaching_data/', '') for img in image_list]
	print(len(img_list))

	
	classes = np.zeros([len(image_list), ])
	for i in range(len(img_list)): 
		class_name = img_list[i].split('/')[-2]
		if class_name == 'grass': 
			classes[i] = 1
		elif class_name == 'mound': 
			classes[i] = 2
		elif class_name == 'stem': 
			classes[i] = 3

	return classes 


def get_class_color_mapping(classes): 
	colors = []
	for i in range(len(classes)): 
		if classes[i] == 1: 
			colors.append('red')
		elif classes[i] == 2: 
			colors.append('green')
		elif classes[i] == 3: 
			colors.append('blue')
	return colors



def main(): 

	'''
	load data
	'''
	kernel_file = '../static/X_initial.npy'
	# kernel_file = './static/X_initial_seabed.npy'
	kernel = np.load(kernel_file)

	'''
	PCA into 2D
	'''
	pca = decomposition.PCA(n_components=2)
	pca.fit(kernel)
	kernel_pca = pca.transform(kernel)

	'''
	map image indices to class 
	'''
	classes = get_img_class_mapping()

	'''
	map class to color
	'''
	colors = get_class_color_mapping(classes)	

	fig = plt.figure()
	plt.title('Chinese Character Kernel')
	plt.scatter(kernel_pca[:,0], kernel_pca[:,1], c=colors)
	red_patch = mpatches.Patch(color='red', label='grass')
	green_patch = mpatches.Patch(color='green', label='mound')
	blue_patch = mpatches.Patch(color='blue', label='stem')
	plt.legend(handles=[red_patch, green_patch, blue_patch])
	plt.savefig('chinese_kernel.png')
	plt.show()




if __name__ == '__main__':
	main()

