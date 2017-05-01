import random
import numpy as np
from sklearn import decomposition
from sklearn.manifold import TSNE
import cPickle as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os 


def get_img_class_mapping(): 
	path = '/home/cs101teaching/MachineTeaching/static/oct/data_resized/'
	image_list = glob.glob(path + '*/*')
	image_list.sort()

	img_list = [img.replace('/home/cs101teaching/MachineTeaching', '') for img in image_list]
	print(len(img_list))

	
	classes = np.zeros([len(image_list), ])
	for i in range(len(img_list)): 
		class_name = img_list[i].split('/')[-2]
		if class_name == 'Macular_edema': 
			classes[i] = 1
		elif class_name == 'Normal_drusen': 
			classes[i] = 2
		elif class_name == 'Subretinal_fluid': 
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
			colors.append('cyan')
	return colors



def main(): 

	'''
	load data
	'''
	kernel_file = '../static/X_initial_opt.npy'
	kernel = np.load(kernel_file)

	'''
	PCA into 2D
	'''
	#pca = decomposition.PCA(n_components=2)
	#pca.fit(kernel)
	#kernel_pca = pca.transform(kernel)
	model = TSNE()
	kernel_pca = model.fit_transform(kernel)
	'''
	map image indices to class 
	'''
	classes = get_img_class_mapping()


	'''
	map class to color
	'''
	colors = get_class_color_mapping(classes)	

	fig = plt.figure()
	plt.title('Opt Kernel')
	plt.scatter(kernel_pca[:,0], kernel_pca[:,1], c=colors)
	red_patch = mpatches.Patch(color='red', label='Macular_edema')
	green_patch = mpatches.Patch(color='green', label='Normal_drusen')
	blue_patch = mpatches.Patch(color='cyan', label='Subretinal_fluid')
	plt.legend(handles=[red_patch, green_patch, blue_patch])
	plt.savefig('opt_kernel1.png')
	plt.show()




if __name__ == '__main__':
	main()

