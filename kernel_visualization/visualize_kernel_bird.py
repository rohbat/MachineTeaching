import random
import numpy as np
from sklearn import decomposition
from sklearn.manifold import TSNE
import cPickle as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os 

colors_all = ['black', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'lime', 'darkviolet']

def get_img_class_mapping(): 
	# path = '/home/cs101teaching/MachineTeaching/static/oct/data_resized/'
	path = os.path.expanduser('~') + '/Documents/birds/'

	image_list = glob.glob(path + '*/*')
	image_list.sort()

	img_list = [img.replace(os.path.expanduser('~') + '/Documents/', '') for img in image_list]
	print(len(img_list))

	
	classes = np.zeros([len(image_list), ])
	class_names = []
	count = 0
	prev = ''
	for i in range(len(img_list)): 
		class_name = img_list[i].split('/')[-2]
		if class_name not in class_names: 
			class_names.append(class_name)
		# print(class_name)
		# if class_name == 'Macular_edema': 
		# 	classes[i] = 1
		# elif class_name == 'Normal_drusen': 
		# 	classes[i] = 2
		# elif class_name == 'Subretinal_fluid': 
		# 	classes[i] = 3
		if class_name != prev: 
			count += 1
		classes[i] = count
		prev = class_name

	return classes, class_names 


def get_class_color_mapping(classes): 
	colors = []
	for i in range(len(classes)): 
		# if classes[i] == 1: 
		# 	colors.append('red')
		# elif classes[i] == 2: 
		# 	colors.append('green')
		# elif classes[i] == 3: 
		# 	colors.append('cyan')
		colors.append(colors_all[classes[i]-1])
	return colors



def main(): 



	'''
	load data
	'''
	# kernel_file = '../static/X_initial_opt.npy'
	kernel_file = '../static/X_initial_bird.npy'
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
	classes, class_names = get_img_class_mapping()
	classes = classes.astype(int)


	'''
	map class to color
	'''
	colors = get_class_color_mapping(classes)	

	fig = plt.figure()
	ax = plt.subplot(111)
	plt.title('Bird Kernel')
	ax.scatter(kernel_pca[:,0], kernel_pca[:,1], c=colors)

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	h = []
	for i in range(len(colors_all)): 
		h.append(mpatches.Patch(color = colors_all[i], label=class_names[i]))
	# red_patch = mpatches.Patch(color='red', label='Macular_edema')
	# green_patch = mpatches.Patch(color='green', label='Normal_drusen')
	# blue_patch = mpatches.Patch(color='cyan', label='Subretinal_fluid')
	# plt.legend(handles=[red_patch, green_patch, blue_patch])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=h, prop={'size':10})
	plt.savefig('bird_kernel_best.png')
	plt.show()




if __name__ == '__main__':
	main()

