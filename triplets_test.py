from __future__ import print_function
from __future__ import division 
import numpy as np
import cPickle as cp
import os
import glob


db_data = np.load('triplets_chinese.npy')
triplets = []
for pg_model in db_data:
    indices = pg_model.get_index_list()
    if not None in indices:
        triplets.append(indices)


path = os.path.expanduser('~')
class_names = glob.glob(path + '/Documents/machine_teaching_data/chinese/ims/*')
class_names.sort()

image_list = glob.glob(path + '/Documents/machine_teaching_data/chinese/ims/*/*')
image_list.sort()

print(path)
print(image_list[:10])

class_dic = {}
score_dic = {}
for i in range(len(image_list)):  
	score_dic[i] = [0, 0] # (right, total)
	
	if 'grass' in image_list[i]: 
		class_dic[i] = 0
	elif 'stem' in image_list[i]: 
		class_dic[i] = 2
	else: 
		class_dic[i] = 1



for i in range(len(triplets)): 
	x = triplets[i]

	if class_dic[x[0]] == class_dic[x[1]]: 
		score_dic[x[0]][0] += 1 

	score_dic[x[0]][1] += 1

scores = np.zeros(len(image_list))
for key in score_dic: 
	scores[key] = score_dic[key][0] / score_dic[key][1]

np.save('chinese_triplet_scores.npy', scores)









# if __name__ == '__main__':
# 	main()