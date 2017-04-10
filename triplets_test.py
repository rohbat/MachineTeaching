from __future__ import print_function
from __future__ import division 
import numpy as np
import cPickle as cp
import os
import glob
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


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


class_dic = {}
score_dic = {}
num_dic = np.zeros(len(image_list))
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
	num_dic[x[0]] += 1

	if class_dic[x[0]] == class_dic[x[1]]: 
		score_dic[x[0]][0] += 1 

	score_dic[x[0]][1] += 1

scores = np.zeros(len(image_list))
for key in score_dic: 
	scores[key] = score_dic[key][0] / score_dic[key][1]

# np.save('chinese_triplet_scores.npy', scores)


# plt.subplot(2, 1, 1)
# plt.title('kernel target accuracy')
# plt.hist(scores, bins=20)

# plt.subplot(2, 1, 2)
# plt.title('kernel target frequency')
# plt.hist(num_dic, bins = 20)
# plt.show()



x_initial = np.load('X_initial.npy')

distances = np.zeros([x_initial.shape[0], x_initial.shape[0]])

for i in range(len(x_initial)): 
	for j in range(len(x_initial)): 
		distances[i, j] = np.linalg.norm(x_initial[i,:]-x_initial[j,:])

start_1 = 240
start_2 = 480

d_1_2 = np.zeros([240])
d_1_3 = np.zeros([240])

# 147
# distance to cluster 2
for i in range(start_1): 
	d_1_2[i] = np.sum(distances[i, start_1:start_2])
	d_1_3[i] = np.sum(distances[i, start_2:])


x_1_2 = np.argmin(d_1_2) 
print(x_1_2)
print(scores[x_1_2])

d12 = distances[np.argmin(d_1_2)]
y_1_2 = np.argmax(d12[:start_1])
print(y_1_2)
print(scores[y_1_2])

z_1_2 = np.argmin(d12[start_1:start_2])
print(z_1_2 + 240)
print(scores[z_1_2 + 240])

print('\n')

x_1_3 = np.argmin(d_1_3)
print(x_1_3)
print(scores[x_1_3])

d13 = distances[np.argmin(d_1_3)] 
y_1_3 = np.argmax(d13[:start_1])
print(y_1_3)
print(scores[y_1_3])

z_1_3 = np.argmin(d13[start_2:])
print(z_1_3 + 480)
print(scores[z_1_3 + 480])

print('\n')

d_2_1 = np.zeros([240])
d_2_3 = np.zeros([240])
for i in range(start_1, start_2): 
	d_2_1[i - start_1] = np.sum(distances[i, :start_1])
	d_2_3[i - start_1] = np.sum(distances[i, start_2:])

x_2_1 = np.argmin(d_2_1) 
print(x_2_1 + 240)
print(scores[x_2_1 + 240])

d21 = distances[np.argmin(d_2_1)]
y_2_1 = np.argmax(d21[start_1:start_2])
print(y_2_1 + 240)
print(scores[y_2_1 + 240])

z_2_1 = np.argmin(d21[:start_1])
print(z_2_1)
print(scores[z_2_1])

print('\n')

x_2_3 = np.argmin(d_2_3)
print(x_2_3 + 240)
print(scores[x_2_3 + 240])

d23 = distances[np.argmin(d_2_3)] 
y_2_3 = np.argmax(d23[start_1:start_2])
print(y_2_3 + 240)
print(scores[y_2_3 + 240])

z_2_3 = np.argmin(d23[start_2:])
print(z_2_3 + 480)
print(scores[z_2_3 + 480])

print('\n')


d_3_1 = np.zeros([240])
d_3_2 = np.zeros([240])
for i in range(start_1, start_2): 
	d_3_1[i - start_1] = np.sum(distances[i, :start_1])
	d_3_2[i - start_1] = np.sum(distances[i, start_2:])

x_3_1 = np.argmin(d_3_1) 
print(x_3_1 + 480)
print(scores[x_3_1 + 480])

d31 = distances[np.argmin(d_3_1)]
y_3_1 = np.argmax(d31[start_2:])
print(y_3_1 + 480)
print(scores[y_3_1 + 480])

z_3_1 = np.argmin(d31[:start_1])
print(z_3_1)
print(scores[z_3_1])

print('\n')

x_3_2 = np.argmin(d_3_2)
print(x_3_2 + 480)
print(scores[x_3_2 + 480])

d32 = distances[np.argmin(d_3_2)] 
y_3_2 = np.argmax(d32[start_2:])
print(y_3_2 + 480)
print(scores[y_3_2 + 480])

z_3_2 = np.argmin(d32[start_1:start_2])
print(z_3_2 + 240)
print(scores[z_3_2 + 240])




# if __name__ == '__main__':
# 	main()