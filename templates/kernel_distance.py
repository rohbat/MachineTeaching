'''
centrality

closest to self cluster
farthest from self cluster

closest to other clusters
farthest from other clusters

accuracy during kernel collection
'''

distance = np.zeros(len(image_list), len(image_list))

for i in range(len(image_list)): 
	for j in range(len(image_list)): 
		distance[i][j] = np.linalg.norm(image_list(i) - image_list(j))

class1 = 239
class2 = 0
class3 = 0



