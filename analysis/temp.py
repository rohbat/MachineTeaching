from __future__ import print_function
from __future__ import division
import math
import random
import numpy as np
import matplotlib.pyplot as plt


results = [False, True, True, False, True, False, True, True, True, False, True, True, True, True, True, True, True]


score = np.zeros(len(results))
print(score)

for i in range(len(results)): 
	if results[i] == True: 
		score[i:] += 1


fig = plt.figure()
plt.title('Progress Over Teaching Phase')
plt.xlabel('Teaching Iteration')
plt.ylabel('# Correct so far (out of 16)')
plt.plot(range(0, 17), score)
plt.show()



