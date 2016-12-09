# MachineTeaching 

Our project was to conduct the interactive multi-class machine teaching of images through triplets. We first collected data from the crowd, where we showed users triplets of images and had users identify which images were similar. For each dataset, we collected ~6000 triplets using Amazon Mturk. Then, we used the triplets to create a t-Stochastic Triplet Embedding (tSTE) for the images.

The next step after obtaining the embedding was to actually teach through triplets. We noted the fact that the tSTE gave us a probabilisitc model representing the probability of each triplet under the embedding (p_ijk). We further realized that instead of optimizing over the kernel, we could directly optimize the tSTE objective function in an online manner through stochastic gradient descent. This led to the development of four different selection strategies for determining the next triplet to show to users:

1) Random- Simply show a random triplet to the user.

2) Most Uncertainty- Show the triplet with current probability closest to 0.5.

3) Best Gradient Increase- Take a gradient step assuming the user identifies the triplet correctly. Next, take a gradient step assuming the user identifies the triplet incorrectly. Let p be the probability of the triplet before the gradient steps, and let p1, p2 be the probabilities after gradient step assuming the user gets the triplet correct/incorrect. So, select the triplet that maximizes p * p1 + (1-p) * p2 - p.

4) Best Gradient Increase Random Sample- Similar the 3, except recognize the fact that a gradient step affects not only the current triplet, but also O(n^2) other triplets. So, for each triplet randomly sample from all other triplets that are affected by the gradient step and average across these, selecting the triplet the maximizes p * p1_avg + (1-p) * p2_avg - p_avg.

For strategies 2-4, it is too computationaly intensive to evalute all possible triplets in the online scenerio. So, for each strategy we randomly sampled and scored as many triplets as possible given the online setting (<1.5s time limit).

We launched a session on Amazon Mechanical Turk to evaluate our aglorithms. The hit consisted of a teaching phase and a testing phase. In the teaching phase, we randomly assigned each user to a selection strategy, and presented a series of triplets using that strategy. We displayed both the triplet and the class labels, and told users whether they identified the triplet correctly/incorrectly. In the testing phase, we only showed single images to the users and asked users to select which class the image belongs to. 

The test results of the different selection strategies for the China dataset can be found in china_dataset_results.png. We will follow up with results on the Seabed dataset as well.

In order to create the teaching interface, we used a Flask & SqlAlchemy backend hosted on Pythonanywhere along with a Javascript front end. We used "sessions" to personlize the teaching for each user. Note that since we designed the ML algorithms to take ~1.5s per click. So, in order to make our application responsive for real-time users, we had to scale our solution with multiple processors and randomized load-balancing to handle the concurrent load from Amazon MTURK. 

