# K-means Clustering

The K-Means Algorithm is the most popular and widely used algorithm for automatically grouping data into coherent subsets.

1. Randomly initialize two points in the dataset called the cluster centroids.
2. Cluster assignment: assign all examples into one of two groups based on which cluster centroid the example is closest to.
3. Move centroid: compute the averages for all the points inside each of the two cluster centroid groups, then move the cluster centroid points to those averages.
4. Re-run (2) and (3) until we have found our clusters.

Our main variables are:

* K (number of clusters)
* Training set x(1),x(2),…,x(m)
* Where x(i)∈ Rn

Note that we __will not use__ the x0=1 convention.

__The algorithm:__

	Randomly initialize K cluster centroids mu(1), mu(2), ..., mu(K)
	Repeat:
   	for i = 1 to m:
      	c(i):= index (from 1 to K) of cluster centroid closest to x(i)
   	for k = 1 to K:
      	mu(k):= average (mean) of points assigned to cluster k

The __first for-loop__ is the 'Cluster Assignment' step. We make a vector c where c(i) represents the centroid assigned to example x(i).

We can write the operation of the Cluster Assignment step more mathematically as follows:

c(i) = argmink ∣∣x(i) − μk∣∣^2

That is, each c(i) contains the index of the centroid that has minimal distance to x(i).

By convention, we square the right-hand-side, which makes the function we are trying to minimize more sharply increasing. It is mostly just a convention. But a convention that helps reduce the computation load because the Euclidean distance requires a square root but it is canceled.

Without the square:

![alt text](/Week_8/Clustering/Assets/1.png)

With the square:

![alt text](/Week_8/Clustering/Assets/2.png)

...so the square convention serves two purposes, minimize more sharply and less computation.

The __second for-loop__ is the 'Move Centroid' step where we move each centroid to the average of its group.

More formally, the equation for this loop is as follows:

μk = n1[x(k1) + x(k2) +⋯+x(kn)] ∈ Rn

Where each of x(k1), x(k2), ⋯ x(kn) are the training examples assigned to group mμk.

If you have a cluster centroid with __0 points__ assigned to it, you can randomly __re-initialize__ that centroid to a new point. You can also simply __eliminate__ that cluster group.

After a number of iterations the algorithm will __converge__, where new iterations do not affect the clusters.

Note on non-separated clusters: some datasets have no real inner separation or natural structure. K-means can still evenly segment your data into K subsets, so can still be useful in this case.

# Optimization Objective

Recall some of the parameters we used in our algorithm:

* c(i) = index of cluster (1,2,...,K) to which example x(i) is currently assigned
* μk = cluster centroid k (μk∈ℝn)
* μc(i) = cluster centroid of cluster to which example x(i) has been assigned

Using these variables we can define our __cost function__:

![alt text](/Week_8/Clustering/Assets/3.png)

Our __optimization objective__ is to minimize all our parameters using the above cost function:

_minc,μ J(c,μ)_

That is, we are finding all the values in sets c, representing all our clusters, and μ, representing all our centroids, that will minimize the __average of the distances__ of every training example to its corresponding cluster centroid.

The above cost function is often called the __distortion__ of the training examples.

In the __cluster assignment step__, our goal is to:

Minimize J(…) with c(1),…,c(m) (holding μ1,…,μK fixed)

In the __move centroid step__, our goal is to:

Minimize J(…) with μ1,…,μK
​	 
With k-means, it is __not possible for the cost function to sometimes increase__. It should always descend.

# Random Initialization

There's one particular recommended method for randomly initializing your cluster centroids.

1. Have K<m. That is, make sure the number of your clusters is less than the number of your training examples.

2. Randomly pick K training examples. (Not mentioned in the lecture, but also be sure the selected examples are unique).

3. Set μ1,…,μK equal to these K examples.

K-means __can get stuck in local optima__. To decrease the chance of this happening, you can run the algorithm on many different random initializations. In cases where K<10 it is strongly recommended to run a loop of random initializations.

	for i = 1 to 100:
   		randomly initialize k-means
   		run k-means to get 'c' and 'm'
   		compute the cost function (distortion) J(c,m)
	pick the clustering that gave us the lowest cost

# Choosing the Number of Clusters

Choosing K can be quite arbitrary and ambiguous.

__The elbow method__: plot the cost J and the number of clusters K. The cost function should reduce as we increase the number of clusters, and then flatten out. Choose K at the point where the cost function starts to flatten out.

However, fairly often, the curve is _very gradual__, so there's no clear elbow.

__Note__: J will __always__ decrease as K is increased. The one exception is if k-means gets stuck at a bad local optimum.

Another way to choose K is to observe how well k-means performs on a __downstream purpose__. In other words, you choose K that proves to be most useful for some goal you're trying to achieve from using these clusters.

# Discussion of the drawbacks of K-Means

This links to a discussion that shows various situations in which K-means gives totally correct but unexpected results: http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means