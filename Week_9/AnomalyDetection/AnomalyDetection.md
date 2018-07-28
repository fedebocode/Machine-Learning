# Anomaly Detection

Problem Motivation

Just like in other learning problems, we are given a dataset x(1),x(2),…,x(m).

We are then given a new example, xtest, and we want to know whether this new example is abnormal/anomalous.

We define a "model" p(x) that tells us the probability the example is not anomalous. We also use a threshold ϵ (epsilon) as a dividing line so we can say which examples are anomalous and which are not.

A very common application of anomaly detection is detecting fraud:

* x(i) = features of user i's activities

* Model p(x) from the data.

* Identify unusual users by checking which have p(x)<ϵ.

If our anomaly detector is flagging __too many__ anomalous examples, then we need to __decrease__ our threshold ϵ

# Gaussian Distribution

The Gaussian Distribution is a familiar bell-shaped curve that can be described by a function N(μ,σ2)

Let x ∈ ℝ. If the probability distribution of x is Gaussian with mean μ, variance σ2, then:

_x ∼ N(μ,σ2)_

The little ∼ or 'tilde' can be read as "distributed as."

The Gaussian Distribution is parameterized by a mean and a variance.

Mu, or μ, describes the center of the curve, called the mean. The width of the curve is described by sigma, or σ, called the standard deviation.

The full function is as follows:

![alt text](/Week_9/AnomalyDetection/Assets/1.png)

We can estimate the parameter μ from a given dataset by simply taking the average of all the examples:

![alt text](/Week_9/AnomalyDetection/Assets/2.png)

We can estimate the other parameter, σ2, with our familiar squared error formula:

![alt text](/Week_9/AnomalyDetection/Assets/3.png)

# Algorithm

Given a training set of examples, {x(1),…,x(m)} where each example is a vector, x ∈ Rn.

![alt text](/Week_9/AnomalyDetection/Assets/4.png)

In statistics, this is called an "independence assumption" on the values of the features inside training example x.

More compactly, the above expression can be written as follows:

![alt text](/Week_9/AnomalyDetection/Assets/5.png)

__The algorithm__

Choose features _xi_ that you think might be indicative of anomalous examples.

Fit parameters _μ1, … ,μn,σ^21, … ,σ^2n_

Calculate: ![alt text](/Week_9/AnomalyDetection/Assets/6.png)

Calculate: ![alt text](/Week_9/AnomalyDetection/Assets/7.png)

Given a new example x, compute p(x):

![alt text](/Week_9/AnomalyDetection/Assets/8.png)

Anomaly if p(x)<ϵ

A vectorized version of the calculation for μ is μ = 1m∑i=1m x(i). You can vectorize σ2 similarly.

# Developing and Evaluating an Anomaly Detection System

To evaluate our learning algorithm, we take some labeled data, categorized into anomalous and non-anomalous examples ( y = 0 if normal, y = 1 if anomalous).

Among that data, take a large proportion of __good__, non-anomalous data for the training set on which to train p(x).

Then, take a smaller proportion of mixed anomalous and non-anomalous examples (you will usually have many more non-anomalous examples) for your cross-validation and test sets.

For example, we may have a set where 0.2% of the data is anomalous. We take 60% of those examples, all of which are good (y=0) for the training set. We then take 20% of the examples for the cross-validation set (with 0.1% of the anomalous examples) and another 20% from the test set (with another 0.1% of the anomalous).

In other words, we split the data 60/20/20 training/CV/test and then split the anomalous examples 50/50 between the CV and test sets.

__Algorithm evaluation:__

Fit model p(x) on training set {x(1),…,x(m)}

On a cross validation/test example x, predict:

If p(x) < ϵ (__anomaly__), then y=1

If p(x) ≥ ϵ (__normal__), then y=0

Possible evaluation metrics (see "Machine Learning System Design" section):

* True positive, false positive, false negative, true negative.

* Precision/recall

* F1 score

Note that we use the cross-validation set to choose parameter ϵ

# Anomaly Detection vs. Supervised Learning

When do we use anomaly detection and when do we use supervised learning?

Use anomaly detection when...

* We have a very small number of positive examples (y=1 ... 0-20 examples is common) and a large number of negative (y=0) examples.

* We have many different "types" of anomalies and it is hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous examples we've seen so far.

Use supervised learning when...

* We have a large number of both positive and negative examples. In other words, the training set is more evenly divided into classes.

* We have enough positive examples for the algorithm to get a sense of what new positives examples look like. The future positive examples are likely similar to the ones in the training set.

# Choosing What Features to Use

The features will greatly affect how well your anomaly detection algorithm works.

We can check that our features are __gaussian__ by plotting a histogram of our data and checking for the bell-shaped curve.

Some __transforms__ we can try on an example feature x that does not have the bell-shaped curve are:

* log(x)
* log(x+1)
* log(x+c) for some constant
* sqrt(x)
* x^(1/3)
 
We can play with each of these to try and achieve the gaussian shape in our data.

There is an __error analysis procedure__ for anomaly detection that is very similar to the one in supervised learning.

Our goal is for p(x) to be large for normal examples and small for anomalous examples.

One common problem is when p(x) is similar for both types of examples. In this case, you need to examine the anomalous examples that are giving high probability in detail and try to figure out new features that will better distinguish the data.

In general, choose features that might take on unusually large or small values in the event of an anomaly.

# Multivariate Gaussian Distribution

The multivariate gaussian distribution is an extension of anomaly detection and may (or may not) catch more anomalies.

Instead of modeling p(x1),p(x2),… separately, we will model p(x) all in one go. Our parameters will be: μ ∈ Rn and Σ ∈ Rn×n

![alt text](/Week_9/AnomalyDetection/Assets/9.png)

The important effect is that we can model oblong gaussian contours, allowing us to better fit data that might not fit into the normal circular contours.

Varying Σ changes the shape, width, and orientation of the contours. Changing μ will move the center of the distribution.

Check also:

* The Multivariate Gaussian Distribution http://cs229.stanford.edu/section/gaussians.pdf Chuong B. Do, October 10, 2008.

# Anomaly Detection using the Multivariate Gaussian Distribution

When doing anomaly detection with multivariate gaussian distribution, we compute μ and Σ normally. We then compute p(x) using the new formula in the previous section and flag an anomaly if p(x) < ϵ.

The original model for p(x) corresponds to a multivariate Gaussian where the contours of p(x;μ,Σ) are axis-aligned.

The multivariate Gaussian model can automatically capture correlations between different features of x.

However, the original model maintains some advantages: it is computationally cheaper (no matrix to invert, which is costly for large number of features) and it performs well even with small training set size (in multivariate Gaussian model, it should be greater than the number of features for Σ to be invertible).