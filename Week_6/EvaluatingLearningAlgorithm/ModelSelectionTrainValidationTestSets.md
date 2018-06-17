# Model Selection and Train/Validation/Test Sets

Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. It could over fit and as a result your predictions on the test set would be poor. The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set.

Given many models with different polynomial degrees, we can use a systematic approach to identify the 'best' function. In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

One way to break down our dataset into the three sets is:

Training set: 60%
Cross validation set: 20%
Test set: 20%

We can now calculate three separate error values for the three different sets using the following method:

* Optimize the parameters in Θ using the training set for each polynomial degree.
* Find the polynomial degree d with the least error using the cross validation set.
* Estimate the generalization error using the test set with _Jtest(Θ(d))_, (d = theta from polynomial with lower error);

This way, the degree of the polynomial _d_ has not been trained using the test set.

# Diagnosis Bias vs. Variance

In this section we examine the relationship between the degree of the polynomial _d_ and the underfitting or overfitting of our hypothesis.

We need to distinguish whether bias or variance is the problem contributing to bad predictions.
High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.
The training error will tend to decrease as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to decrease as we increase d up to a point, and then it will increase as d is increased, forming a convex curve.

High bias (underfitting): both _Jtrain(Θ)_ and _JCV(Θ)_ will be high. Also, _JCV(Θ) ≈ Jtrain(Θ)_.

High variance (overfitting): _Jtrain(Θ)_ will be low and _JCV(Θ)_ will be much greater than _Jtrain(Θ)_.

The is summarized in the figure below:

![alt text](/Week_6/EvaluatingLearningAlgorithm/Assets/1.png)

# Regularization and Bias/Variance

![alt text](/Week_6/EvaluatingLearningAlgorithm/Assets/2.png)

In the figure above, we see that as λ increases, our fit becomes more rigid. On the other hand, as λ approaches 0, we tend to over overfit the data. So how do we choose our parameter λ to get it 'just right' ? In order to choose the model and the regularization term λ, we need to:

* Create a list of lambdas (i.e. λ∈{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
* Create a set of models with different degrees or any other variants.
* Iterate through the λs and for each λ go through all the models to learn some Θ.
* Compute the cross validation error using the learned Θ (computed with λ) on the JCV(Θ) without regularization or λ = 0.
* Select the best combo that produces the lowest error on the cross validation set.
* Using the best combo Θ and λ, apply it on Jtest(Θ) to see if it has a good generalization of the problem.

# Learning Curves

Training an algorithm on a very few number of data points (such as 1, 2 or 3) will easily have 0 errors because we can always find a quadratic curve that touches exactly those number of points. Hence:

* As the training set gets larger, the error for a quadratic function increases.
* The error value will plateau out after a certain m, or training set size.

__Experiencing high bias:__

__Low training set size:__ causes _Jtrain(Θ)_ to be low and _JCV(Θ)_ to be high.

Large training set size: causes both _Jtrain(Θ)_ and _JCV(Θ)_ to be high with _Jtrain(Θ) ≈ JCV(Θ)_.

If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

![alt text](/Week_6/EvaluatingLearningAlgorithm/Assets/3.png)

__Experiencing high variance:__

__Low training set size:__ _Jtrain(Θ)_ will be low and _JCV(Θ)_ will be high.

__Large training set size:__ _Jtrain(Θ)_ increases with training set size and _JCV(Θ)_ continues to decrease without leveling off. 
Also, _Jtrain(Θ) < JCV(Θ)_ but the difference between them remains significant.

If a learning algorithm is suffering from high variance, getting more training data is likely to help.

![alt text](/Week_6/EvaluatingLearningAlgorithm/Assets/4.png)

# What's Next

Our decision process can be broken down as follows:

* __Getting more training examples:__ Fixes high variance

* __Trying smaller sets of features:__ Fixes high variance

* __Adding features:__ Fixes high bias

* __Adding polynomial features:__ Fixes high bias

* __Decreasing λ:__ Fixes high bias

* __Increasing λ:__ Fixes high variance.


__Diagnosing Neural Networks__

A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase λ) to address the overfitting.
Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

__Model Complexity Effects:__

* Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.

* Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.

*In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.
