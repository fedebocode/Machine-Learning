### Multiple Features

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.


* _x(i)j_ = value of feature _j_ in the _ith_ training example
* _x(i)_ = the input (features) of the _ith_ training example
* _m_ = the number of training examples
* _n_ = the number of features

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

hθ(x)= θ0 + θ1x1 + θ2x2 + θ3x3 + ... + θnxn

In order to develop intuition about this function, we can think about θ0 as the basic price of a house, θ1 as the price per square meter, θ2 as the price per floor, etc. x1 will be the number of square meters in the house, x2 the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

![alt text](/Week_2/MultiVariateLinearRegression/Assets/1.png)

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume _x0(i) = 1 for (i∈1,…,m)_. This allows us to do matrix operations with theta and x. Hence making the two vectors _'θ'_ and _x(i)_ match each other element-wise (that is, have the same number of elements: n+1).

### Gradient Descent For Multiple Variables

he gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

![alt text](/Week_2/MultiVariateLinearRegression/Assets/2.png)

In other words:

![alt text](/Week_2/MultiVariateLinearRegression/Assets/3.png)

The following image compares gradient descent with one variable to gradient descent with multiple variables:

![alt text](/Week_2/MultiVariateLinearRegression/Assets/4.png)

### Gradient Descent in Practice I - Feature Scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

−1 ≤ _x(i)_ ≤ 1

or

−0.5 ≤ _x(i)_ ≤ 0.5

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are __feature scaling__ and __mean normalization__. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

_xi := (xi − μi) / si_
​

Where _μi_ is the average of all the values for feature (i) and _si_ is the range of values (max - min), or _si_ is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results.
​	 
For example, if _xi_ represents housing prices with a range of 100 to 2000 and a mean value of 1000, then, 

_xi := (price − 1000) / 1900_

Example:

![alt text](/Week_2/MultiVariateLinearRegression/Assets/Example.png)

So for any individual feature f:

	f_norm = (f - f_mean) / (f_max - f_min)

	e.g. for x2,(midterm exam)^2 = {7921, 5184, 8836, 4761}

	> x2 <- c(7921, 5184, 8836, 4761)
	> mean(x2)
 	6676
	> max(x2) - min(x2)
 	4075
	>(x2 - mean(x2)) / (max(x2) - min(x2))
 	0.306  -0.366  0.530 -0.470

 Hence __norm(5184) = 0.366__

### Gradient Descent in Practice II - Learning Rate

Debugging gradient descent. Make a plot with number of iterations on the x-axis. Now plot the cost function, _J(θ)_ over the number of iterations of gradient descent. If _J(θ)_ ever increases, then you probably need to decrease _α_.

Automatic convergence test. Declare convergence if _J(θ)_ decreases by less than _E_ in one iteration, where _E_ is some small value such as 10−3. However in practice it's difficult to choose this threshold value.

It has been proven that if learning rate _α_ is sufficiently small, then _J(θ)_ will decrease on every iteration.

![alt text](/Week_2/MultiVariateLinearRegression/Assets/5.png)

To summarize:

 * If _α_ is too small: slow convergence.

 * If _α_ is too large: may not decrease on every iteration and thus may not converge.

### Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can combine multiple features into one. For example, we can combine _x1_ and _x2_ into a new feature _x3_ by taking _x1 * x2_.

#### Polynomial Regression

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is _hθ(x)= θ0 + θ1x1_ then we can create additional features based on x1, to get the quadratic function _hθ(x)= θ0 + θ1x1+θ2x1^2_ or the cubic function _hθ(x)= θ0 + θ1x1 + θ2x1^2 + θ3x1^3_.
In the cubic version, we have created new features _x2_ and _x3_ where _x2 = x1^2 and x3 = x1^3_.

To make it a square root function, we could do: 

_hθ(x) = θ0 + θ1x1 + θ2root(x1)_ 	 

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.