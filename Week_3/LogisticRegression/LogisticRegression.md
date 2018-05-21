# Logistic Regression Model

### Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:

![alt text](/Week_3/LogisticRegression/Assets/1.png)

When y = 1, we get the following plot for _J(θ)_ vs _hθ(x)_:

![alt text](/Week_3/LogisticRegression/Assets/2.png)

When y = 0, we get the following plot for _J(θ)_ vs _hθ(x)_:

![alt text](/Week_3/LogisticRegression/Assets/3.png)

![alt text](/Week_3/LogisticRegression/Assets/4.png)

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that _J(θ)_ is convex for logistic regression.

### Simplified Cost Function and Gradient Descent

We can compress our cost function's two conditional cases into one case:

___Cost(hθ(x),y) = −ylog(hθ(x)) − (1−y)log(1−hθ(x))___

Notice that when y is equal to 1, then the second term _(1−y)log(1−hθ(x))_ will be zero and will not affect the result. If y is equal to 0, then the first term _−ylog(hθ(x))_ will be zero and will not affect the result.

We can fully write out our entire cost function as follows:

![alt text](/Week_3/LogisticRegression/Assets/6.png)

A vectorized implementation is:

![alt text](/Week_3/LogisticRegression/Assets/7.png)

__Gradient Descent__

Remember that the general form of gradient descent is:

![alt text](/Week_3/LogisticRegression/Assets/8.png)

We can work out the derivative part using calculus to get:

![alt text](/Week_3/LogisticRegression/Assets/9.png)

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:

_θ:= θ − (α / m)* X'(g(Xθ)− y)_

Also:

![alt text](/Week_3/LogisticRegression/Assets/5.png)

# Advanced Optimization

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize _θ_ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value _θ_:

	* J(θ)
	* (∂ / ∂θj) * J(θ)

We can write a single function that returns both of these:

		function [jVal, gradient] = costFunction(theta)
  			jVal = [...code to compute J(theta)...];
  			gradient = [...code to compute derivative of J(theta)...];
		end

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".

	options = optimset('GradObj', 'on', 'MaxIter', 100);
	initialTheta = zeros(2,1);
   	[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

# Multiclass Classification: One-vs-all

Now we will approach the classification of data when we have more than two categories. Instead of _y = {0,1}_ we will expand our definition so that _y = {0,1...n}_.

Since _y = {0,1...n}_, we divide our problem into _n+1_ (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

![alt text](/Week_3/LogisticRegression/Assets/10.png)

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

The following image shows how one could classify 3 classes:

![alt text](/Week_3/LogisticRegression/Assets/11.png)

__To summarize__:

Train a logistic regression classifier hθ(x) for each class￼to predict the probability that y = i.

To make a prediction on a new x, pick the class that maximizes hθ(x).