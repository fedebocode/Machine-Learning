# Solving the Problem of Overfitting 

Consider the problem of predicting y from x ∈ R. The leftmost figure below shows the result of fitting a _y = θ0+θ1_
to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good.

![alt text](/Week_3/OverfittingProblem/Assets/1.png)

Instead, if we had added an extra feature _x2_, and fit _y = θ0+θ1x+θ2x^2_, then we obtain a slightly better fit to the data (See middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5th order polynomial _y = ∑j = 05θjx^j_. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices _(y)_ for different living areas _(x)_. Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of underfitting—in which the data clearly shows structure not captured by the model—and the figure on the right is an example of overfitting.

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features:

* Manually select which features to keep.
* Use a model selection algorithm (studied later in the course).

2) Regularization

* Keep all the features, but reduce the magnitude of parameters _θj_.
* Regularization works well when we have a lot of slightly useful features.

# Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:

_θ0+θ1x+θ2x^2+θ3x^3+θ4x^4_

We'll want to eliminate the influence of _θ3x3θ^3_ and _θ4x4θ^4_. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our __cost function__:

![alt text](/Week_3/OverfittingProblem/Assets/2.png)	 

We've added two extra terms at the end to inflate the cost of θ3 and θ4. Now, in order for the cost function to get close to zero, we will have to reduce the values of θ3 and θ4 to near zero. This will in turn greatly reduce the values of θ3x3θ3 and θ4x4θ4 in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms θ3x3θ3 and θ4x4θ4.

We could also regularize all of our theta parameters in a single summation as:

![alt text](/Week_3/OverfittingProblem/Assets/3.png)

The λ, or lambda, is the __regularization parameter__. It determines how much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if λ = 0 or is too small ?

![alt text](/Week_3/OverfittingProblem/Assets/4.png)

# Regularized Linear Regression

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

__Gradient Descent__

We will modify our gradient descent function to separate out _θ0_ from the rest of the parameters because we do not want to penalize _θ0_.

![alt text](/Week_3/OverfittingProblem/Assets/5.png)

The term _(λ\m) * θj_ performs our regularization. With some manipulation our update rule can also be represented as:

![alt text](/Week_3/OverfittingProblem/Assets/6.png)

The first term in the above equation, 1 − α * (λ\m) will always be less than 1. Intuitively you can see it as reducing the value of _θj_ by some amount on every update. Notice that the second term is now exactly the same as it was before.

__Normal Equation__

Now let's approach regularization using the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

![alt text](/Week_3/OverfittingProblem/Assets/7.png)

_L_ is a matrix with _0_ at the top left and 1's down the diagonal, with _0's_ everywhere else. It should have dimension _(n+1)×(n+1)_. Intuitively, this is the identity matrix (though we are not including _x0_), multiplied with a single real number λ.

Recall that if _m < n_, then _X'T_ is non-invertible. However, when we add the term _λ⋅L_, then _X'X + λ⋅L_ becomes invertible.

# Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:

![alt text](/Week_3/OverfittingProblem/Assets/8.png)

__Cost Function__

Recall that our cost function for logistic regression was:

![alt text](/Week_3/OverfittingProblem/Assets/9.png)

We can regularize this equation by adding a term to the end:

![alt text](/Week_3/OverfittingProblem/Assets/10.png)

The second sum, means to explicitly exclude the bias term, _θ0_. I.e. the _θ_ vector is indexed from _0_ to _n_ (holding n+1 values, _θ0_ through _θn_), and this sum explicitly skips _θ0_ , by running from 1 to _n_, skipping _0_. Thus, when computing the equation, we should continuously update the two following equations:

![alt text](/Week_3/OverfittingProblem/Assets/11.png)