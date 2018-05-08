### Normal Equation

Gradient descent gives one way of minimizing _J_. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize _J_ by explicitly taking its derivatives with respect to the _θj’s_, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below:

![alt text](/Week_2/ComputingParametersAnalytically/Assets/1.png)

![alt text](/Week_2/ComputingParametersAnalytically/Assets/2.png)

There is __no need__ to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

![alt text](/Week_2/ComputingParametersAnalytically/Assets/3.png)

With the normal equation, computing the inversion has complexity 
_O(n3)_. So if we have a very large number of features, the normal equation will be slow. In practice, when _n_ exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

### Normal Equation Noninvertibility

When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of _θ_ even if _XTX_ is not invertible.

If _XTX_ is __non invertible__, the common causes might be having :

* Redundant features, where two features are very closely related (i.e. they are linearly dependent)

* Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.
