# Support Vector Machines

The Support Vector Machine (SVM) is yet another type of supervised machine learning algorithm. It is sometimes cleaner and more powerful.

Recall that in logistic regression, we use the following rules:

if y=1, then hθ(x) ≈ 1 and ΘTx≫0

if y=0, then hθ(x) ≈ 0 and ΘTx≪0

Recall the cost function for (unregularized) logistic regression:

![alt text](/Week_7/SupportVectorMachines/Assets/1.png)

To make a support vector machine, we will modify the first term of the cost function −log(hθ(x)) = −log(1 / 1+e−θTx) so that when θTx (from now on, we shall refer to this as z) is __greater than 1__, it outputs 0. Furthermore, for values of z less than 1, we shall use a straight decreasing line instead of the sigmoid curve.(In the literature, this is called a hinge loss (https://en.wikipedia.org/wiki/Hinge_loss) function.)

![alt text](/Week_7/SupportVectorMachines/Assets/2.png)

Similarly, we modify the second term of the cost function −log(1−hθ(x)) = −log(1−1/1+e−θTx) so that when z is __less than__ -1, it outputs 0. We also modify it so that for values of z greater than -1, we use a straight increasing line instead of the sigmoid curve.

![alt text](/Week_7/SupportVectorMachines/Assets/3.png)

We shall denote these as cost1(z) and cost0(z) (respectively, note that cost1(z) is the cost for classifying when y=1, and cost0(z) is the cost for classifying when y=0), and we may define them as follows (where k is an arbitrary constant defining the magnitude of the slope of the line):

z = θTx

cost0(z) = max(0,k(1+z))

cost1(z)=max(0,k(1−z))

Recall the full cost function from (regularized) logistic regression:

![alt text](/Week_7/SupportVectorMachines/Assets/4.png)

Note that the negative sign has been distributed into the sum in the above equation.

We may transform this into the cost function for support vector machines by substituting 
cost0(z) and cost1(z):

![alt text](/Week_7/SupportVectorMachines/Assets/5.png)

We can optimize this a bit by multiplying this by m (thus removing the m factor in the denominators). Note that this does not affect our optimization, since we're simply multiplying our cost function by a positive constant (for example, minimizing (u−5)+1 gives us 5; multiplying it by 10 to make it 10(u−5)^2 +10 still gives us 5 when minimized).

![alt text](/Week_7/SupportVectorMachines/Assets/6.png)

Furthermore, convention dictates that we regularize using a factor C, instead of λ, like so:

![alt text](/Week_7/SupportVectorMachines/Assets/7.png)

This is equivalent to multiplying the equation by C=1/λ, and thus results in the same values when optimized. Now, when we wish to regularize more (that is, reduce overfitting), we decrease C, and when we wish to regularize less (that is, reduce underfitting), we increase C.

Finally, note that the hypothesis of the Support Vector Machine is not interpreted as the probability of y being 1 or 0 (as it is for the hypothesis of logistic regression). Instead, it outputs either 1 or 0. (In technical terms, it is a discriminant function.)

![alt text](/Week_7/SupportVectorMachines/Assets/8.png)

# Large Margin Intuition

A useful way to think about Support Vector Machines is to think of them as Large Margin Classifiers.

If y=1, we want ΘTx≥1 (not just ≥0)

If y=0, we want ΘTx≤−1 (not just <0)

Now when we set our constant C to a very __large__ value (e.g. 100,000), our optimizing function will constrain Θ such that the equation A (the summation of the cost of each example) equals 0. We impose the following constraints on Θ:

ΘTx≥1 if y=1 and ΘTx≤−1 if y=0.

If C is very large, we must choose Θ parameters such that:

∑ i=1m y(i)cost 1(Θ Tx)+(1−y (i))cost 0(ΘTx)=0

This reduces our cost function to:

![alt text](/Week_7/SupportVectorMachines/Assets/9.png)

Recall the decision boundary from logistic regression (the line separating the positive and negative examples). In SVMs, the decision boundary has the special property that it is __as far away as possible__ from both the positive and the negative examples.

The distance of the decision boundary to the nearest example is called the __margin__. Since SVMs maximize this margin, it is often called a Large Margin Classifier.

The SVM will separate the negative and positive examples by a __large margin__.

This large margin is only achieved when __C is very large__.

Data is __linearly separable__ when a __straight line__ can separate the positive and negative examples.

If we have __outlier__ examples that we don't want to affect the decision boundary, then we can __reduce C__.

Increasing and decreasing C is similar to respectively decreasing and increasing λ, and can simplify our decision boundary.

# Mathematics Behing Large Margin Classification

Say we have two vectors, u and v:

u=[u1;u2]

v=[v1;v2]

The length of vector v is denoted ∣∣v∣∣, and it describes the line on a graph from origin (0,0) to (v1,v2). The length of vector v can be calculated with sqrt(v12+v22) by the Pythagorean theorem.

The projection of vector v onto vector u is found by taking a right angle from u to the end of v, creating a right triangle.

* p = length of projection of v onto the vector u.

* uTv =p⋅∣∣u∣∣

Note that uTv = ∣∣u∣∣⋅∣∣v∣∣cosθ where θ is the angle between u and v. Also, p = ∣∣v∣∣ cosθ. If you substitute p for ∣∣v∣∣cosθ, you get uTv = p⋅∣∣u∣|.

So the product uTv is equal to the length of the projection times the length of vector u.

In our example, since u and v are vectors of the same length, uTv = vTu.

uTv = vTu = p⋅∣∣u∣∣ = u1v1 + u2v2

If the __angle__ between the lines for v and u is __greater than 90 degrees__, then the projection p will be __negative__.

![alt text](/Week_7/SupportVectorMachines/Assets/10.png)

We can use the same rules to rewrite ΘTx(i):

ΘTx(i) = p(i)⋅∣∣Θ∣∣ = Θ1x1(i) + Θ2x2(i) + ⋯ + Θnxn(i)

So we now have a new __optimization objective__ by substituting p(i) ⋅∣∣Θ∣∣ in for ΘTx(i) :

If y=1, we want p(i)⋅∣∣Θ∣∣ ≥ 1

If y=0, we want p(i)⋅∣∣Θ∣∣ ≤ −1

The reason this causes a "large margin" is because: the vector for Θ is perpendicular to the decision boundary. In order for our optimization objective (above) to hold true, we need the absolute value of our projections p(i) to be as large as possible.

If Θ0 = 0, then all our decision boundaries will intersect (0,0). If Θ0 ≠ 0, the support vector machine will still find a large margin for the decision boundary.

# Kernels I

__Kernels__ allow us to make complex, non-linear classifiers using Support Vector Machines.

Given x, compute new feature depending on proximity to landmarks l(1), l(2, l(3).

To do this, we find the "similarity" of x and some landmark l(i):

fi = similarity(x,l(i)) = exp(− ∣∣x−l (i)∣∣^2) / 2σ^2)

This "similarity" function is called a __Gaussian Kernel__. It is a specific example of a kernel.

The similarity function can also be written as follows:

fi = similarity(x,l(i)) = exp(−∑j=1n(x j−lj(i))2 / 2σ^2)
 
There are a couple properties of the similarity function:

If x ≈ l(i), then fi = exp(− ≈0^2/2σ2) ≈ 1

If x is far from l(i), then f(i) = exp(−(large number)^2 / 2σ^2) ≈ 0

In other words, if x and the landmark are close, then the similarity will be close to 1, and if x and the landmark are far away from each other, the similarity will be close to 0.

Each landmark gives us the features in our hypothesis:

![alt text](/Week_7/SupportVectorMachines/Assets/11.png)

σ2 is a parameter of the Gaussian Kernel, and it can be modified to increase or decrease the __drop-off__ of our feature fi. Combined with looking at the values inside Θ, we can choose these landmarks to get the general shape of the decision boundary.

# Kernel II

One way to get the landmarks is to put them in the __exact same__ locations as all the training examples. This gives us m landmarks, with one landmark per training example.

Given example x:

f1 = similarity(x,l(1)), f2 = similarity(x,l(2)), f3 = similarity(x,l(3)), and so on.

This gives us a "feature vector," f(i) of all our features for example x(i). We may also set f0 = 1 to correspond with Θ0. Thus given training example x(i):

![alt text](/Week_7/SupportVectorMachines/Assets/12.png)

Now to get the parameters Θ we can use the SVM minimization algorithm but with f(i)substituted in for x(i):

minΘ C∑i=1m y(i)cost 1(Θ Tf(i))+(1−y(i))cost0(θ Tf(i))+ 1/2∑j=1n Θj2

Using kernels to generate f(i) is not exclusive to SVMs and may also be applied to logistic regression. However, because of computational optimizations on SVMs, kernels combined with SVMs is much faster than with other algorithms, so kernels are almost always found combined only with SVMs.
​
## Choosing SVM Parameters

Choosing C (recall that C = 1/λ

* If C is large, then we get higher variance/lower bias
* If C is small, then we get lower variance/higher bias

The other parameter we must choose is σ2 from the Gaussian Kernel function:

With a large σ2, the features fi vary more smoothly, causing higher bias and lower variance.

With a small σ2, the features fi vary less smoothly, causing lower bias and higher variance.

## Using An SVM

There are lots of good SVM libraries already written. A. Ng often uses 'liblinear' and 'libsvm'. In practical application, you should use one of these libraries rather than rewrite the functions.

In practical application, the choices you do need to make are:

* Choice of parameter C
* Choice of kernel (similarity function)
* No kernel ("linear" kernel) -- gives standard linear classifier
* Choose when n is large and when m is small
* Gaussian Kernel (above) -- need to choose σ2 
* Choose when n is small and m is large

The library may ask you to provide the kernel function.

__Note__: do perform feature scaling before using the Gaussian Kernel.

__Note__: not all similarity functions are valid kernels. They must satisfy "Mercer's Theorem" which guarantees that the SVM package's optimizations run correctly and do not diverge.

You want to train C and the parameters for the kernel function using the training and cross-validation datasets.

## Multi-class Classification

Many SVM libraries have multi-class classification built-in.

You can use the one-vs-all method just like we did for logistic regression, where y ∈ 1,2,3,…,K with Θ(1),Θ(2),…,Θ(K). We pick class i with the largest Θ(i)Tx.

## Logistic Regression vs. SVMs

If n is large (relative to m), then use logistic regression, or SVM without a kernel (the "linear kernel")

If n is small and m is intermediate, then use SVM with a Gaussian Kernel

If n is small and m is large, then manually create/add more features, then use logistic regression or SVM without a kernel.

In the first case, we don't have enough examples to need a complicated polynomial hypothesis. In the second example, we have enough examples that we may need a complex non-linear hypothesis. In the last case, we want to increase our features so that logistic regression becomes applicable.

__Note__: a neural network is likely to work well for any of these situations, but may be slower to train.

#### Additional references

"An Idiot's Guide to Support Vector Machines": http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf