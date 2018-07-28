# Recommender Systems

Problem Formulation

Recommendation is currently a very popular application of machine learning.

Say we are trying to recommend movies to customers. We can use the following definitions

* nu = number of users
* nm = number of movies
* r(i,j) = r(i,j)=1 if user j has rated movie i
* y(i,j)= rating given by user j to movie i (defined only if r(i,j)=1)

# Content Based Recommendations

We can introduce two features, _x1_ and _x2_ which represents how much romance or how much action a movie may have (on a scale of 0−1).

One approach is that we could do linear regression for every single user. For each user j, learn a parameter θ(j) ∈ R^3.

Predict user j as rating movie i with (θ(j))^Tx(i) stars.

* θ(j) = parameter vector for user j
* x(i) = feature vector for movie i

For user j, movie i, predicted rating: (θ(j))^T(x(i))

* m(j) = number of movies rated by user j

To learn θ(j), we do the following

![alt text](/Week_9/RecommenderSystems/Assets/1.png)

This is our familiar linear regression. The base of the first summation is choosing all i such that r(i,j) = 1.

To get the parameters for all our users, we do the following:

![alt text](/Week_9/RecommenderSystems/Assets/2.png)

We can apply our linear regression gradient descent update using the above cost function.

The only real difference is that we __eliminate the constant 1/m__.

# Collaborative Filtering

It can be very difficult to find features such as "amount of romance" or "amount of action" in a movie. To figure this out, we can use feature finders.

We can let the users tell us how much they like the different genres, providing their parameter vector immediately for us.

To infer the features from given parameters, we use the squared error function with regularization over all the users:

![alt text](/Week_9/RecommenderSystems/Assets/3.png)

You can also __randomly guess__ the values for theta to guess the features repeatedly. You will actually converge to a good set of features.

# Collaborative Filtering Algorithm

To speed things up, we can simultaneously minimize our features and our parameters:

![alt text](/Week_9/RecommenderSystems/Assets/4.png)

It looks very complicated, but we've only combined the cost function for theta and the cost function for x.

Because the algorithm can learn them itself, the bias units where x0=1 have been removed, therefore x∈ℝn and θ∈ℝn.

These are the steps in the algorithm:

1. Initialize x(i),...,x(nm),θ(1),...,θ(nu) to small random values. This serves to break symmetry and ensures that the algorithm x(i),...,x(nm) learns features that are different from each other.
2. Minimize J(x(i),...,x(nm),θ(1),...,θ(nu)) using gradient descent (or an advanced optimization algorithm).E.g. for every j = 1,...,nu,i=1,...nm:

![alt text](/Week_9/RecommenderSystems/Assets/5.png)

3. For a user with parameters θ and a movie with (learned) features x, predict a star rating of θ^Tx.

# Vectorization: Low Rank Matrix Factorization

Given matrices X (each row containing features of a particular movie) and Θ (each row containing the weights for those features for a given user), then the full matrix Y of all predicted ratings of all movies by all users is given simply by: Y = XΘ^T.

Predicting how similar two movies i and j are can be done using the distance between their respective feature vectors x. Specifically, we are looking for a small value of ∣∣x(i) − x(j)∣∣.

If the ranking system for movies is used from the previous lectures, then new users (who have watched no movies), will be assigned new movies incorrectly. Specifically, they will be assigned θ with all components equal to zero due to the minimization of the regularization term. That is, we assume that the new user will rank all movies 0, which does not seem intuitively correct.

We rectify this problem by normalizing the data relative to the mean. First, we use a matrix Y to store the data from previous ratings, where the ith row of Y is the ratings for the ith movie and the jth column corresponds to the ratings for the jth user.

We can now define a vector

_μ = [μ1,μ2,…,μnm]_

such that

![alt text](/Week_9/RecommenderSystems/Assets/6.png)

Which is effectively the mean of the previous ratings for the ith movie (where only movies that have been watched by users are counted). We now can normalize the data by subtracting u, the mean rating, from the actual ratings for each user (column in matrix Y):

As an example, consider the following matrix Y and mean ratings μ:

![alt text](/Week_9/RecommenderSystems/Assets/7.png)

The resulting Y′ vector is:

![alt text](/Week_9/RecommenderSystems/Assets/8.png)

Now we must slightly modify the linear regression prediction to include the mean normalization term:

_(θ(j))^Tx(i) + μi_

Now, for a new user, the initial predicted values will be equal to the μ term instead of simply being initialized to zero, which is more accurate.