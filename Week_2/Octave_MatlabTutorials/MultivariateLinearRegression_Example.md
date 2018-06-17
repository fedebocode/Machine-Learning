# Multivariate Linear Regression Example

In this exercise, you will implement linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.
The dataset contains a training set of housing prices. The first column is the size of the house, the second column is the number of bedrooms, and the third column is the price of the house.

Octave implementation:

Load Data:

	data = load('ex1data2.txt');
	X = data(:, 1:2);
	y = data(:, 3);
	m = length(y);

One of the column feature has values over 1000 times larger then the others; when features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly.

Scale features and set them to zero mean:

	[X mu sigma] = featureNormalize(X);

Add intercept term to X:

	X = [ones(m, 1) X];

First solve using gradient descent:

	alpha = 0.01;
	num_iters = 400;

Initiate Theta and Run Gradient Descent or compute Theta from the Normal Equation:

	theta = zeros(3, 1);

	[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
	or
	theta = normalEqn(X, y);

Predict house value for 1650sq/feet and 3 bedrooms:

	feature_Norm_1 = (1650 - mu(1,1)) / sigma(1,1);
	feature_Norm_2 = (3 - mu(1,2)) / sigma(1,2);
	
	predict1 = [1 feature_Norm_1 feature_Norm_2] * theta;

## Functions

__Feature Normalize__

	function [X_norm, mu, sigma] = featureNormalize(X)

		X_norm = X;
		mu = zeros(1, size(X, 2));
		sigma = zeros(1, size(X, 2));

		% Number of features
		N = size(X, 2);

		for i=1:N,
    		% Get ith feature/column
    		feature = X(:, i);                         		

    		% ith feature mean
    		mu(i) = mean(feature);  
    		% ith feature standard deviation                   		
    		sigma(i) = std(feature);

    		% replace normalized feature
    		X_norm(:, i) = (feature - mu(i)) / sigma(i);   
  			end
		end

__Normal Equation__

	function [theta] = normalEqn(X, y)

		theta = zeros(size(X, 2), 1);

		theta = pinv(X' * X) * X' * y;

	end

Plot the convergence graph:

	figure;
	plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
	xlabel('Number of iterations');
	ylabel('Cost J');

![alt text](/Week_2/Octave_MatlabTutorials/Assets/ConvergenceGraph.png)