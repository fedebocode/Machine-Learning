# Linear Regression Example

In this exercise, you will implement linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next.

The objective of linear regression is to minimize the cost function:

![alt text](/Week_2/Octave_MatlabTutorials/Assets/1.png)

where the hypothesis hθ(x) is given by the linear model:

![alt text](/Week_2/Octave_MatlabTutorials/Assets/2.png)

Recall that the parameters of your model are the θj values. These are the values you will adjust to minimize cost J(θ). One way to do this is to use the batch gradient descent algorithm. In batch gradient descent, each iteration performs the update

![alt text](/Week_2/Octave_MatlabTutorials/Assets/3.png)

With each step of gradient descent, your parameters θj come closer to the optimal values that will achieve the lowest cost J(θ).

### Solving Linear Regression

Assuming a dataset of scalar values as a matrix(n°Values,2):

	data = load('data1.txt');									// Load training set from txt file
	X = data(:, 1); 											// Build a matrix X of size n°Values x 1;
	y = data(:, 2);												// Build a matrix y of size n°Values x 1;
	m = length(y); 												// Number of training set    
	X = [ones(m, 1), data(:,1)]; 								// Add a column of ones to x to allow matrix multiplication
 
	n = size(X, 2); 											// The number of features 
	theta = zeros(n, 1); 										// Initialize theta as a n x 1 matrix with zeros
 
Gradient descent settings:

	iterations = 1500;											// Number of iterations to run gradient descent
	alpha = 0.01;												// Learning rate
 
Compute and display cost function for initial theta:

	computeCost(X, y, theta); 

Compute new theta using gradient descent:

	theta = gradientDescent(X, y, theta, alpha, iterations); 	// Theta contains value of theta0 and theta1
	
Predict values for new input data:

	predict1 = [1, 3.5] * theta; 

__Cost Function__

	function J = computeCost(X, y, theta)

		m = length(y); 

		h = X * theta;
		squaredErrors = (h - y).^2;

		J = 1/(2 * m) * sum(squaredErrors);

	end

__Gradient Descent__

	function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

	m = length(y);
	J_history = zeros(num_iters, 1);

	for iter = 1:num_iters
    
    	x = X(:,2);
    	h = theta(1) + (theta(2)*x);

    	theta_zero = theta(1) - alpha * (1/m) * sum(h - y);
    	theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x);

    	theta = [theta_zero; theta_one];
   
    	J_history(iter) = computeCost(X, y, theta);
    	disp(J_history(iter));

		end

	disp(min(J_history));

	end

### Plot the Cost Function J(theta0,theta1)

__Linear Fit:__	

	hold on; 												// Keep previous plot visible
	plot(X(:,2), X*theta, '-')
	legend('Training data', 'Linear regression')
	hold off 												// Don't overlay any more plots on this figure

![alt text](/Week_2/Octave_MatlabTutorials/Assets/LinearFit.png)

__Surface__:

	% Grid over which we will calculate J
	theta0_vals = linspace(-10, 10, 100);
	theta1_vals = linspace(-1, 4, 100);

	% initialize J_vals to a matrix of 0's
	J_vals = zeros(length(theta0_vals), length(theta1_vals));

	% Fill out J_vals
	for i = 1:length(theta0_vals)
    	for j = 1:length(theta1_vals)
	  	t = [theta0_vals(i); theta1_vals(j)];
	  	J_vals(i,j) = computeCost(X, y, t);
    	end
	end							

	% Because of the way meshgrids work in the surf command, we need to
	% transpose J_vals before calling surf, or else the axes will be flipped
	J_vals = J_vals';

![alt text](/Week_2/Octave_MatlabTutorials/Assets/CostFunctionSurface.png)

	% Surface plot
	figure;
	surf(theta0_vals, theta1_vals, J_vals)
	xlabel('\theta_0'); ylabel('\theta_1');

__Contour Plot:__
	
	figure;
	contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
	xlabel('\theta_0'); ylabel('\theta_1');
	hold on;
	plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

![alt text](/Week_2/Octave_MatlabTutorials/Assets/ContourGraph.png)
