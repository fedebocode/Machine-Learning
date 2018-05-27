# Regularized Logistic Regression Example

In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.
Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

Initialize model:

	clear ; close all; clc

Load Data (The first two columns contains the X values and the third column contains the label (y)).

	data = load('ex2data2.txt');
	X = data(:, [1, 2]); y = data(:, 3);

	plotData(X, y);

Put some labels:
	
	hold on;
	xlabel('Microchip Test 1')
	ylabel('Microchip Test 2')
	legend('y = 1', 'y = 0')
	hold off;

![alt text](/Week_3/Octave_MatlabTutorials/Assets/RegularizedLogisticRegressionDataPlot.png)

You are given a dataset with data points that are not linearly separable. However, you would still like to use logistic regression to classify the data points. To do so, you introduce more features to use; in particular, you add polynomial features to our data matrix (similar to polynomial regression).

Add Polynomial Features (note that mapFeature also adds a column of ones for us, so the intercept term is handled):

	X = mapFeature(X(:,1), X(:,2));

Initialize fitting parameters:

	initial_theta = zeros(size(X, 2), 1);

Set regularization parameter lambda to 1:

	lambda = 1;

Compute and display initial cost and gradient for regularized logistic regression:

	[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

Compute and display cost and gradient with all-ones theta and lambda = 10:

	test_theta = ones(size(X,2),1);
	[cost, grad] = costFunctionReg(test_theta, X, y, 10);

Initialize fitting parameters:

	initial_theta = zeros(size(X, 2), 1);

Set regularization parameter lambda to 1 (you should vary this):

	lambda = 1;

Set Options:
	
	options = optimset('GradObj', 'on', 'MaxIter', 400);

Optimize:

	[theta, J, exit_flag] = ...
		fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

Plot Boundary:

	plotDecisionBoundary(theta, X, y);

	hold on;
	title(sprintf('lambda = %g', lambda))
	xlabel('Microchip Test 1')
	ylabel('Microchip Test 2')
	legend('y = 1', 'y = 0', 'Decision boundary')
	hold off;

![alt text](/Week_3/Octave_MatlabTutorials/Assets/RegularizedLogisticRegressionDataPlot2.png)

Compute accuracy on our training set:
	
	p = predict(theta, X);

	fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
	fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');

__Map Feature__

Returns a new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
Inputs X1, X2 must be the same size

	function out = mapFeature(X1, X2)

		degree = 6;
		out = ones(size(X1(:,1)));
		for i = 1:degree
    		for j = 0:i
        		out(:, end+1) = (X1.^(i-j)). * (X2.^j);
    		end
		end
		end

__Cost Function Regularized__

	function [J, grad] = costFunctionReg(theta, X, y, lambda)

	m = length(y); 
	J = 0;
	grad = zeros(size(theta));

	h = sigmoid(X * theta);
	cost = sum(-y. * log(h) - (1 - y). * log(1 - h));
	grad = X' * (h - y);

	grad_reg = lambda * theta;
	grad_reg(1) = 0;
	grad = grad + grad_reg;

	J = cost / m + (lambda / (2.0 * m)) * sum(theta(2:size(theta)).^ 2);
	grad = grad / m;

	end





