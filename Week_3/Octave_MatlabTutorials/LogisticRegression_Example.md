# Logistic Regression Example

In this exercise, you will implement logistic regression and apply it to two different datasets.

In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university.
Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions decision.
Your task is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams.


Initialize model:

	clear ; close all; clc

Load Data (the first two columns contains the exam scores and the third column contains the label):

	data = load('ex2data1.txt');
	X = data(:, [1, 2]); y = data(:, 3);
	plotData(X, y);

Put some labels:

	hold on;
	xlabel('Exam 1 score')
	ylabel('Exam 2 score')

Specified in plot order:

	legend('Admitted', 'Not admitted')
	hold off;

![alt text](/Week_3/Octave_MatlabTutorials/Assets/DataPlot.png)

Compute Cost Function and Gradient:

Setup the data matrix appropriately, and add ones for the intercept term:
	
	[m, n] = size(X);

Add intercept term to x and X_test:
	
	X = [ones(m, 1) X];

Initialize fitting parameters:

	initial_theta = zeros(n + 1, 1);

Compute and display initial cost and gradient:

	[cost, grad] = costFunction(initial_theta, X, y);

Compute and display cost and gradient with non-zero theta:

	test_theta = [-24; 0.2; 0.2];
	[cost, grad] = costFunction(test_theta, X, y);

_Optimization using __fminunc__ function_

Set options for fminunc:

	options = optimset('GradObj', 'on', 'MaxIter', 400);

Run fminunc to obtain the optimal theta (this function will return theta and the cost):

	[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

Plot Boundary:

	plotDecisionBoundary(theta, X, y);

Put some labels:

	hold on;
	xlabel('Exam 1 score')
	ylabel('Exam 2 score')

	legend('Admitted', 'Not admitted')
	hold off;

![alt text](/Week_3/Octave_MatlabTutorials/Assets/DecisionBoundary.png)

Predict and Accuracies:

	prob = sigmoid([1 45 85] * theta);

Compute accuracy on our training set:

	p = predict(theta, X);

	fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
	fprintf('Expected accuracy (approx): 89.0\n');
	fprintf('\n');

__Plot Data__

	function plotData(X, y)

		figure; hold on;

		pos = find(y==1); 
		neg = find(y == 0);

		plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...'MarkerSize', 7);
		plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...'MarkerSize', 7);

		hold off;

		end


__Cost Function__

	function [J, grad] = costFunction(theta, X, y)

		m = length(y);

		J = 0;
		grad = zeros(size(theta));

		h = sigmoid(X * theta);
		cost = sum(-y. * log(h) - (1 - y). * log(1 - h));
		grad = X' * (h - y);

		J = cost / m;
		grad = grad / m;


		end

__Plot Decision Boundary__

		function plotDecisionBoundary(theta, X, y)

		plotData(X(:,2:3), y);
		hold on

		if size(X, 2) <= 3
    	plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
    	plot_y = (-1./theta(3)). * (theta(2). * plot_x + theta(1));

    	plot(plot_x, plot_y)
    
    	legend('Admitted', 'Not admitted', 'Decision Boundary')
    	axis([30, 100, 30, 100])
	else
    	% Grid range
    	u = linspace(-1, 1.5, 50);
    	v = linspace(-1, 1.5, 50);

    	z = zeros(length(u), length(v));

    	for i = 1:length(u)												 //Evaluate z = theta * x over the grid
        	for j = 1:length(v)
            	z(i,j) = mapFeature(u(i), v(j)) * theta;
        	end
    	end

    	z = z'; 														//Important to transpose z before calling contour)

    	contour(u, v, z, [0, 0], 'LineWidth', 2)	   					//Plot z = 0. Notice you need to specify the range [0, 0]
	end
	hold off

	end

__Sigmoid Function__

	function g = sigmoid(z)

		g = 1 ./ (1 + e.^-z);

	end

__Predict__

Predict whether the label is _0_ or 1 using learned logistic regression parameters theta. 
Computes the predictions for _X_ using a threshold at 0.5 (i.e., if _sigmoid(theta'*x)_ >= 0.5, predict 1):

	function p = predict(theta, X)
		m = size(X, 1);
		p = zeros(m, 1);

		for i = 1 : m
    		p(i) = sigmoid(theta' * X(i,:)') >= 0.5;
		end;
		end