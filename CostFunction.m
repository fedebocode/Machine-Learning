function J = CostFunction(X,y,theta)

% X is the "design matrix" containing our training examples
% y is the class labels
% theta is the values for which we want to solve teh cost function

m = size(X,1);							% Number of training examples
predictions = X * theta;				% Predictions of hypothesis on all m
squaredErrors = (predictions - y).^2;	% Squared errors 

J = 1/(2 * m) * sum(squaredErrors);