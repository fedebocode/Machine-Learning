function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);
cost = sum(-y .* log(h) - (1 - y) .* log(1 - h));
grad = X' * (h - y);

J = cost / m;
grad = grad / m;


end
