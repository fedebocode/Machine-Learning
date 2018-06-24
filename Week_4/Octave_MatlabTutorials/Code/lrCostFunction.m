function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); 

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);
cost = sum(-y .* log(h) - (1 - y) .* log(1 - h));
grad = X' * (h - y);

grad_reg = lambda * theta;
grad_reg(1) = 0;
grad = grad + grad_reg;

J = cost / m + (lambda / (2.0 * m)) * sum(theta(2:size(theta)) .^ 2);
grad = grad / m;

grad = grad(:);

end
