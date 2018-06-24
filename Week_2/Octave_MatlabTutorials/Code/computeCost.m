function J = computeCost(X, y, theta)

m = length(y); 

h = X * theta;
squaredErrors = (h - y) .^ 2;
J = sum(squaredErrors) / (2 * m);

end
