function J = computeCostMulti(X, y, theta)

m = length(y);

cost = 0;
for i = 1:m
    cost = cost + (theta' * X(i,:)' - y(i))^2;
end;

J = cost / (2 * m);

end
