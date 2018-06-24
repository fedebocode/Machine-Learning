function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
N = size(X, 2);                                % number of features

mu = mean(X);
sigma = std(X);

for i = 1:N
    X_norm(:,i) = (X(:, i) - mu(i)) / sigma(i);
end;

