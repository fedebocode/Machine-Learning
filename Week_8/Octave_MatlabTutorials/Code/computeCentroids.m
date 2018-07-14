function centroids = computeCentroids(X, idx, K)

[m n] = size(X);
centroids = zeros(K, n);

for k=1:K
    % use logical arrays for indexing
    % see http://www.mathworks.com/help/matlab/math/matrix-indexing.html#bq7egb6-1
    indexes = idx == k;
    centroids(k, :) = mean(X(indexes, :));
end;

end

