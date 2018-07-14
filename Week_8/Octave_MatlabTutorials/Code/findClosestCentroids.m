function idx = findClosestCentroids(X, centroids)

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

for i=1:size(X, 1)
    distances = zeros(size(centroids, 1), 1);
    for k=1:K
        distances(k) = sumsq(X(i,:) - centroids(k, :));
    end;
    [minDistance, minIndex] = min(distances);
    idx(i) = minIndex;
end;

end

