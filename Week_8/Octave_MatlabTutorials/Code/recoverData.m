function X_rec = recoverData(Z, U, K)

X_rec = zeros(size(Z, 1), size(U, 1));

Ureduce = U(:, 1:K);

for i=1:size(Z, 1)
z = Z(i, :)';
x = Ureduce * z;
X_rec(i, :) = x';
end;

end
