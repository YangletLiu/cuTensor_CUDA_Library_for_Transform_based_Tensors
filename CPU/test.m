A = zeros(6,1)
for ii=1:1:6
    A(ii,1) = (ii-1) + (ii-1)*i
end

A = reshape(A', 3, 2)
[u, s, v] = svd(A)
