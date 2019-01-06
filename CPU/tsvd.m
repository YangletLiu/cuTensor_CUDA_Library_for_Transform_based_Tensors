function [ U, S, V ] = tsvd( t )
% Perform complex tensor svd.

[n1,n2,n3] = size(t);

% take fft along with third dimension

t = fft(t,[],3);

k = zeros(2*n1,2*n2,n3);

U = zeros(n1,n1,n3);
S = zeros(n1,n2,n3);
V = zeros(n2,n2,n3);

% construction of K
k(1:n1,1:n2,:) = real(t);
k(1:n1,n2+1:2*n2,:) = imag(t);
k(n1+1:2*n1,1:n2,:) = -imag(t);
k(n1+1:2*n1,n2+1:2*n2,:) = real(t);
% take svd 
for i = 1:n3
    [u,s,v] = svd( k(:,:,i));

    U(:,:,i) = 1i*u(1:n1,1:2:2*n1) + u(n1+1:2*n1,1:2:2*n1);
    S(:,:,i) = s(1:2:2*n1,1:2:2*n2);
    V(:,:,i) = 1i*v(1:n2,1:2:2*n2) + v(n2+1:2*n2,1:2:2*n2);
end


% take ifft along with third dimension
U = ifft(U,[],3);
S = ifft(S,[],3);
V = ifft(V,[],3);

