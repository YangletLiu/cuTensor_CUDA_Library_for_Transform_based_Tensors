function [ k ] = tem_svd( t )
%TEM_SVD Summary of this function goes here
%   Detailed explanation goes here
% Perform complex tensor svd.

[n1,n2,n3] = size(t);

% take fft along with third dimension

t1 = fft(t,[],3);

k = zeros(2*n1,2*n2,n3);

% construction of K
k(1:n1,1:n2,:) = real(t1);
k(1:n1,n2+1:2*n2,:) = imag(t1);
k(n1+1:2*n1,1:n2,:) = -imag(t1);
k(n1+1:2*n1,n2+1:2*n2,:) = real(t1);
