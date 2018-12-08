function [Q,R] = tqr(T)
% Tensor orthogonal-triangular decomposition.
%
%  [Q,R] = tqr(T) where T is n1*n2*n3, R is n1*n2*n3 upper triangular tensorand Q is a n1*n1*n3 orthogonal tensor.
%
%@CREATE ON 9 28, 2018
%@AUTHOR HAILI
%

[n1,n2,n3] = size(T);
T = fft(T,[],3);

Q = zeros(n1,n2,n3);
R = zeros(n2,n2,n3);

for i = 1:n3
	[Q(:,:,i),R(:,:,i)] = qr(T(:,:,i)); 
end

Q = ifft(Q,[],3);
R = ifft(R,[],3);
