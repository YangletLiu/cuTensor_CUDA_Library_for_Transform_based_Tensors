function [v,a] = tnormlize(t)
% the normlization of tnesor t:
%       t = v * a;
% ||v|| = 1,it's mean <v,v> = e.
% INPUT: t is m×1×n
%
% OUTPUT: v*a  = t.
%
[m,~,n] = size(t);

a = zeros(1,1,n);

tol = 1.e-128;

v = fft(t,[],3);

for j = 1 : n 

	a(1,1,j) = norm(v(:,:,j));

	if a(1,1,j) > tol

	    v(:,:,j) = v(:,:,j)/a(1,1,j);

    else

	    v(:,:,j) = randn(m,1);

	    a(1,1,j) = norm(v(:,:,j));

	    v(:,:,j) = v(:,:,j)/a(1,1,j);

	    a(1,1,j) = 0;
	end 
end

v = ifft(v,[],3);

a = ifft(a,[],3);

