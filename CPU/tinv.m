function invT = tinv(t)

% tinv(t) is the inverse of the tensor t of size n*n*n3.


[n1,n2,n3] = size(t);
if n1 ~= n2
    error('Error using tinv. Tensor must be square.');
end

t = fft(t,[],3);
invT = zeros(n1,n2,n3);
I = eye(n1);

for i = 1 : n3
    invT(:,:,i) = t(:,:,i)\I;
end

invT = ifft(invT,[],3);

