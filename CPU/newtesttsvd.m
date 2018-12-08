n1=3;
n2=3;
n3=3;
a=randn(n1,n2,n3);
[u,s,v]=tsvd(a);
v=fft(v,[],3);

    
temp_v=zeros(n2,n2,n3);
for i=1:n3
temp_v(:,:,i)=v(:,:,i)';
end

temp_v=ifft(temp_v,[],3);

b = a-tprod(tprod(u,s),temp_v);
error = norm(b(:))/norm(a(:));

