time = zeros(20,1)
for i=1:20
    l = i*50
    A = rand(l, l, l)
    C = zeros(l, l, l)
    t1 = clock;
    for j=1:l
        C(:,:,j) = A(:,:,j)*A(:,:,j)
    end
    t2 = clock
    time(i) = etime(t2, t1)
end
