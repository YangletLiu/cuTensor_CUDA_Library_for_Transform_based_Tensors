time = zeros(20,1);
it = 1;
for i=200:100:2000
    l = i;
    A = rand(l, l, 128);
    t1 = clock;
    Af = fft(A, [], 3);
    A = ifft(Af, [], 3);
    t2 = clock;
    time(it) = etime(t2, t1);
    it = it+1;
end
for j=1:20
    fprintf('time %f\n', time(j));
end
