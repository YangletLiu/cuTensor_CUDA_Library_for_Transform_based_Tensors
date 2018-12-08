rng(42)
t_ar = zeros(1, 10)
result = zeros(1, 10)
for i=1:10
    N = 50*i
    A = rand(1, N*N*N);
    t1 = clock
    result(i) = norm(A)
    t2 = clock
    t_ar(i) = etime(t2, t1)
end
