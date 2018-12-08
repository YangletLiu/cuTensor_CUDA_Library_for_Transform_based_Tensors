A = rand(1000, 1000, 128);
t1 = clock;
[U, S, V] = t_svd(A);
t2 = clock;
t = etime(t2, t1);
disp(t);
