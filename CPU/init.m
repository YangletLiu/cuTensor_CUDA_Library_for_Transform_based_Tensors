A = importdata('../data.txt')
B = reshape(A', 240, 180, 151)
[U, S, V] = t_svd(B)
