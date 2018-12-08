function [time]=alter_min_LS(M,N,K)
% % parameter setting
% rng(234923);    % for reproducible results
m   = M; % 100      % the tensor is m * n * k
n   = N;
k = K;
r   = 5;        % the tubal-rank

% low-tubal-rank tensor
%T = rand(m,n,k);  %a ranom tensor: m * n * k
%T = t_svd_threshold(T,r);  %make it to be tubal-rank = r

T = tprod(rand(m,r,k), rand(r,n,k));
%[m,n,k] = size(T);

T_f = fft(T, [], 3);
% observations
p=0.05:0.05:0.25;
error = zeros(1,5);
T_omega = zeros(m,n,k);

for ii = 5:5
    omega = rand(m,n,k) <= 0.5;
    T_omega = omega .* T;

    T_omega_f = fft(T_omega,[],3);
    omega_f = fft(omega, [], 3);
% X: m * r * k
% Y: r * n * k
%% Given Y, do LS to get X
    Y = rand(r, n, k);
    %Y= init(T_omega, m,r,k);
    
    %[U, Theta, V]=t_svd(T_omega);
    %Y = V(1:r, :, :);
    
    Y_f = fft(Y, [], 3);

% do the transpose for each frontal slice
    Y_f_trans = zeros(n,r,k);
    X_f = zeros(m,r,k);
    T_omega_f_trans = zeros(n,m,k);
    omega_f_trans = zeros(n,m,k);
for i = 1: k
     Y_f_trans(:,:,i) = Y_f(:,:,i)';
     T_omega_f_trans(:,:,i) = T_omega_f(:,:,i)';
     omega_f_trans(:,:,i) = omega_f(:,:,i)';
end

iter=1;
start=clock;
while iter <=15
    fprintf('Sampling--%f---Round--#%d\n', p(ii), iter);
    [X_f_trans] = alter_min_LS_one_step(T_omega_f_trans, omega_f_trans * 1/k, Y_f_trans);
    
    for i =1:k
        X_f(:,:,i) = X_f_trans(:,:,i)';
    end

    % Given X, do LS to get Y
    [Y_f] = alter_min_LS_one_step(T_omega_f, omega_f * 1/k, X_f);
    
    for i = 1: k
    Y_f_trans(:,:,i) = Y_f(:,:,i)';
    end
    
    iter = iter + 1;
end
endt=clock;
time=etime(endt,start);

% The relative error:
temp = 0;
X_est = ifft(X_f, [], 3); 
Y_est = ifft(Y_f, [], 3);
T_est = tprod(X_est, Y_est);

temp = T - T_est;   
error(ii) = norm(temp(:)) / norm(T(:));
end
