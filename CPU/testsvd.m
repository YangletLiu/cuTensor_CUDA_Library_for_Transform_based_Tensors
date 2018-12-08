close all
clear all
clc

ft = fopen('./newtsvd_time.txt','a+');


for k=128:128:1024
for i=50:50:500
    T = randn(i, i, k);
    t1 = clock;
   // [U,S,V] = t_svd(T);
    t2 = clock;
    time = etime(t2, t1);
   
    fprintf(ft,'%d %d %d %f\n',i,i,k,time);
    
end
end

fclose(ft);
