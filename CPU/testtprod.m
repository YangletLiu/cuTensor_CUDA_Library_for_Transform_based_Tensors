close all
clear all
clc
k=32;

ft = fopen('./tprod_time.txt','a+');

for j=1:4
for i=100:100:2000
    A = rand(i, i, k);
    B = rand(i, i, k);
    t1 = clock;
    C = tprod(A, B);
    t2 = clock;
    time = etime(t2, t1);
   
    fprintf(ft,'%d %d %d %f\n',i,i,k,time);
    
end
     k=k+k;
end

fclose(ft);
