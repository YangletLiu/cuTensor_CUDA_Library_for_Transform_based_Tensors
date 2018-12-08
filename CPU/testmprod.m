close all
clear all
clc

ft = fopen('./mtprod_time.txt','a+');

for i=100:100:2000
    A = rand(i, i);
    B = rand(i, i);
    t1 = clock;
    C = mprod(A, B);
    t2 = clock;
    time = etime(t2, t1);
   
    fprintf(ft,'%d %d %f\n',i,i,time);
end

fclose(ft);
