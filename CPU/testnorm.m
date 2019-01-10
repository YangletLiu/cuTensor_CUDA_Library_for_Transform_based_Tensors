close all
clc

ft = fopen('./norm.txt','a+');
%for k=[32 64]
%for i=100:100:2000
    i=10000; 
    T = rand(i,1,i);
    t1 = clock;
    [v,a] = tnormlize(T);
    t2 = clock;
    time = etime(t2,t1);
    fprintf(ft,'%d  1 %d %f\n',i,i,time);
%end
%end

fclose(ft);
