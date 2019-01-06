close all
clc

ft = fopen('./compress.txt','a+');
for k=[32 64]
for i=100:100:2000
    T = rand(i,i,k);
    t1 = clock;
%    [result] = compress(T);
    t2 = clock;
    time = etime(t2,t1);
    fprintf(ft,'%d %d %d %f\n',i,i,k,time);
end
end

fclose(ft);
