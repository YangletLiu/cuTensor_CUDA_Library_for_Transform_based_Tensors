close all
clear all
clc

ft = fopen('./ls_time.txt','a+');

for k=30
for i=100:100:500
    [time]=alter_min_LS(i,i,k); 
    fprintf(ft,'%d %d %d %f\n',i,i,k,time);
    
end
end

fclose(ft);
