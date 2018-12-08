fid = fopen('compress.txt', 'w')
for k=1:151
    for i= 1:240
        for j=1:180
            fprintf(fid, '%d ', USVc(i, j, k))
        end
        fprintf(fid, '\n')
    end
end
fclose(fid)
