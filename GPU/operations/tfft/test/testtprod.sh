#!/bin/bash
n=180
m=270
k=32
bat=batched
for (( i=1; i!=3; i+=1 ))
do
    for (( j=1; j!=5; j+=1 ))
    do
        ./test $m $n $k $n $m $k $bat >> out.txt
        let "k=k*2"
    done
    let "m=m*2"
    let "n=n*2"
done
