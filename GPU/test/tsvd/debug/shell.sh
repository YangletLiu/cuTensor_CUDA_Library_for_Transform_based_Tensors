#!/bin/bash
set -x
nvidia-smi
touch result.txt
m=64
n=64
k=100
i=0
j=0
p=0
while [ $p -lt 2 ]; do
while [ $j -lt 3 ]; do
while [ $i -lt 20 ]; do
  # echo "$m $n $k"
   ./test $m $n $k batched  >> result.txt
   k=`expr $k + 100`
   i=`expr $i + 1`
done
   k=100
   m=`expr $m + $m`
   n=`expr $n + $n`
   i=0 
   j=`expr $j + 1`
done
   j=0 
   m=64
   n=64
   p=`expr $p + 1`
done
exit 0

