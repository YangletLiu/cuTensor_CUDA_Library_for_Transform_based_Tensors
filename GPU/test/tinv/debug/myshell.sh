#!/bin/bash
set -x
nvidia-smi
touch result.txt
m=100
n=100
k=128
i=0
j=0
p=0
while [ $p -lt 2 ]; do
while [ $j -lt 3 ]; do
while [ $i -lt 20 ]; do
   echo "$m $n $k"
  # ./test $m $n $k batched  >> result.txt
   m=`expr $m + 100`
   n=`expr $n + 100`
   i=`expr $i + 1`
done
   k=`expr $k + $k`
   m=100
   n=100
   i=0
   j=`expr $j + 1`
done
   j=0
   k=128
   p=`expr $p + 1`
done
exit 0
