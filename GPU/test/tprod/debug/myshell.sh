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
echo "based++++++++++++++++++++"  >> result.txt
while [ $p -lt 1 ]; do
while [ $j -lt 2 ]; do
while [ $i -lt 20 ]; do
   echo "$m $n $k"  >> result.txt
#   ./test $m $n $k based  >> result.txt
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
   p=0;

echo "streamed++++++++++++++++++++"  >> result.txt
while [ $p -lt 1 ]; do
while [ $j -lt 2 ]; do
while [ $i -lt 20 ]; do
   echo "$m $n $k" >> result.txt
  # ./test $m $n $k streamed  >> result.txt
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
  p=0;

echo "batched++++++++++++++++++++"  >> result.txt
while [ $p -lt 1 ]; do
while [ $j -lt 2 ]; do
while [ $i -lt 20 ]; do
#   echo "$m $n $k"  >> result.txt
   ./test $m $n $k batched  >> result.txt
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
  p=0;

exit 0
