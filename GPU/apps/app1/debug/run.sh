#!/bin/bash
set -x
nvidia-smi
touch result.txt
m=100
n=100
k=30
i=0
j=0
while [ $j -lt 1 ]; do
while [ $i -lt 10 ]; do
#	echo "$m $n $k"
	./result $m $n $k >> result.txt
	m=`expr $m + 100`
	n=`expr $n + 100`
	i=`expr $i + 1`
done
	i=0;
	m=100;
	n=100;
	k=`expr $k + $k`
	j=`expr $j + 1`
done
exit 0

