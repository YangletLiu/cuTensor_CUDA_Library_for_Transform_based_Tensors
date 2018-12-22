#!/bin/bash
nvidia-smi
set -x
touch result.txt
Am=100
An=100
i=0
while [ $i -lt 20 ]; do
  # echo "$Am $An  $i"
   ./test $Am $An >> result.txt
   Am=`expr $Am + 100`
   An=`expr $An + 100`
   i=`expr $i + 1`
done
exit 0
