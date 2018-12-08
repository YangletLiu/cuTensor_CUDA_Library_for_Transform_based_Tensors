#!/bin/bash
nvidia-smi
set -x
touch mprod_time.txt
Am=100
An=100
Bn=100
i=0
while [ $i -lt 20 ]; do
  # echo "$Am $An $Bn $i"
   ./test $Am $An $Bn >> mprod_time.txt
   Am=`expr $Am + 100`
   An=`expr $An + 100`
   Bn=`expr $Bn + 100`
   i=`expr $i + 1`
done
exit 0
