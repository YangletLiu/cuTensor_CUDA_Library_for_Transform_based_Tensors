#!/bin/bash
i=200
bat=batched
str=streamed
while [ $i -le 2000 ]
do
    tprod/test $i $i 512 $i $i 512 $bat >> tprod/outbatched.txt
    tprod/test $i $i 512 $i $i 512 $str >> tprod/outbatched.txt
    let "i=i+100"
done
