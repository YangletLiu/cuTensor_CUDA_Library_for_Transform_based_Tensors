#!/bin/bash
i=200
bat=batched
str=streamed
while [ $i -le 2000 ]
do
    tfft/test $i $i 128 $bat >> tfft/outbatched.txt
    #tfft/test $i $i 128 $str >> tfft/outstreamed.txt
    let "i=i+100"
done
