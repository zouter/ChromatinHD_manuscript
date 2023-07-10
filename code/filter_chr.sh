#!/bin/bash

chrs=($(for i in $(seq 1 23) X Y; do echo chr"$i";done))
 
awk -v var="${chrs[*]}" 'BEGIN{
    OFS="\t";split(var,list); for (i in list) chrs[list[i]]=""}{
        if($1 in chrs) print $0}END{for(i in ' $input_bed