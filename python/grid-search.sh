#!/bin/bash

set -e
PYTHON_FILE="./main_grid_search.py"
echo "Will run the search space as following:"
total_loop=0
for l1 in f t b;
do
    for l2 in f t b;
    do
	for l3 in f t b;
	do
	    for l4 in f t b;
	    do
		for l5 in f t b;
		do
		    total_loop=$((total_loop+1))
		    echo "loop: $total_loop l1: $l1 l2: $l2 l3: $l3 l4: $l4 l5: $l5"
		done
	    done
	done
    done
done


read -p "Run over this search space? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
for l1 in f t b;
do
    for l2 in f t b;
    do
	for l3 in f t b;
	do
	    for l4 in f t b;
	    do
		for l5 in f t b;
		do
			progress=$((progress+1))
			echo "Current Progess: " $progress/$total_loop "***********************************"
			total_loop=$((total_loop+1))
			python $PYTHON_FILE -l1 $l1 -l2 $l2 -l3 $l3 -l4 $l4 -l5 $l5
		done
	    done
	done
    done
done
fi
