#!/bin/bash

# Go up one directory
cd ..

for dataset in dunnhumby instacart tafeng; do
    for fold_id in 0 1 2; do
        python keyset_fold.py --dataset $dataset --fold_id $fold_id
    done
done
