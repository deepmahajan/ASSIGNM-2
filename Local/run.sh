#!/bin/bash
# python3 lr_inmemory.py <dictionary_size> <initial_learning_rate> <regularization_param> <# of epochs> <# of examples per epoch>  <path of test_file>
for((i=1;i<=50;i++));
do shuf /scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt
done | python3 lr_inmemory.py 350000 1 0.0001 50 214997  /scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt
