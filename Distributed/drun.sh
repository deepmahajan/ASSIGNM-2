#!/bin/bash
for((i=1;i<=100;i++));
do shuf /user/ds222/assignment-1/DBPedia.full/full_train.txt
done | python3 dist_lr.py --ps_hosts=10.24.1.201:2221 --worker_hosts=10.24.1.202:2222,10.24.1.203:2222 --job_name=worker --task_index=0
