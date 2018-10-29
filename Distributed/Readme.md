#### Running in local ###########
Make changes in the script run.sh to update 
1) dictionary_size e.g 350000
2) initial_learning_rate e.g 1
3) regularization_param e.g 0.0001
4) # of epochs e.g 50
5) examples per epoch (run wc -l <training_file_path>)
6) path of test_file (<path_of_test_file>)

and then run using command ./run.sh


#### Running in distributed setting ###########
Make changes in drun.sh to update the ips of machines acting as "workers"(comma separated) and ips of machines acting as "servers"(comma separated) and update whether the current machine is "worker" or "server" and also edit it's "task_index"(0 based)

For e.g on machine which is a "server" the drun.sh should look like the following :-
for((i=1;i<=100;i++));
do shuf /user/ds222/assignment-1/DBPedia.full/full_train.txt
done | python3 dist_lr.py --ps_hosts=10.24.1.201:2221 --worker_hosts=10.24.1.202:2222,10.24.1.203:2222 --job_name=ps --task_index=0

On machine which is a "worker" the drun.sh should look like the following :-
for((i=1;i<=100;i++));
do shuf /user/ds222/assignment-1/DBPedia.full/full_train.txt
done | python3 dist_lr.py --ps_hosts=10.24.1.201:2221 --worker_hosts=10.24.1.202:2222,10.24.1.203:2222 --job_name=worker --task_index=0
