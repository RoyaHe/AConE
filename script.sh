# NELL
CUDA_VISIBLE_DEVICES=1 python main.py --do_train --do_test \
--data_path ./data/NELL-betae -n 128 -b 512 -d 800 -g 20 -cenr 0.02  --data NELL \
-lr 0.0001 --max_steps 300001 --cpu_num 0 --valid_steps 60000 --test_batch_size 4 --log_steps 100 \
--drop 0.2 --tag train --save_checkpoint_steps 3000 

# WN18RR
CUDA_VISIBLE_DEVICES=3 python main.py --cuda --do_train --do_test\
--data_path ./data/WN18RR-QA -n 128 -b 512 -d 800 -g 20 -cenr 0.02  --data wn18rr \
-lr 0.0001 --max_steps 300001 --cpu_num 0 --valid_steps 60000 --test_batch_size 4 --log_steps 100 \
--drop 0.2 --tag train --save_checkpoint_steps 3000  