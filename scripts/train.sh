# for test only
python3 train_imgcap.py \
--batch_size 4 \
--num_epoches 50 \
--grad_accu_step 8 \
--learning_rate 5e-5 \
--optimizer AdamW \
--val_max_target_length 60

