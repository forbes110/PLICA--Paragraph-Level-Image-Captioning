# for test only
python3 train_imgcap.py \
--batch_size 4 \
--num_epoches 20 \
--grad_accu_step 8 \
--learning_rate 5e-5 \
--optimizer AdamW \
--model_path $1 \
--val_max_target_length 60 \
--use_pretrain_imgcap

