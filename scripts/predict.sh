python3 predict_imgcap.py \
--batch_size 4 \
--model_path $1 \
--val_max_target_length 60 \
--num_beams 3 \
--repetition_penalty 1.2 \
--no_repeat_ngram_size 2 \
--use_pretrain_imgcap