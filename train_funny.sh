python3 run.py --gpu_ids 0 \
--path_to_train_data ../dataset/log_preprocess/train.txt \
--path_to_eval_data ../dataset/log_preprocess/test.txt \
--logdir model_save_funny_20180623 \
--hidden_layers 128,256,256 \
--non_linearity_type selu \
--batch_size 64 \
--drop_prob 0.8 \
--optimizer momentum \
--lr 0.01 \
--weight_decay 0 \
--aug_step 1 \
--noise_prob 0 \
--num_epochs 40 \
--summary_frequency 1000


#--path_to_train_data ../dataset/log_preprocess/train.txt \
#--path_to_eval_data ../dataset/log_preprocess/test.txt \
#--logdir model_save_funny \

#--path_to_train_data ../dataset/log_preprocess/train_binary.txt \
#--path_to_eval_data ../dataset/log_preprocess/test_binary.txt \
#--logdir model_save_funny_binary \
