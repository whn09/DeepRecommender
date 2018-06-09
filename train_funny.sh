python3 run.py --gpu_ids 0 \
--path_to_train_data ../dataset/log_preprocess/train \
--path_to_eval_data ../dataset/log_preprocess/test \
--logdir model_save_funny \
--hidden_layers 512,512,1024 \
--non_linearity_type selu \
--batch_size 128 \
--drop_prob 0.8 \
--optimizer momentum \
--lr 0.005 \
--weight_decay 0 \
--aug_step 1 \
--noise_prob 0 \
--num_epochs 50 \
--summary_frequency 1000


#--path_to_train_data ../dataset/log_preprocess/train \
#--path_to_eval_data ../dataset/log_preprocess/test \
#--logdir model_save_funny \

#--path_to_train_data ../dataset/log_preprocess/train_binary \
#--path_to_eval_data ../dataset/log_preprocess/test_binary \
#--logdir model_save_funny_binary \