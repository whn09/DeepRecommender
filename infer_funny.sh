python3 infer.py \
--path_to_train_data ../dataset/log_preprocess/muid_contentid_train \
--path_to_eval_data ../dataset/log_preprocess/muid_contentid_test \
--hidden_layers 512,512,1024 \
--non_linearity_type selu \
--save_path model_save_funny/model.epoch_199 \
--drop_prob 0.8 \
--predictions_path model_save_funny/preds.txt