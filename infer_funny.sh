python3 infer.py \
--hidden_layers 128,256,256 \
--non_linearity_type selu \
--drop_prob 0.8 \
--path_to_train_data ../dataset/log_preprocess/train.txt \
--path_to_eval_data ../dataset/log_preprocess/test.txt \
--save_path model_save_funny/model.epoch_199 \
--predictions_path model_save_funny/preds.txt

#--path_to_train_data ../dataset/log_preprocess/train \
#--path_to_eval_data ../dataset/log_preprocess/test \
#--save_path model_save_funny/model.epoch_299 \
#--predictions_path model_save_funny/preds.txt

#--path_to_train_data ../dataset/log_preprocess/train_binary.txt \
#--path_to_eval_data ../dataset/log_preprocess/test_binary.txt \
#--save_path model_save_funny_binary/model.epoch_99 \
#--predictions_path model_save_funny_binary/preds.txt