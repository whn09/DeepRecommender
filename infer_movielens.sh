python3 infer.py \
--path_to_train_data ../dataset/movielens/ml-20m/ml-20m.train \
--path_to_eval_data ../dataset/movielens/ml-20m/ml-20m.test \
--hidden_layers 512,512,1024 \
--non_linearity_type selu \
--save_path model_save_movielens/model.epoch_9 \
--drop_prob 0.8 \
--predictions_path model_save_movielens/preds.txt