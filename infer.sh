python3 infer.py \
--path_to_train_data ../dataset/nf_prize_dataset/Netflix/N3M_TRAIN \
--path_to_eval_data ../dataset/nf_prize_dataset/Netflix/N3M_TEST \
--hidden_layers 512,512,1024 \
--non_linearity_type selu \
--save_path model_save/model.epoch_199 \
--drop_prob 0.8 \
--predictions_path model_save/preds.txt