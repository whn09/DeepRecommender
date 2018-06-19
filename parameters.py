import os

#Funny data
DATA_ROOT = '/media/henan.wang/workspace/dataset/log_preprocess'
TRAIN = os.path.join(DATA_ROOT, 'train.txt')
EVAL = os.path.join(DATA_ROOT, 'test.txt')
TEST = os.path.join(DATA_ROOT, 'test.txt')

#Autoencoder parameters
GPUS = 0
USE_GPU = False
ACTIVATION = 'selu'
OPTIMIZER = 'momentum'
HIDDEN = '128,256,256'
BATCH_SIZE = 64
DROPOUT = 0.8
LR = 0.01
WD = 0
EPOCHS = 100
AUG_STEP = 1
MODEL_OUTPUT_DIR = 'model_save_funny'

#Evaluation
INFER_OUTPUT = os.path.join(MODEL_OUTPUT_DIR, 'preds.txt')
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model.epoch_' + str(EPOCHS-1))
MOVIE_TITLES = os.path.join(DATA_ROOT,'contentid_titles.txt')




